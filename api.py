"""
WC 2026 Match Predictor — Flask Backend API
============================================
Endpoints:
  GET  /api/health
  GET  /api/tournament-data
  GET  /api/team-path/<team>
  POST /api/predict-match
"""

import os, json, logging
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

from flask import Flask, jsonify, request
from flask_cors import CORS

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH   = os.path.join(MODELS_DIR, "wc2026_predictions.csv")

# ─── Name standardisation (mirrors training script) ───────────
NAME_MAP = {
    # Defunct states
    "German DR": "Germany", "East Germany": "Germany", "West Germany": "Germany",
    "USSR": "Russia", "Soviet Union": "Russia", "CIS": "Russia",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia", "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Netherlands Antilles": "Curaçao",
    "Swaziland": "Eswatini",
    "Macedonia": "North Macedonia",
    "Irish Free State": "Republic of Ireland",
    "Éire": "Republic of Ireland",
    "Zaire": "DR Congo", "Zaïre": "DR Congo",
    "Western Samoa": "Samoa",
    "Burma": "Myanmar",
    # Common alternate spellings / short forms
    "Holland": "Netherlands",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Ivory Coast": "Ivory Coast",
    "USA": "United States",
    "US": "United States",
    "UAE": "UAE",
    "IR Iran": "Iran",
    "Czech Republic": "Czech Republic",
}

CONFEDERATION_STRENGTH = {
    "UEFA": 1.08, "CONMEBOL": 1.06, "CONCACAF": 0.96,
    "CAF": 0.94,  "AFC": 0.93,      "OFC": 0.88,
}

TEAM_CONFEDERATION = {
    **{t: "UEFA" for t in [
        "Germany","France","Spain","Italy","England","Portugal","Netherlands",
        "Belgium","Croatia","Switzerland","Poland","Denmark","Sweden","Austria",
        "Serbia","Ukraine","Czech Republic","Hungary","Slovakia","Romania",
        "Turkey","Greece","Scotland","Norway","Finland","Wales",
        "Bosnia and Herzegovina","Slovenia","North Macedonia","Albania","Kosovo",
        "Montenegro","Iceland","Republic of Ireland","Northern Ireland","Belarus",
        "Bulgaria","Russia","Georgia","Armenia","Azerbaijan","Kazakhstan",
        "Estonia","Latvia","Lithuania","Malta","Luxembourg","San Marino",
        "Andorra","Moldova","Cyprus",
    ]},
    **{t: "CONMEBOL" for t in [
        "Brazil","Argentina","Uruguay","Colombia","Chile","Ecuador","Peru",
        "Paraguay","Bolivia","Venezuela",
    ]},
    **{t: "CONCACAF" for t in [
        "United States","Mexico","Canada","Costa Rica","Honduras","Jamaica",
        "Panama","El Salvador","Trinidad and Tobago","Haiti","Guatemala",
        "Cuba","Curaçao","Suriname","Guyana","Belize","Nicaragua",
        "Dominican Republic","Puerto Rico","Barbados","Grenada",
        "Saint Kitts and Nevis","Saint Lucia","Martinique","Guadeloupe",
    ]},
    **{t: "CAF" for t in [
        "Senegal","Morocco","Egypt","Nigeria","Cameroon","Tunisia","Ghana",
        "Algeria","Ivory Coast","Mali","Burkina Faso","Guinea","Congo",
        "DR Congo","Zimbabwe","Zambia","Uganda","Tanzania","Kenya","Ethiopia",
        "Cape Verde","Benin","Gabon","Rwanda","Mozambique","Madagascar",
        "Guinea-Bissau","Equatorial Guinea","Mauritania","Sudan","South Africa",
        "Angola","Togo","Niger","Sierra Leone","Liberia","Djibouti","Eswatini",
        "Malawi","Namibia","Botswana","Lesotho","Central African Republic",
        "South Sudan","Somalia",
    ]},
    **{t: "AFC" for t in [
        "Japan","South Korea","Iran","Australia","Saudi Arabia","Qatar","UAE",
        "Iraq","China","Uzbekistan","Jordan","Oman","Bahrain","Kuwait","Vietnam",
        "Thailand","India","Indonesia","Malaysia","Philippines","Myanmar",
        "North Korea","Syria","Lebanon","Palestine","Yemen","Tajikistan",
        "Kyrgyzstan","Nepal","Sri Lanka","Bangladesh","Pakistan","Afghanistan",
        "Mongolia","Maldives","Bhutan","Guam","Hong Kong","Singapore",
        "Chinese Taipei","Macau",
    ]},
    **{t: "OFC" for t in [
        "New Zealand","Fiji","Papua New Guinea","Vanuatu","Solomon Islands",
        "New Caledonia","Tahiti","American Samoa","Samoa","Tonga",
        "Cook Islands","Mauritius",
    ]},
}

def standardise(name: str) -> str:
    """Resolve historical / alternate team names to current canonical names."""
    return NAME_MAP.get(name, name)

def conf_multiplier(team: str) -> float:
    conf = TEAM_CONFEDERATION.get(team, None)
    return CONFEDERATION_STRENGTH.get(conf, 1.0)

# ─── Model state ──────────────────────────────────────────────
_models: dict = {}
_tournament_df: pd.DataFrame | None = None
_loaded_at: str | None = None

def load_models() -> bool:
    """Load all artifacts from /models directory. Returns True if successful."""
    global _models, _tournament_df, _loaded_at
    try:
        required = ["best_classifier.pkl", "poisson_home.pkl",
                    "poisson_away.pkl", "label_encoder.pkl"]

        missing = [f for f in required
                   if not os.path.exists(os.path.join(MODELS_DIR, f))]
        if missing:
            log.warning(f"Missing model files: {missing} — running in CSV-only mode.")
        else:
            _models["classifier"]    = joblib.load(os.path.join(MODELS_DIR, "best_classifier.pkl"))
            _models["poisson_home"]  = joblib.load(os.path.join(MODELS_DIR, "poisson_home.pkl"))
            _models["poisson_away"]  = joblib.load(os.path.join(MODELS_DIR, "poisson_away.pkl"))
            _models["label_encoder"] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

            feat_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
            _models["feature_cols"] = (joblib.load(feat_path) if os.path.exists(feat_path)
                                       else ["elo_diff","rest_diff","home_form_avg","away_form_avg",
                                             "home_gd_last10","away_gd_last10",
                                             "home_conf_mult","away_conf_mult"])
            log.info("All model artifacts loaded.")

        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            # Normalise column names to lowercase / short form
            col_map = {
                "Team": "team",
                "Round of 32": "r32", "Round of 16": "r16",
                "Quarter-final": "qf", "Semi-final": "sf",
                "Final": "final", "Champion": "champ",
                # Allow alternate column names from Kaggle / other exports
                "r32": "r32", "r16": "r16", "qf": "qf",
                "sf": "sf", "final": "final", "champ": "champ",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                      inplace=True)

            # Strip "%" and cast to float
            pct_cols = [c for c in ["r32","r16","qf","sf","final","champ",
                                    "group_winner_prob"] if c in df.columns]
            for col in pct_cols:
                if df[col].dtype == object:
                    df[col] = df[col].str.rstrip("%").astype(float)

            # Add group_winner_prob if missing (use r32 as proxy)
            if "group_winner_prob" not in df.columns and "r32" in df.columns:
                df["group_winner_prob"] = df["r32"] / 2   # rough heuristic

            _tournament_df = df
            log.info(f"Tournament CSV loaded: {len(df)} teams.")
        else:
            log.warning(f"No CSV found at {CSV_PATH}.")

        _loaded_at = datetime.utcnow().isoformat() + "Z"
        return True

    except Exception as exc:
        log.error(f"Error loading models: {exc}", exc_info=True)
        return False


def _build_feature_row(team1: str, team2: str) -> np.ndarray:
    """
    Build a feature vector for a head-to-head prediction.
    Uses neutral / mean values for stats we don't have in real-time.
    team1 is treated as 'home' (advantage=0 since neutral).
    """
    # ELO difference: we don't have live ELO here, so use 0 (neutral)
    elo_diff       = 0.0
    rest_diff      = 0.0
    home_form_avg  = 0.5   # mean win-rate
    away_form_avg  = 0.5
    home_gd_last10 = 0.0
    away_gd_last10 = 0.0
    home_conf_mult = conf_multiplier(team1)
    away_conf_mult = conf_multiplier(team2)

    # If we have the simulation CSV, use champ% difference as a rough proxy for quality
    if _tournament_df is not None:
        t1_row = _tournament_df[_tournament_df["team"] == team1]
        t2_row = _tournament_df[_tournament_df["team"] == team2]
        if not t1_row.empty and not t2_row.empty:
            # Use champion probability ratio to proxy ELO difference
            c1 = float(t1_row["champ"].iloc[0]) + 0.1   # avoid div/0
            c2 = float(t2_row["champ"].iloc[0]) + 0.1
            # Map to rough ELO scale: 100 ELO pts ≈ 64% expected score
            elo_diff = (c1 - c2) * 15   # heuristic scaling

    feat_map = {
        "elo_diff":       elo_diff,
        "rest_diff":      rest_diff,
        "home_form_avg":  home_form_avg,
        "away_form_avg":  away_form_avg,
        "home_gd_last10": home_gd_last10,
        "away_gd_last10": away_gd_last10,
        "home_conf_mult": home_conf_mult,
        "away_conf_mult": away_conf_mult,
    }

    cols = _models.get("feature_cols", list(feat_map.keys()))
    return np.array([[feat_map.get(c, 0.0) for c in cols]])


def _fallback_predict(team1: str, team2: str) -> dict:
    """
    Pure ELO/CSV-based prediction when ML models aren't loaded.
    Uses champion probabilities as a quality proxy.
    """
    t1_champ, t2_champ = 0.05, 0.05   # default equal
    if _tournament_df is not None:
        r1 = _tournament_df[_tournament_df["team"] == team1]
        r2 = _tournament_df[_tournament_df["team"] == team2]
        if not r1.empty: t1_champ = float(r1["champ"].iloc[0]) / 100 + 0.001
        if not r2.empty: t2_champ = float(r2["champ"].iloc[0]) / 100 + 0.001

    total = t1_champ + t2_champ + 0.25          # 0.25 reserved for draw
    p1    = round(t1_champ / total * 100, 1)
    p2    = round(t2_champ / total * 100, 1)
    pd_   = round(100 - p1 - p2, 1)

    winner = team1 if p1 > p2 else team2
    conf   = round(max(p1, p2), 1)

    exp_h = round(1.3 + (t1_champ - t2_champ) * 5, 2)
    exp_a = round(1.3 - (t1_champ - t2_champ) * 5, 2)

    return {
        "team1": team1, "team2": team2,
        "team1_win": p1, "draw": pd_, "team2_win": p2,
        "winner": winner, "confidence": conf,
        "expected_goals_team1": max(0.3, exp_h),
        "expected_goals_team2": max(0.3, exp_a),
        "model_used": "fallback_csv",
    }


# ─── Load on startup ──────────────────────────────────────────
load_models()


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": bool(_models),
        "csv_loaded": _tournament_df is not None,
        "teams_count": len(_tournament_df) if _tournament_df is not None else 0,
        "loaded_at": _loaded_at,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/tournament-data", methods=["GET"])
def tournament_data():
    if _tournament_df is None:
        return jsonify({"error": "Tournament data not loaded"}), 503

    df = _tournament_df.copy()

    # Top 10 by champion probability
    top10 = (df.nlargest(10, "champ")
               [["team","champ"]]
               .rename(columns={"champ":"champion_prob"})
               .to_dict(orient="records"))

    # All teams full data
    stage_cols = [c for c in ["r32","r16","qf","sf","final","champ","group_winner_prob"]
                  if c in df.columns]
    all_teams = df[["team"] + stage_cols].to_dict(orient="records")

    # Champion probabilities dict
    champ_dict = dict(zip(df["team"], df["champ"].round(2)))

    # Group winner probabilities dict
    gw_col = "group_winner_prob" if "group_winner_prob" in df.columns else "r32"
    gw_dict = dict(zip(df["team"], df[gw_col].round(2)))

    return jsonify({
        "top10":                    top10,
        "all_teams":                all_teams,
        "champion_probabilities":   champ_dict,
        "group_winner_probabilities": gw_dict,
        "last_updated":             _loaded_at,
        "teams_count":              len(df),
    })


@app.route("/api/team-path/<path:team_name>", methods=["GET"])
def team_path(team_name: str):
    if _tournament_df is None:
        return jsonify({"error": "Tournament data not loaded"}), 503

    team = standardise(team_name)
    row  = _tournament_df[_tournament_df["team"].str.lower() == team.lower()]
    if row.empty:
        # Try partial match
        row = _tournament_df[_tournament_df["team"].str.lower().str.contains(
            team.lower(), na=False)]
    if row.empty:
        return jsonify({"error": f"Team '{team_name}' not found"}), 404

    r = row.iloc[0]
    payload = {
        "team":              r["team"],
        "r32":               float(r["r32"])               if "r32"               in r.index else None,
        "r16":               float(r["r16"])               if "r16"               in r.index else None,
        "qf":                float(r["qf"])                if "qf"                in r.index else None,
        "sf":                float(r["sf"])                if "sf"                in r.index else None,
        "final":             float(r["final"])             if "final"             in r.index else None,
        "champ":             float(r["champ"])             if "champ"             in r.index else None,
        "group_winner_prob": float(r["group_winner_prob"]) if "group_winner_prob" in r.index else None,
    }
    return jsonify(payload)


@app.route("/api/predict-match", methods=["POST"])
def predict_match():
    body = request.get_json(silent=True) or {}
    raw1 = body.get("team1", "").strip()
    raw2 = body.get("team2", "").strip()

    if not raw1 or not raw2:
        return jsonify({"error": "Both team1 and team2 are required"}), 400

    team1 = standardise(raw1)
    team2 = standardise(raw2)

    if team1 == team2:
        return jsonify({"error": "Teams must be different"}), 400

    # ── Use ML models if available ────────────────────────────
    if "classifier" in _models:
        try:
            feat = _build_feature_row(team1, team2)
            clf  = _models["classifier"]
            proba = clf.predict_proba(feat)[0]   # [away_win, draw, home_win]

            # Map class indices; classes_ order may vary
            classes = list(clf.classes_)
            # 0 = away win, 1 = draw, 2 = home win
            p_t1  = round(float(proba[classes.index(2)]) * 100, 1)
            p_dr  = round(float(proba[classes.index(1)]) * 100, 1)
            p_t2  = round(float(proba[classes.index(0)]) * 100, 1)

            # Poisson goal predictions
            ph = _models["poisson_home"]
            pa = _models["poisson_away"]
            exp_h = round(float(ph.predict(feat)[0]), 2)
            exp_a = round(float(pa.predict(feat)[0]), 2)

            winner = team1 if p_t1 > p_t2 else (team2 if p_t2 > p_t1 else "Draw")
            conf   = round(max(p_t1, p_t2), 1)

            return jsonify({
                "team1": team1, "team2": team2,
                "team1_win": p_t1, "draw": p_dr, "team2_win": p_t2,
                "winner": winner, "confidence": conf,
                "expected_goals_team1": max(0.3, exp_h),
                "expected_goals_team2": max(0.3, exp_a),
                "model_used": "ml_ensemble",
            })

        except Exception as exc:
            log.error(f"ML prediction failed: {exc}", exc_info=True)
            # Fall through to fallback

    # ── Fallback: CSV-based probability estimate ───────────────
    result = _fallback_predict(team1, team2)
    return jsonify(result)


# ─── 404 / Error handlers ─────────────────────────────────────
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
