"""
WC 2026 Match Predictor — Flask Backend API
============================================
Endpoints:
  GET  /api/health
  GET  /api/tournament-data
  GET  /api/team-path/<team>
  POST /api/predict-match
"""

import os, logging
from datetime import datetime

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

# ─── Robust path detection ────────────────────────────────────
# Works whether run as: python api.py  OR  gunicorn api:app  OR in Jupyter
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()   # Jupyter fallback

# Search for the models folder in multiple likely locations
def _find_models_dir() -> str:
    candidates = [
        os.path.join(_HERE, "models"),           # ./models/  (standard)
        os.path.join(_HERE, "..", "models"),      # ../models/
        os.path.join(os.getcwd(), "models"),      # cwd/models/
        "/app/models",                            # Docker / Render absolute
        "/opt/render/project/src/models",         # Render build path
        _HERE,                                    # files sitting next to api.py
        os.getcwd(),                              # files in cwd
    ]
    for path in candidates:
        path = os.path.normpath(path)
        # Consider a directory valid if it contains the CSV or any .pkl
        if os.path.isdir(path):
            contents = os.listdir(path)
            if any(f.endswith(".pkl") or f.endswith(".csv") for f in contents):
                log.info(f"Models directory found: {path}  contents={contents}")
                return path
    # Fallback — return the default even if empty
    default = os.path.join(_HERE, "models")
    log.warning(f"No models directory found with files. Defaulting to: {default}")
    return default

MODELS_DIR = _find_models_dir()
CSV_PATH   = os.path.join(MODELS_DIR, "wc2026_predictions.csv")

log.info(f"BASE_DIR={_HERE}")
log.info(f"MODELS_DIR={MODELS_DIR}")
log.info(f"CSV_PATH={CSV_PATH}  exists={os.path.exists(CSV_PATH)}")

# ─── Name standardisation ─────────────────────────────────────
NAME_MAP = {
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
    "Holland": "Netherlands",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "US": "United States",
    "IR Iran": "Iran",
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
    return NAME_MAP.get(name, name)

def conf_multiplier(team: str) -> float:
    conf = TEAM_CONFEDERATION.get(team)
    return CONFEDERATION_STRENGTH.get(conf, 1.0) if conf else 1.0

# ─── Model state ──────────────────────────────────────────────
_models: dict = {}
_tournament_df = None
_loaded_at: str | None = None
_load_errors: list = []

def load_models() -> bool:
    global _models, _tournament_df, _loaded_at, _load_errors
    _load_errors = []

    # ── 1. Load ML model files ─────────────────────────────────
    pkl_files = {
        "classifier":    "best_classifier.pkl",
        "poisson_home":  "poisson_home.pkl",
        "poisson_away":  "poisson_away.pkl",
        "label_encoder": "label_encoder.pkl",
    }
    missing = []
    for key, fname in pkl_files.items():
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            try:
                _models[key] = joblib.load(fpath)
                log.info(f"Loaded {fname}")
            except Exception as e:
                err = f"Failed to load {fname}: {e}"
                log.error(err)
                _load_errors.append(err)
        else:
            missing.append(fname)

    if missing:
        msg = f"Missing model files (CSV-only mode): {missing}"
        log.warning(msg)
        _load_errors.append(msg)
    else:
        # Optional feature_cols.pkl
        fc_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
        if os.path.exists(fc_path):
            try:
                _models["feature_cols"] = joblib.load(fc_path)
                log.info("Loaded feature_cols.pkl")
            except Exception as e:
                log.warning(f"Could not load feature_cols.pkl: {e}")

        if "feature_cols" not in _models:
            _models["feature_cols"] = [
                "elo_diff", "rest_diff", "home_form_avg", "away_form_avg",
                "home_gd_last10", "away_gd_last10", "home_conf_mult", "away_conf_mult"
            ]

    # ── 2. Load tournament CSV ─────────────────────────────────
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            log.info(f"CSV loaded: {CSV_PATH}  shape={df.shape}  cols={list(df.columns)}")

            # Normalise column names
            col_map = {
                "Team": "team",
                "Round of 32": "r32",
                "Round of 16": "r16",
                "Quarter-final": "qf",
                "Semi-final": "sf",
                "Final": "final",
                "Champion": "champ",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                      inplace=True)

            # Strip "%" and cast to float for any column that needs it
            for col in ["r32","r16","qf","sf","final","champ","group_winner_prob"]:
                if col in df.columns and df[col].dtype == object:
                    df[col] = df[col].str.rstrip("%").astype(float)

            # Add group_winner_prob column if absent
            if "group_winner_prob" not in df.columns and "r32" in df.columns:
                df["group_winner_prob"] = (df["r32"] / 2).round(1)

            _tournament_df = df
            log.info(f"Tournament data ready: {len(df)} teams.")
        except Exception as e:
            err = f"Failed to load CSV: {e}"
            log.error(err)
            _load_errors.append(err)
    else:
        err = f"CSV not found at {CSV_PATH}. Directory listing: {os.listdir(MODELS_DIR) if os.path.isdir(MODELS_DIR) else 'DIR MISSING'}"
        log.error(err)
        _load_errors.append(err)

    _loaded_at = datetime.utcnow().isoformat() + "Z"
    return _tournament_df is not None


def _build_feature_row(team1: str, team2: str) -> np.ndarray:
    """Build feature vector — uses CSV champion% as ELO proxy when live ELO unavailable."""
    elo_diff = 0.0
    if _tournament_df is not None:
        r1 = _tournament_df[_tournament_df["team"] == team1]
        r2 = _tournament_df[_tournament_df["team"] == team2]
        if not r1.empty and not r2.empty:
            c1 = float(r1["champ"].iloc[0]) + 0.1
            c2 = float(r2["champ"].iloc[0]) + 0.1
            elo_diff = (c1 - c2) * 15   # heuristic scaling to ELO-like range

    feat_map = {
        "elo_diff":       elo_diff,
        "rest_diff":      0.0,
        "home_form_avg":  0.5,
        "away_form_avg":  0.5,
        "home_gd_last10": 0.0,
        "away_gd_last10": 0.0,
        "home_conf_mult": conf_multiplier(team1),
        "away_conf_mult": conf_multiplier(team2),
    }
    cols = _models.get("feature_cols", list(feat_map.keys()))
    return np.array([[feat_map.get(c, 0.0) for c in cols]])


def _fallback_predict(team1: str, team2: str) -> dict:
    """CSV-based prediction when ML models are not loaded."""
    t1_champ, t2_champ = 0.05, 0.05
    if _tournament_df is not None:
        r1 = _tournament_df[_tournament_df["team"] == team1]
        r2 = _tournament_df[_tournament_df["team"] == team2]
        if not r1.empty: t1_champ = float(r1["champ"].iloc[0]) / 100 + 0.001
        if not r2.empty: t2_champ = float(r2["champ"].iloc[0]) / 100 + 0.001

    total = t1_champ + t2_champ + 0.25
    p1  = round(t1_champ / total * 100, 1)
    p2  = round(t2_champ / total * 100, 1)
    pd_ = round(100 - p1 - p2, 1)

    return {
        "team1": team1, "team2": team2,
        "team1_win": p1, "draw": pd_, "team2_win": p2,
        "winner": team1 if p1 > p2 else team2,
        "confidence": round(max(p1, p2), 1),
        "expected_goals_team1": max(0.3, round(1.3 + (t1_champ - t2_champ) * 5, 2)),
        "expected_goals_team2": max(0.3, round(1.3 - (t1_champ - t2_champ) * 5, 2)),
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
        "status":        "ok",
        "models_loaded": bool(_models.get("classifier")),
        "csv_loaded":    _tournament_df is not None,
        "teams_count":   len(_tournament_df) if _tournament_df is not None else 0,
        "models_dir":    MODELS_DIR,
        "csv_path":      CSV_PATH,
        "dir_exists":    os.path.isdir(MODELS_DIR),
        "dir_contents":  os.listdir(MODELS_DIR) if os.path.isdir(MODELS_DIR) else [],
        "load_errors":   _load_errors,
        "loaded_at":     _loaded_at,
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/tournament-data", methods=["GET"])
def tournament_data():
    if _tournament_df is None:
        return jsonify({
            "error": "Tournament CSV not loaded",
            "hint":  "Upload wc2026_predictions.csv to the models/ folder in your repo",
            "errors": _load_errors,
        }), 503

    df = _tournament_df.copy()

    top10 = (df.nlargest(10, "champ")[["team","champ"]]
               .rename(columns={"champ":"champion_prob"})
               .to_dict(orient="records"))

    stage_cols = [c for c in ["r32","r16","qf","sf","final","champ","group_winner_prob"]
                  if c in df.columns]
    all_teams  = df[["team"] + stage_cols].to_dict(orient="records")

    gw_col = "group_winner_prob" if "group_winner_prob" in df.columns else "r32"

    return jsonify({
        "top10":                      top10,
        "all_teams":                  all_teams,
        "champion_probabilities":     dict(zip(df["team"], df["champ"].round(2))),
        "group_winner_probabilities": dict(zip(df["team"], df[gw_col].round(2))),
        "last_updated":               _loaded_at,
        "teams_count":                len(df),
    })


@app.route("/api/team-path/<path:team_name>", methods=["GET"])
def team_path(team_name: str):
    if _tournament_df is None:
        return jsonify({"error": "Tournament CSV not loaded"}), 503

    team = standardise(team_name)
    row  = _tournament_df[_tournament_df["team"].str.lower() == team.lower()]
    if row.empty:
        row = _tournament_df[
            _tournament_df["team"].str.lower().str.contains(team.lower(), na=False)]
    if row.empty:
        return jsonify({"error": f"Team '{team_name}' not found",
                        "available": _tournament_df["team"].tolist()}), 404

    r = row.iloc[0]
    return jsonify({
        "team":              r["team"],
        "r32":               float(r["r32"])               if "r32"               in r.index else None,
        "r16":               float(r["r16"])               if "r16"               in r.index else None,
        "qf":                float(r["qf"])                if "qf"                in r.index else None,
        "sf":                float(r["sf"])                if "sf"                in r.index else None,
        "final":             float(r["final"])             if "final"             in r.index else None,
        "champ":             float(r["champ"])             if "champ"             in r.index else None,
        "group_winner_prob": float(r["group_winner_prob"]) if "group_winner_prob" in r.index else None,
    })


@app.route("/api/predict-match", methods=["POST"])
def predict_match():
    body  = request.get_json(silent=True) or {}
    team1 = standardise(body.get("team1", "").strip())
    team2 = standardise(body.get("team2", "").strip())

    if not team1 or not team2:
        return jsonify({"error": "Both team1 and team2 are required"}), 400
    if team1 == team2:
        return jsonify({"error": "Teams must be different"}), 400

    if "classifier" in _models:
        try:
            feat   = _build_feature_row(team1, team2)
            clf    = _models["classifier"]
            proba  = clf.predict_proba(feat)[0]
            classes = list(clf.classes_)

            p_t1 = round(float(proba[classes.index(2)]) * 100, 1)
            p_dr = round(float(proba[classes.index(1)]) * 100, 1)
            p_t2 = round(float(proba[classes.index(0)]) * 100, 1)

            exp_h = round(float(_models["poisson_home"].predict(feat)[0]), 2)
            exp_a = round(float(_models["poisson_away"].predict(feat)[0]), 2)

            return jsonify({
                "team1": team1, "team2": team2,
                "team1_win": p_t1, "draw": p_dr, "team2_win": p_t2,
                "winner":     team1 if p_t1 > p_t2 else (team2 if p_t2 > p_t1 else "Draw"),
                "confidence": round(max(p_t1, p_t2), 1),
                "expected_goals_team1": max(0.3, exp_h),
                "expected_goals_team2": max(0.3, exp_a),
                "model_used": "ml_ensemble",
            })
        except Exception as e:
            log.error(f"ML prediction error: {e}", exc_info=True)

    return jsonify(_fallback_predict(team1, team2))


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
