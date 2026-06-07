FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY api.py .

# Copy model artifacts (put your .pkl files + CSV in ./models/)
COPY models/ ./models/

ENV PORT=5000
EXPOSE 5000

CMD gunicorn api:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120
