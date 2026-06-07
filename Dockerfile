FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .

ENV PORT=5000
EXPOSE 5000

CMD gunicorn api:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120
