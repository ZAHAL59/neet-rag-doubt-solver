FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the index at start-up so it uses whichever API key the host provides.
CMD python -m app.ingest && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
