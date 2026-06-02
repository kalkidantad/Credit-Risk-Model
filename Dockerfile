FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    MLFLOW_MODEL_URI=models:/credit_risk_classifier/Production \
    FEATURE_COLUMNS_PATH=/app/data/processed/feature_columns.json

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
