# Image légère + roues wheels pour xgboost disponibles
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dépendances système minimales (certs, build basique)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements d'abord (layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY main.py /app/main.py
COPY model_utils.py /app/model_utils.py

# Variables attendues:
# - MLFLOW_TRACKING_URI
# - REGISTERED_NAME (optionnel, par défaut "fraud-detection-xgb-pipeline")
# - PORT (optionnel)
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]