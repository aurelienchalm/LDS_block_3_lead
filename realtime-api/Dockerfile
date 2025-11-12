FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Dataset par défaut
ENV PAYMENTS_CSV="https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"

# Le port exposé PAR CETTE API (différent de ton fraud-api)
EXPOSE 8001

# Lancement du serveur FastAPI sur le port 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]