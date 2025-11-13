import os
import json
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_utils import (
    load_champion_pipeline_and_threshold,
    insert_prediction,send_fraud_alert_email
)

# ─────────────────────────────────────────────
# Initialisation FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="API de prédiction de fraude en temps réel (XGBoost + MLflow + NeonDB).",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Chargement du modèle champion depuis MLflow
# ─────────────────────────────────────────────
PIPELINE, BEST_THRESHOLD, MODEL_VERSION = load_champion_pipeline_and_threshold()
print(f"✅ Modèle champion chargé (version {MODEL_VERSION}, seuil = {BEST_THRESHOLD})")

# URL de l'API temps réel (qui renvoie /current-transactions)
REALTIME_API_URL = os.getenv(
    "REALTIME_API_URL",
    "http://localhost:8001/current-transactions",  # à adapter si besoin
)


# ─────────────────────────────────────────────
# Modèle Pydantic pour /predict (features déjà préparées)
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Features catégorielles
    merchant: str = Field(
        ...,
        example="fraud_Altenwerth-Kilback",
        description="Identifiant du marchand",
    )
    category: str = Field(
        ...,
        example="home",
        description="Catégorie de la transaction",
    )
    gender: Literal["M", "F"] = Field(
        ...,
        example="F",
        description="Genre du porteur de carte",
    )
    state: str = Field(
        ...,
        example="NY",
        description="État / région",
    )
    job: str = Field(
        ...,
        example="Comptroller",
        description="Profession du client",
    )

    # Features numériques
    amt: float = Field(
        ...,
        example=36.96,
        description="Montant de la transaction",
    )
    city_pop: int = Field(
        ...,
        example=722,
        description="Population de la ville du client",
    )
    distance_km: float = Field(
        ...,
        example=12.3,
        description="Distance (km) entre client et marchand",
    )
    age: int = Field(
        ...,
        example=35,
        description="Âge du client",
    )
    year: int = Field(
        ...,
        example=2020,
        description="Année de la transaction",
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        example=6,
        description="Mois de la transaction (1-12)",
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        example=2,
        description="Jour de la semaine (0=lundi, 6=dimanche)",
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        example=14,
        description="Heure de la transaction (0-23)",
    )
    is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
        example=0,
        description="1 si la transaction a lieu le week-end, sinon 0",
    )


# ─────────────────────────────────────────────
# Healthcheck
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "best_threshold": BEST_THRESHOLD,
    }


# ─────────────────────────────────────────────
# Endpoint /predict : prédiction directe (features déjà préparées)
# ─────────────────────────────────────────────
@app.post("/predict")
def predict(record: PredictRequest):
    """
    Prédit la fraude à partir d'un enregistrement déjà pré-traité
    (features alignées avec celles utilisées pour l'entraînement du modèle).
    """
    try:
        df = pd.DataFrame([record.dict()])
        proba = PIPELINE.predict_proba(df)[:, 1][0]
        pred = int(proba >= BEST_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

    return {
        "input": record.dict(),
        "proba_fraud": round(float(proba), 4),
        "is_fraud": bool(pred),
        "threshold": float(BEST_THRESHOLD),
    }


# ─────────────────────────────────────────────
# Endpoint /realtime-predict : appelle l'API temps réel + insert NeonDB
# ─────────────────────────────────────────────
@app.get("/realtime-predict")
def realtime_predict():
    """
    Récupère une transaction depuis l'API temps réel,
    applique le modèle champion et insère la transaction + prédiction dans NeonDB.
    """
    # 1) Appel de l'API temps réel
    try:
        resp = requests.get(REALTIME_API_URL, timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, str):
            payload = json.loads(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API temps réel : {e}")

    # 2) Reconstruction du DataFrame brut
    try:
        if not isinstance(payload, dict):
            raise ValueError("Le payload n'est pas un objet JSON")
        if "columns" not in payload or "data" not in payload:
            raise KeyError("Clés 'columns' ou 'data' manquantes dans la réponse de l'API temps réel")
        df_raw = pd.DataFrame(payload["data"], columns=payload["columns"])
        if df_raw.empty:
            raise ValueError("Aucune transaction reçue depuis l'API temps réel")

        # 🔹 Récupérer l'index de la transaction (ex: 455664) et l'ajouter au DF
        source_index = None
        if "index" in payload and isinstance(payload["index"], list) and len(payload["index"]) > 0:
            source_index = payload["index"][0]
            df_raw["source_index"] = source_index

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur parsing JSON : {e}")

        # 3) Feature engineering (comme dans le notebook d'entraînement)
    try:
        df = df_raw.copy()

        # Dates
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["transaction_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

        # Features dérivées
        df["age"] = df["transaction_time"].dt.year - df["dob"].dt.year
        df["year"] = df["transaction_time"].dt.year
        df["month"] = df["transaction_time"].dt.month
        df["day_of_week"] = df["transaction_time"].dt.dayofweek
        df["hour"] = df["transaction_time"].dt.hour
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Distance Haversine
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        df["distance_km"] = haversine_distance(
            df["lat"], df["long"], df["merch_lat"], df["merch_long"]
        )

        feature_cols = [
            "merchant", "category", "amt", "gender", "state", "job",
            "city_pop", "distance_km", "age", "year", "month",
            "day_of_week", "hour", "is_weekend",
        ]
        X = df[feature_cols]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur feature engineering : {e}")

    # 4) Prédiction avec le modèle champion
    try:
        proba = PIPELINE.predict_proba(X)[:, 1][0]
        pred = int(proba >= BEST_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inference : {e}")

    # 5) Préparation des données pour NeonDB
    try:
        row = df.iloc[0]

        df_sql = pd.DataFrame([{
            # 🔹 nouveau champ pour la table
            "source_index":     int(source_index) if source_index is not None else None,

            "merchant":         row["merchant"],
            "category":         row["category"],
            "amt":              float(row["amt"]),
            "gender":           row["gender"],
            "state":            row["state"],
            "job":              row["job"],
            "city_pop":         int(row["city_pop"]),
            "lat":              float(row["lat"]),
            "long":             float(row["long"]),
            "merch_lat":        float(row["merch_lat"]),
            "merch_long":       float(row["merch_long"]),
            "dob":              row["dob"].date() if not pd.isna(row["dob"]) else None,
            "cc_num":           str(row["cc_num"]),
            "trans_num":        row["trans_num"],
            "transaction_time": row["transaction_time"],
            "age":              int(row["age"]),
            "year":             int(row["year"]),
            "month":            int(row["month"]),
            "day_of_week":      int(row["day_of_week"]),
            "hour":             int(row["hour"]),
            "is_weekend":       int(row["is_weekend"]),
            "distance_km":      float(row["distance_km"]),
            "proba_fraud":      float(proba),
            "is_fraud":         bool(pred),
            "prediction_time":  datetime.utcnow(),
        }])

        # Insertion dans NeonDB
        insert_prediction(df_sql)
        
                # Si fraude détectée, on envoie une alerte email
        if pred == 1:
            try:
                send_fraud_alert_email(
                    merchant=row["merchant"],
                    amount=float(row["amt"]),
                    proba_fraud=float(proba),
                    trans_num=row.get("trans_num"),
                    transaction_time=row["transaction_time"],
                    city=row.get("city"),      # dispo dans df_raw
                    state=row.get("state"),    # idem
                    )
            except Exception as e:
                # On ne bloque pas l’API pour un souci d’email
                print(f"⚠️ Erreur lors de l’envoi de l’alerte email : {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur insertion NeonDB : {e}")

    # 6) Réponse API
    return {
        "source_index": source_index,
        "merchant":     row["merchant"],
        "amount":       float(row["amt"]),
        "proba_fraud":  round(float(proba), 4),
        "is_fraud":     bool(pred),
        "threshold":    float(BEST_THRESHOLD),
        "model_version": MODEL_VERSION,
    }


# ─────────────────────────────────────────────
# Lancement local (optionnel si tu utilises uvicorn en ligne de commande)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)