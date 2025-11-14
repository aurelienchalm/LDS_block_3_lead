#!/usr/bin/env python3
"""
Script pour créer une baseline Evidently à partir de la table fraud_raw dans NeonDB.

- Lit la table fraud_raw
- Échantillonne N lignes (par défaut 50 000)
- Sauvegarde un CSV local : data/fraud_baseline.csv
- Optionnel : upload sur S3 si les variables AWS sont définies

À lancer une seule fois (sauf si tu veux changer de baseline).
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Optionnel pour S3
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None  # si pas installé, la partie S3 sera ignorée

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger("create_fraud_baseline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────
# Config chemins / env
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    logger.info(f".env chargé depuis : {ENV_PATH}")
else:
    logger.warning(f"Aucun fichier .env trouvé à : {ENV_PATH}")

# NeonDB / Postgres
DATABASE_URL = os.getenv("POSTGRES_DATABASE")
if not DATABASE_URL:
    raise RuntimeError("La variable d'environnement POSTGRES_DATABASE n'est pas définie.")

# Paramètres baseline
BASELINE_OUTPUT_DIR = ROOT_DIR / "data"
BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_CSV_PATH = BASELINE_OUTPUT_DIR / "fraud_baseline.csv"

# Nombre de lignes dans la baseline (à adapter si tu veux)
BASELINE_SAMPLE_SIZE = int(os.getenv("BASELINE_SAMPLE_SIZE", 50_000))

# Config S3 (optionnelle)
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# The `AWS_DEFAULT_REGION` variable is used to specify the default region to be used when interacting
# with AWS services. In the provided script, if the `AWS_DEFAULT_REGION` environment variable is not
# defined, it defaults to `"eu-west-3"`.
AWS_DEFAULT_REGION    = os.getenv("AWS_DEFAULT_REGION")
BUCKET_NAME           = os.getenv("BUCKET_NAME")              # ex: "mon-bucket-mlops"
BASELINE_S3_KEY       = os.getenv("FRAUD_BASELINE_S3_KEY", "fraud/baseline/fraud_baseline.csv")


def get_engine():
    logger.info("Création de l'engine SQLAlchemy pour NeonDB...")
    engine = create_engine(DATABASE_URL)
    return engine


def extract_fraud_raw(engine) -> pd.DataFrame:
    """
    Lit l'ensemble de la table fraud_raw.
    Comme il y a ~555k lignes, ça reste ok pour une machine standard.
    Si un jour la table grossit, on pourra passer en mode chunk.
    """
    logger.info("Lecture de la table fraud_raw...")
    df = pd.read_sql("SELECT * FROM fraud_raw", con=engine)
    logger.info(f"Shape de fraud_raw : {df.shape}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduit le feature engineering du notebook / API :
    - age, year, month, day_of_week, hour, is_weekend
    - distance_km via Haversine
    """

    df = df.copy()

    # Dates
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

    # Age + time features
    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year
    df["year"] = df["trans_date_trans_time"].dt.year
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["hour"] = df["trans_date_trans_time"].dt.hour
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

    return df


FEATURE_COLS = [
    "merchant", "category", "amt", "gender", "state", "job",
    "city_pop", "distance_km", "age", "year", "month",
    "day_of_week", "hour", "is_weekend"
]

def build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la baseline Evidently :
    - garde uniquement les colonnes utilisées par le modèle
    - échantillonne si nécessaire
    """
    # On garde uniquement les colonnes de features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour la baseline : {missing}")

    df_feat = df[FEATURE_COLS]

    n = len(df_feat)
    if n <= BASELINE_SAMPLE_SIZE:
        return df_feat
    else:
        return df_feat.sample(BASELINE_SAMPLE_SIZE, random_state=42)

def save_baseline_csv(df_baseline: pd.DataFrame):
    logger.info(f"Sauvegarde de la baseline au format CSV → {BASELINE_CSV_PATH}")
    df_baseline.to_csv(BASELINE_CSV_PATH, index=False)
    logger.info("✅ Baseline CSV sauvegardée.")


def upload_baseline_to_s3():
    """
    Upload du fichier de baseline vers S3 (optionnel).
    Nécessite boto3 et les variables AWS_* et BUCKET_NAME.
    """
    if not boto3:
        logger.warning("boto3 n'est pas installé, skip upload S3.")
        return

    if not BUCKET_NAME:
        logger.warning("BUCKET_NAME n'est pas défini, skip upload S3.")
        return

    if not BASELINE_CSV_PATH.exists():
        logger.warning(f"{BASELINE_CSV_PATH} n'existe pas, skip upload S3.")
        return

    logger.info(f"Upload de la baseline vers s3://{BUCKET_NAME}/{BASELINE_S3_KEY} ...")

    session = boto3.session.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )
    s3 = session.client("s3")

    try:
        s3.upload_file(
            Filename=str(BASELINE_CSV_PATH),
            Bucket=BUCKET_NAME,
            Key=BASELINE_S3_KEY,
            ExtraArgs={"ContentType": "text/csv"},
        )
        logger.info("✅ Baseline uploadée sur S3 avec succès.")
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Erreur lors de l'upload S3 : {e}")
        raise


def main():
    logger.info("🚀 Début création de la baseline fraud_raw")

    engine = get_engine()
    df_raw = extract_fraud_raw(engine)
    df_features = engineer_features(df_raw)
    df_baseline = build_baseline(df_features)
    save_baseline_csv(df_baseline)

    # Upload optionnel sur S3
    try:
        upload_baseline_to_s3()
    except Exception as e:
        logger.warning(f"Upload S3 échoué (non bloquant) : {e}")

    logger.info("✅ Script terminé, baseline prête.")


if __name__ == "__main__":
    main()