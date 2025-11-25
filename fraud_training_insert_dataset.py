# init_fraud_training_dataset.py

import os
from pathlib import Path
import io

import pandas as pd
import boto3
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ─────────────────────────────────────────
# 1. Config & connexion NeonDB + S3
# ─────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

DATABASE_URL = os.getenv("POSTGRES_DATABASE")
assert DATABASE_URL, "Variable d'environnement POSTGRES_DATABASE manquante dans le .env"

BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_KEY = os.getenv("FRAUD_DATASET_S3_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-3")

assert BUCKET_NAME, "BUCKET_NAME manquant dans le .env"
assert S3_KEY, "FRAUD_DATASET_S3_KEY manquant dans le .env"


def load_csv_from_s3() -> pd.DataFrame:
    """
    Charge le CSV d'entraînement depuis S3 :
      - Bucket : BUCKET_NAME
      - Key    : S3_KEY
    """
    print(f"📥 Téléchargement du CSV depuis S3 : s3://{BUCKET_NAME}/{S3_KEY}")

    # boto3 utilisera AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
    s3 = boto3.client("s3", region_name=AWS_REGION)

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_KEY)
    csv_bytes = obj["Body"].read()
    csv_str = csv_bytes.decode("utf-8")

    df = pd.read_csv(io.StringIO(csv_str))
    print(f"✅ CSV S3 chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")
    return df


# ─────────────────────────────────────────
# 2. Fonction principale
# ─────────────────────────────────────────

def load_csv_into_fraud_training_dataset():
    # 2.1 Lecture brute du CSV depuis S3
    df = load_csv_from_s3()

    # Colonnes attendues dans le CSV original (fraudTest)
    expected_cols = [
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
        "first",
        "last",
        "gender",
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "trans_num",
        "unix_time",
        "merch_lat",
        "merch_long",
        "is_fraud",
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Colonnes manquantes dans le CSV : {missing}")

    # On ne garde que ce qui nous intéresse pour la table fraud_training_dataset
    df_out = df[
        [
            "merchant",
            "category",
            "amt",
            "gender",
            "state",
            "job",
            "city_pop",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "dob",
            "cc_num",
            "trans_num",
            "trans_date_trans_time",
            "is_fraud",
        ]
    ].copy()

    # ─────────────
    # 3. Typage
    # ─────────────

    # Dates
    df_out["dob"] = pd.to_datetime(df_out["dob"], errors="coerce").dt.date
    df_out["trans_date_trans_time"] = pd.to_datetime(
        df_out["trans_date_trans_time"], errors="coerce"
    )

    # Numériques
    df_out["amt"] = pd.to_numeric(df_out["amt"], errors="coerce")
    df_out["city_pop"] = pd.to_numeric(df_out["city_pop"], errors="coerce").astype("Int64")
    df_out["lat"] = pd.to_numeric(df_out["lat"], errors="coerce")
    df_out["long"] = pd.to_numeric(df_out["long"], errors="coerce")
    df_out["merch_lat"] = pd.to_numeric(df_out["merch_lat"], errors="coerce")
    df_out["merch_long"] = pd.to_numeric(df_out["merch_long"], errors="coerce")

    # Texte
    df_out["merchant"] = df_out["merchant"].astype(str)
    df_out["category"] = df_out["category"].astype(str)
    df_out["gender"] = df_out["gender"].astype(str)
    df_out["state"] = df_out["state"].astype(str)
    df_out["job"] = df_out["job"].astype(str)
    df_out["cc_num"] = df_out["cc_num"].astype(str)
    df_out["trans_num"] = df_out["trans_num"].astype(str)

    # Target
    df_out["is_fraud"] = (
        pd.to_numeric(df_out["is_fraud"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    print("🧹 Types nettoyés, préparation pour insertion en base...")
    print(df_out.dtypes)

    # ─────────────
    # 4. Insertion NeonDB
    # ─────────────

    engine = create_engine(DATABASE_URL)

    table_name = "fraud_training_dataset"
    chunksize = 10_000

    total = len(df_out)
    inserted = 0

    print(f"🚀 Insertion dans la table {table_name} sur NeonDB...")

    with engine.begin() as conn:  # transaction
        for start in range(0, total, chunksize):
            end = start + chunksize
            chunk = df_out.iloc[start:end]
            chunk.to_sql(
                table_name,
                con=conn,
                if_exists="append",  # 👉 on conserve l'historique
                index=False,
                method="multi",
            )
            inserted += len(chunk)
            print(f"   → {inserted}/{total} lignes insérées")

    print(f"✅ Terminé ! {inserted} lignes insérées dans {table_name}.")


if __name__ == "__main__":
    load_csv_into_fraud_training_dataset()