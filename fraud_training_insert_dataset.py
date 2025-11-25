# init_fraud_training_dataset.py

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ─────────────────────────────────────────
# 1. Config & connexion NeonDB
# ─────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

DATABASE_URL = os.getenv("POSTGRES_DATABASE")
assert DATABASE_URL, "Variable d'environnement POSTGRES_DATABASE manquante dans le .env"

# Chemin vers le CSV (adapter si besoin)
CSV_PATH = Path("data/fraudTestLight.csv")


# ─────────────────────────────────────────
# 2. Fonction principale
# ─────────────────────────────────────────

def load_csv_into_fraud_training_dataset():
    print(f"Chargement du CSV : {CSV_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CSV_PATH}")

    # Lecture brute du CSV
    df = pd.read_csv(CSV_PATH)

    print(f"✅ CSV chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")

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
    # Schéma cible :
    #   merchant, category, amt, gender, state, job, city_pop,
    #   lat, long, merch_lat, merch_long, dob, cc_num, trans_num,
    #   trans_date_trans_time, is_fraud, created_at (géré par DEFAULT now())
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
    df_out["is_fraud"] = pd.to_numeric(df_out["is_fraud"], errors="coerce").fillna(0).astype(int)

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
                if_exists="append",
                index=False,
                method="multi",
            )
            inserted += len(chunk)
            print(f"   → {inserted}/{total} lignes insérées")

    print(f"✅ Terminé ! {inserted} lignes insérées dans {table_name}.")


if __name__ == "__main__":
    load_csv_into_fraud_training_dataset()