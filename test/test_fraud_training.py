# test/test_fraud_training.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from sqlalchemy import create_engine

# Ajouter le chemin du projet (racine) dans sys.path
# /app/test/test_fraud_training.py → on remonte d'un niveau vers /app
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fraud_training import main, build_features, TARGET

POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")

def make_minimal_raw_df():
    return pd.DataFrame([
        {
            "trans_date_trans_time": "2020-06-01 12:34:56",
            "transaction_time": None,  # pour vérifier qu'on utilise bien trans_date_trans_time
            "dob": "1985-01-01",
            "lat": 48.8566,
            "long": 2.3522,
            "merch_lat": 43.2965,
            "merch_long": 5.3698,
            "amt": 42.5,
            "city_pop": 100000,
            "merchant": "fraud_Merchant",
            "category": "shopping_pos",
            "gender": "M",
            "state": "CA",
            "job": "Engineer",
            "cc_num": 1234567890,
            "first": "John",
            "last": "Doe",
            "street": "Some street",
            "city": "Paris",
            "zip": 75000,
            "trans_num": "abc",
            "unix_time": 1234567890,
            TARGET: 1,
        }
    ])


def test_build_features_output_schema():
    df_raw = make_minimal_raw_df()
    df_fe, num_cols, cat_cols = build_features(df_raw)

    # La target doit être là
    assert TARGET in df_fe.columns

    # Colonnes dérivées attendues
    for col in ["distance_km", "age", "year", "month", "day_of_week", "hour", "is_weekend"]:
        assert col in df_fe.columns

    # Colonnes "dropped" ne doivent plus être là
    for col in ["cc_num", "first", "last", "street", "city", "zip", "trans_num", "unix_time"]:
        assert col not in df_fe.columns

    # Types raisonnables
    assert df_fe["distance_km"].dtype != "O"
    assert set(df_fe[TARGET].unique()) <= {0, 1}


def test_build_features_with_transaction_time_instead_of_trans_date():
    df_raw = make_minimal_raw_df()

    # On simule le cas "prod" : pas de trans_date_trans_time mais transaction_time
    df_raw["transaction_time"] = df_raw["trans_date_trans_time"]
    df_raw = df_raw.drop(columns=["trans_date_trans_time"])

    df_fe, _, _ = build_features(df_raw)

    # On vérifie que l’age a bien été calculé
    assert "age" in df_fe.columns
    assert not df_fe["age"].isna().any()
    
def get_engine():
    assert POSTGRES_DATABASE, "POSTGRES_DATABASE doit être défini"
    return create_engine(POSTGRES_DATABASE)


def test_fraud_training_dataset_basic_quality():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM fraud_training_dataset LIMIT 10000", con=engine)

    # Non vide
    assert len(df) > 0

    # Target présente & dans {0,1}
    assert "is_fraud" in df.columns
    assert set(df["is_fraud"].unique()) <= {0, 1}

    # Quelques colonnes obligatoires
    for col in ["amt", "city_pop", "merchant", "category", "gender", "state", "job"]:
        assert col in df.columns

    # Pas de nulls sur des colonnes critiques (au moins dans l’échantillon)
    for col in ["amt", "city_pop", "merchant", "category"]:
        assert df[col].notna().all()
        
def test_fraud_training_end_to_end():
    """
    Test d'intégration : vérifie que le script fraud_training.py
    s'exécute sans lever d'exception.

    Il utilisera :
    - POSTGRES_DATABASE pour se connecter à NeonDB
    - MLFLOW_TRACKING_URI pour se connecter à MLflow
    - MODEL_NAME / REGISTERED_NAME pour le registry
    (tous fournis via le .env injecté par Jenkins).
    """
    # Petit sanity check pour être sûr qu'on a bien les vars clés
    assert os.getenv("POSTGRES_DATABASE"), "POSTGRES_DATABASE doit être défini dans le .env"
    assert os.getenv("MLFLOW_TRACKING_URI"), "MLFLOW_TRACKING_URI doit être défini dans le .env"
    assert os.getenv("MODEL_NAME"), "MODEL_NAME doit être défini dans le .env"
    assert os.getenv("REGISTERED_NAME"), "REGISTERED_NAME doit être défini dans le .env"

    # Appel direct à la fonction main() de ton script
    main()
        

