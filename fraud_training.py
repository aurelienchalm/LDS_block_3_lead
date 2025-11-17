import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from decimal import Decimal

from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


# ─────────────────────────────────────────────
# 1. Config & connexions
# ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

# NeonDB
DATABASE_URL = os.getenv("POSTGRES_DATABASE")
assert DATABASE_URL, "POSTGRES_DATABASE manquant dans le .env"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
assert MLFLOW_TRACKING_URI, "MLFLOW_TRACKING_URI manquant dans le .env"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = os.getenv("MODEL_NAME")
EXPERIMENT_NAME = MODEL_NAME
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()
REGISTERED_NAME = os.getenv("REGISTERED_NAME")

# Hyperparamètres globaux
TARGET = "is_fraud"
TEST_SIZE = 0.2
RS = 42

# ─────────────────────────────────────────────
# 2. Feature engineering (mêmes steps que le notebook)
# ─────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance de Haversine en km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Si on a `transaction_time` (venant de la prod), on l'aligne sur trans_date_trans_time
    if "trans_date_trans_time" not in df.columns and "transaction_time" in df.columns:
        df["trans_date_trans_time"] = df["transaction_time"]

    # 1) Dates
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year
    df = df.drop(columns=["dob"])

    df["year"] = df["trans_date_trans_time"].dt.year
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df = df.drop(columns=["trans_date_trans_time"])

    # 👉 2) Distance Haversine (conversion Decimal -> float)
    if {"lat", "long", "merch_lat", "merch_long"}.issubset(df.columns):
        for col in ["lat", "long", "merch_lat", "merch_long"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Decimal -> float / NaN

        df["distance_km"] = haversine_distance(
            df["lat"], df["long"], df["merch_lat"], df["merch_long"]
        )
        df = df.drop(columns=["lat", "long", "merch_lat", "merch_long"])

    # 3) Colonnes numériques / catégorielles / à dropper
    numeric_features = [
        "amt",
        "city_pop",
        "distance_km",
        "age",
        "year",
        "month",
        "day_of_week",
        "hour",
        "is_weekend",
    ]

    categorical_features = [
        "merchant",
        "category",
        "gender",
        "state",
        "job",
    ]

    drop_cols = [
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "zip",
        "trans_num",
        "unix_time",
    ]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)

    return df, numeric_features, categorical_features


# ─────────────────────────────────────────────
# 3. Entraînement + MLflow
# ─────────────────────────────────────────────

def main():
    # 3.1 Charger les données depuis NeonDB
    print("🔄 Chargement des données depuis fraud_training_dataset ...")
    
    
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM fraud_training_dataset"))
        rows = result.mappings().all()  # liste de dicts
        df = pd.DataFrame(rows)

    print(f"✅ {len(df)} lignes chargées")

    # 3.2 Feature engineering
    # 3.2 Feature engineering
    df, numeric_features, categorical_features = build_features(df)
    print(f"Shape après FE : {df.shape}")

    assert TARGET in df.columns, f"Colonne cible {TARGET} manquante après FE"

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 3.3 Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RS,
        stratify=y,
    )

    # 3.4 Préprocesseurs
    numeric_transformer = Pipeline(
        steps=[
            ("scaler_num", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder_cat", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            )),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 3.5 Modèle XGBoost
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb),
    ])

    # 3.6 Run MLflow
    RUN_NAME = "xgb_pipeline_retrain"
    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"🚀 MLflow run_id = {run_id}")

        # Enregistrer quelques params
        mlflow.log_params({
            "test_size": TEST_SIZE,
            "random_state": RS,
            "model_type": "XGBClassifier",
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float(scale_pos_weight),
        })

        # Fit
        print("🧠 Entraînement du modèle...")
        model.fit(X_train, y_train)

        # Probas
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test  = model.predict_proba(X_test)[:, 1]

        # Seuil 0.5
        y_pred_train = (y_proba_train >= 0.5).astype(int)
        y_pred_test  = (y_proba_test  >= 0.5).astype(int)

        metrics_base = {
            "train_f1_at_0_5":        f1_score(y_train, y_pred_train),
            "train_precision_at_0_5": precision_score(y_train, y_pred_train, zero_division=0),
            "train_recall_at_0_5":    recall_score(y_train, y_pred_train, zero_division=0),
            "test_f1_at_0_5":         f1_score(y_test, y_pred_test),
            "test_precision_at_0_5":  precision_score(y_test, y_pred_test, zero_division=0),
            "test_recall_at_0_5":     recall_score(y_test, y_pred_test, zero_division=0),
            "test_roc_auc":           roc_auc_score(y_test, y_proba_test),
            "test_pr_auc":            average_precision_score(y_test, y_proba_test),
        }
        mlflow.log_metrics(metrics_base)

        print("📊 Metrics seuil 0.5 :", metrics_base)

        # 3.7 Recherche du meilleur seuil (F1 sur test)
        best_f1, best_thresh = 0.0, 0.5
        for t in np.arange(0.0, 1.0, 0.01):
            preds = (y_proba_test >= t).astype(int)
            f1 = f1_score(y_test, preds)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        y_pred_train_opt = (y_proba_train >= best_thresh).astype(int)
        y_pred_test_opt  = (y_proba_test  >= best_thresh).astype(int)

        metrics_opt = {
            "best_threshold":           float(best_thresh),
            "train_f1_at_best":         f1_score(y_train, y_pred_train_opt),
            "train_precision_at_best":  precision_score(y_train, y_pred_train_opt, zero_division=0),
            "train_recall_at_best":     recall_score(y_train, y_pred_train_opt, zero_division=0),
            "test_f1_at_best":          f1_score(y_test, y_pred_test_opt),
            "test_precision_at_best":   precision_score(y_test, y_pred_test_opt, zero_division=0),
            "test_recall_at_best":      recall_score(y_test, y_pred_test_opt, zero_division=0),
        }
        mlflow.log_metrics(metrics_opt)

        print(f"🔎 Seuil optimal = {best_thresh:.2f}, F1_test = {metrics_opt['test_f1_at_best']:.4f}")

        # 3.8 Artefact JSON du seuil
        mlflow.log_text(json.dumps({"best_threshold": float(best_thresh)}),
                        "decision_threshold.json")

        # 3.9 Log du modèle (pipeline complet)
        sample_input = X_train.head(50).copy()

        # 🔧 Rendre l'input_example sérialisable (Decimal -> float)
        def to_serializable(x):
            if isinstance(x, Decimal):
                return float(x)
            return x

        sample_input = sample_input.applymap(to_serializable)

        # On peut aussi remplacer les NaN par None pour être safe
        sample_input = sample_input.where(sample_input.notnull(), None)

        signature = infer_signature(
            sample_input,
            model.predict_proba(sample_input)[:, 1]
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",   # OK même si warning "deprecated"
            signature=signature,
            input_example=sample_input.iloc[:3],
        )

        # ─────────────────────────────
        # 3.10 Registry + alias champion
        # ─────────────────────────────
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)
        version = mv.version

        # Tags
        client.set_model_version_tag(
            name=REGISTERED_NAME,
            version=version,
            key="best_threshold",
            value=str(best_thresh),
        )
        client.set_model_version_tag(
            name=REGISTERED_NAME,
            version=version,
            key="champion",
            value="true",
        )

        # Passage en Production + alias champion
        client.transition_model_version_stage(
            name=REGISTERED_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )

        try:
            client.set_registered_model_alias(REGISTERED_NAME, "champion", version)
        except Exception as e:
            print("Alias non défini (MLflow < 2.3 ? OK) :", e)

        # On met champion=false sur les autres versions
        all_versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
        for v in all_versions:
            if v.version != version:
                try:
                    client.set_model_version_tag(
                        REGISTERED_NAME, v.version, "champion", "false"
                    )
                except Exception:
                    pass

        print(f"✅ Nouveau modèle champion : version {version}, seuil = {best_thresh:.2f}")


if __name__ == "__main__":
    main()