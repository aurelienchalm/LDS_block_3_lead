import os
from pathlib import Path
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine
import pandas as pd

import requests

# ───────────────────────────────────────────────
# 1) Chargement du .env
# ───────────────────────────────────────────────
# En local : .env à la racine du repo (../.env par rapport à app/model_utils.py)
# En Docker : les variables viennent de --env-file .env (load_dotenv ne gêne pas)
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
REGISTERED_NAME = os.getenv("REGISTERED_NAME")
DATABASE_URL = os.getenv("POSTGRES_DATABASE")

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_SENDER = os.getenv("RESEND_SENDER", "onboarding@resend.dev")
RESEND_TO = os.getenv("RESEND_TO")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI n'est pas défini dans l'environnement")

if not DATABASE_URL:
    raise RuntimeError("POSTGRES_DATABASE n'est pas défini dans l'environnement")

# ───────────────────────────────────────────────
# 2) Initialisation MLflow
# ───────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


# ───────────────────────────────────────────────
# 3) Fonction : charger pipeline champion
# ───────────────────────────────────────────────
def load_champion_pipeline_and_threshold():
    """
    Charge le pipeline champion (alias @champion) et le seuil optimal.
    Retourne (pipeline_sklearn, best_threshold: float, model_version: int)
    """
    mv = client.get_model_version_by_alias(REGISTERED_NAME, "champion")

    pipeline = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")

    best_threshold = float(mv.tags.get("best_threshold", 0.5))

    return pipeline, best_threshold, mv.version


# ───────────────────────────────────────────────
# 4) Base de données (SQLAlchemy)
# ───────────────────────────────────────────────
engine = create_engine(DATABASE_URL)


def insert_prediction(df: pd.DataFrame, table_name: str = "fraud_realtime_predictions"):
    """
    Insère un DataFrame dans PostgreSQL via SQLAlchemy.
    Utilisé pour stocker les prédictions temps réel dans NeonDB.
    """
    df.to_sql(table_name, engine, if_exists="append", index=False)

# ───────────────────────────────────────────────
# 4) Envoie des mails
# ───────────────────────────────────────────────

def send_fraud_alert_email(
    merchant: str,
    amount: float,
    proba_fraud: float,
    trans_num: str,
    transaction_time,
    city: str | None = None,
    state: str | None = None,
):
    """
    Envoie un email d'alerte via Resend lorsqu'une fraude est détectée.
    """
    if not RESEND_API_KEY or not RESEND_TO:
        print("⚠️ RESEND_API_KEY ou RESEND_TO manquants, pas d'email envoyé.")
        return

    # Formatage propre
    time_str = (
        transaction_time.isoformat()
        if hasattr(transaction_time, "isoformat")
        else str(transaction_time)
    )
    location = f"{city}, {state}" if city and state else "Localisation inconnue"

    subject = f"[ALERTE FRAUDE] Transaction suspecte de {amount:.2f} $ chez {merchant}"

    html_body = f"""
    <html>
      <body>
        <h2>🚨 Alerte fraude détectée</h2>
        <p>Une transaction potentiellement frauduleuse vient d'être détectée :</p>
        <ul>
          <li><b>Commerçant :</b> {merchant}</li>
          <li><b>Montant :</b> {amount:.2f} $</li>
          <li><b>Probabilité de fraude :</b> {proba_fraud:.2%}</li>
          <li><b>Date de transaction :</b> {time_str}</li>
          <li><b>Localisation :</b> {location}</li>
          <li><b>Transac num. :</b> {trans_num}</li>
        </ul>
        <p>Cette alerte a été générée automatiquement par le modèle XGBoost (pipeline MLflow).</p>
      </body>
    </html>
    """

    payload = {
        "from": f"Fraud Detector <{RESEND_SENDER}>",
        "to": [RESEND_TO],
        "subject": subject,
        "html": html_body,
    }

    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=10,
    )

    if resp.status_code >= 400:
        print(f"❌ Erreur Resend ({resp.status_code}): {resp.text}")
    else:
        print("✅ Email d'alerte fraude envoyé avec succès.")