from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.hooks.base import BaseHook

from datetime import datetime, timedelta
import os
import json
import tempfile

import pandas as pd
import boto3
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from io import StringIO

from sqlalchemy import create_engine

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import requests  # pour Resend

# -------------------------------------------------------------------
# Constantes
# -------------------------------------------------------------------

FEATURE_COLS = [
    "merchant", "category", "amt", "gender", "state", "job",
    "city_pop", "distance_km", "age", "year", "month",
    "day_of_week", "hour", "is_weekend",
]

default_args = {
    "owner": "aurelien",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

AWS_CONN_ID = "aws_default"

# -------------------------------------------------------------------
# Helpers : NeonDB, S3, Resend
# -------------------------------------------------------------------

def get_neondb_engine():
    """
    Récupère la connexion NeonDB via Airflow Connection `neondb_fraud`.
    À configurer dans l’UI Airflow :
      Conn Id: neondb_fraud
      Conn Type: Postgres
      Host: ep-...neon.tech
      Schema: neondb
      Login: neondb_owner
      Password: ...
      Port: 5432
      Extra: {"sslmode": "require"}
    """
    conn = BaseHook.get_connection("neondb_fraud")

    # On reconstruit une URL SQLAlchemy
    url = (
        f"postgresql+psycopg2://{conn.login}:{conn.get_password()}"
        f"@{conn.host}:{conn.port}/{conn.schema}"
        f"?sslmode=require"
    )

    return create_engine(url)

def send_resend_email(subject: str, html: str):
    """
    Envoie un mail via Resend.
    Nécessite 3 Variables Airflow :
      - RESEND_API_KEY
      - RESEND_SENDER  (ex: "Fraud Alert <onboarding@resend.dev>")
      - RESEND_TO      (ex: "ton_mail@domaine.com")
    """
    api_key = Variable.get("RESEND_API_KEY")
    sender = Variable.get("RESEND_SENDER")
    receiver = Variable.get("RESEND_TO")

    url = "https://api.resend.com/emails"

    payload = {
        "from": sender,
        "to": receiver,
        "subject": subject,
        "html": html,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers)

    if resp.status_code >= 300:
        raise AirflowFailException(f"Resend email failed: {resp.text}")

    return True


# -------------------------------------------------------------------
# DAG
# -------------------------------------------------------------------

with DAG(
    dag_id="fraud_drift_monitoring",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",           # 1 fois par jour
    catchup=False,
    default_args=default_args,
    tags=["fraud", "drift", "evidently"],
):

    @task
    def run_drift_report():
        """
        1. Charge la baseline (S3, CSV)
        2. Charge les données de prod (NeonDB)
        3. Lance Evidently (DataDriftPreset)
        4. Sauvegarde un HTML + JSON sur S3
        5. Envoie un mail si drift détecté
        """

        # -------------------------
        # 1) Chargement baseline S3
        # -------------------------
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)

        baseline_bucket = Variable.get("FRAUD_BASELINE_BUCKET")
        baseline_key = Variable.get("FRAUD_BASELINE_KEY")

        csv_str = s3_hook.read_key(key=baseline_key, bucket_name=baseline_bucket)
        df_baseline = pd.read_csv(StringIO(csv_str))

        # On s’assure de garder exactement les features
        df_baseline = df_baseline[FEATURE_COLS]

        # -------------------------
        # 2) Charge données prod
        # -------------------------
        engine = get_neondb_engine()

        query = """
        SELECT
            merchant,
            category,
            amt,
            gender,
            state,
            job,
            city_pop,
            distance_km,
            age,
            year,
            month,
            day_of_week,
            hour,
            is_weekend
        FROM fraud_realtime_predictions
        WHERE prediction_time::date = CURRENT_DATE
        """

        df_current = pd.read_sql(query, con=engine)

        if df_current.empty:
            raise AirflowFailException(
                "Aucune donnée de production pour aujourd'hui dans fraud_realtime_predictions."
            )

        df_current = df_current[FEATURE_COLS]

        # -------------------------
        # 3) Evidently – DataDrift
        # -------------------------
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_baseline, current_data=df_current)

        report_dict = report.as_dict()

        # 🔍 On cherche la métrique de drift dans les métriques retournées
        drift_result = None
        for m in report_dict.get("metrics", []):
            metric_name = m.get("metric", "")
            if "DataDrift" in metric_name:
                drift_result = m.get("result")
                break

        if drift_result is None:
            raise AirflowFailException("Impossible de trouver la métrique de drift dans le rapport Evidently")

        # Cas 1 : Evidently retourne directement un bool (dataset_drift)
        if isinstance(drift_result, bool):
            dataset_drift = drift_result
            share_drifted = None

        # Cas 2 : Evidently retourne un dict détaillé
        elif isinstance(drift_result, dict):
            dataset_drift = bool(drift_result.get("dataset_drift", False))
            share_drifted = drift_result.get("share_of_drifted_columns")

        else:
            raise AirflowFailException(f"Format inattendu pour drift_result: {type(drift_result)}")

        print(f"[EVIDENTLY] dataset_drift={dataset_drift}, share_drifted={share_drifted}")

        # Seuil d’alerte configurable
        drift_threshold = float(Variable.get("FRAUD_DRIFT_THRESHOLD", 0.3))

        # -------------------------
        # 4) Sauvegarde HTML + JSON sur S3
        # -------------------------
        drift_bucket = Variable.get("FRAUD_BASELINE_BUCKET")
        drift_prefix = Variable.get("FRAUD_DRIFT_PREFIX")

        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, f"fraud_drift_report_{today_str}.html")
            json_path = os.path.join(tmpdir, f"fraud_drift_report_{today_str}.json")

            # HTML Evidently
            report.save_html(html_path)

            # JSON Evidently
            with open(json_path, "w") as f:
                json.dump(report_dict, f)

            # Upload HTML
            s3_hook.load_file(
                filename=html_path,
                key=f"{drift_prefix}/fraud_drift_report_{today_str}.html",
                bucket_name=drift_bucket,
                replace=True,
            )

            # Upload JSON
            s3_hook.load_file(
                filename=json_path,
                key=f"{drift_prefix}/fraud_drift_report_{today_str}.json",
                bucket_name=drift_bucket,
                replace=True,
            )

        # -------------------------
        # 5) Alerte email si drift
        # -------------------------
        # Si share_drifted est None (cas bool simple), on alerte dès qu'il y a drift
        trigger_alert = False
        if dataset_drift and share_drifted is None:
            trigger_alert = True
        elif dataset_drift and share_drifted is not None and share_drifted >= drift_threshold:
            trigger_alert = True

        if trigger_alert:
            percent = round((share_drifted or 0) * 100, 2)
            report_location = f"s3://{drift_bucket}/{drift_prefix}/fraud_drift_report_{today_str}.html"

            html_msg = f"""
            <h2>🚨 Alerte drift sur le modèle de fraude</h2>
            <p><b>Date :</b> {today_str}</p>
            <p><b>Dataset drift :</b> {dataset_drift}</p>
            <p><b>Part de features driftées :</b> {percent}%</p>
            <p><b>Seuil d'alerte :</b> {drift_threshold * 100}%</p>
            <p>📄 <b>Rapport Evidently :</b> {report_location}</p>
            """

            send_resend_email(
                subject="🚨 Drift détecté sur le modèle de fraude (Evidently)",
                html=html_msg,
            )

        return {
            "dataset_drift": bool(dataset_drift),
            "share_drifted": float(share_drifted) if share_drifted is not None else None,
        }

    run_drift_report()