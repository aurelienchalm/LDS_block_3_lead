from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from datetime import datetime, timedelta
import pandas as pd
import os
from sqlalchemy import create_engine
import requests
from airflow.hooks.base import BaseHook

# -------------------------------------------------------------------
# Connexion NeonDB
# -------------------------------------------------------------------

def get_neondb_engine():
    conn = BaseHook.get_connection("neondb_fraud")

    url = (
        f"postgresql+psycopg2://{conn.login}:{conn.password}"
        f"@{conn.host}:{conn.port}/{conn.schema}"
        f"?sslmode=require"
    )

    return create_engine(url)

default_args = {
    "owner": "aurélien",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# -------------------------------------------------------------------
# Envoi email via Resend
# -------------------------------------------------------------------

def send_resend_email(subject: str, html: str):
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
        "Content-Type": "application/json"
    }

    resp = requests.post(url, json=payload, headers=headers)

    if resp.status_code >= 300:
        raise AirflowFailException(f"Resend email failed: {resp.text}")

    return True

# -------------------------------------------------------------------
# DAG
# -------------------------------------------------------------------

with DAG(
    dag_id="daily_fraud_report",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 1 * * *",
    catchup=False,
    default_args=default_args,
    tags=["fraud", "reporting"],
):

    @task
    def extract_data():
        engine = get_neondb_engine()

        df = pd.read_sql("""
            SELECT *
            FROM fraud_realtime_predictions
            WHERE DATE(prediction_time) = DATE(NOW() AT TIME ZONE 'UTC' - INTERVAL '1 day')
        """, con=engine)

        if df.empty:
            raise AirflowFailException("Aucune transaction trouvée pour hier.")

        return df.to_json(orient="split")

    @task
    def generate_html_report(df_json):
        df = pd.read_json(df_json, orient="split")

        total = len(df)
        fraudulent = df["is_fraud"].sum()
        fraud_rate = round(fraudulent / total * 100, 2)

        fraud_df = df[df["is_fraud"] == True]

        if fraud_df.empty:
            fraud_table_html = "<p>Aucune fraude détectée hier 🎉</p>"
        else:
            fraud_table_html = fraud_df.head(20).to_html(index=False)

        yesterday = (datetime.now() - timedelta(days=1)).date()

        html = f"""
            <h2>📊 Rapport Fraude — {yesterday}</h2>
            <p><b>Total transactions :</b> {total}</p>
            <p><b>Fraudes détectées :</b> {fraudulent}</p>
            <p><b>Taux de fraude :</b> {fraud_rate}%</p>

            <h3>Top 20 fraudes :</h3>
            {fraud_table_html}
        """

        return html

    @task
    def email_report(html):
        send_resend_email(
            subject="Rapport quotidien — Fraude détectée",
            html=html
        )

    data = extract_data()
    html = generate_html_report(data)
    email_report(html)