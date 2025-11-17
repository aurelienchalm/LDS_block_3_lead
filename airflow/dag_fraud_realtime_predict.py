from datetime import datetime, timedelta
import os
import requests

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable


def get_api_url() -> str:
    """
    Récupère l'URL de l'API de prédiction depuis :
    1) la variable d'environnement FRAUD_API_URL (si présente)
    2) sinon la Variable Airflow 'FRAUD_API_URL'
    """
    env_url = os.getenv("FRAUD_API_URL")
    if env_url:
        return env_url

    try:
        return Variable.get("FRAUD_API_URL")
    except Exception:
        raise AirflowFailException(
            "Aucune URL trouvée pour FRAUD_API_URL (ni env var, ni Variable Airflow)"
        )


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(seconds=30),
}


with DAG(
    dag_id="fraud_realtime_predict",
    start_date=datetime(2024, 1, 1),
    schedule="*/30 * * * *",  # toutes les 5 minutes
    catchup=False,
    default_args=default_args,
    tags=["fraude", "api", "realtime"],
) as dag:

    @task(task_id="call_realtime_predict")
    def call_realtime_predict_task():
        api_url = get_api_url()
        print(f"📡 Appel API : {api_url}")

        try:
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"❌ Erreur lors de l'appel API : {e}")
            raise AirflowFailException(f"API request failed: {e}")

        try:
            data = resp.json()
        except Exception:
            data = resp.text  # fallback

        print("✅ Réponse API :")
        print(data)

    call_realtime_predict_task()