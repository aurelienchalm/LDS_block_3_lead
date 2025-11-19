from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.hooks.http import HttpHook
from airflow.hooks.base import BaseHook
from airflow.operators.email import EmailOperator

from datetime import datetime
import requests
import time


JENKINS_JOB_NAME = "fraud_test_train"  


def send_failure_email(context):
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id
    task_id = task_instance.task_id
    log_url = task_instance.log_url

    subject = f"❌ ECHEC du DAG {dag_id} - tâche {task_id}"
    body = f"""
    Le DAG <b>{dag_id}</b> a échoué sur la tâche <b>{task_id}</b>.<br>
    👉 <a href="{log_url}">Voir les logs Airflow</a>
    """

    email_task = EmailOperator(
        task_id='send_failure_email',
        to='aurelien.chalm@gmail.com',
        subject=subject,
        html_content=body
    )
    email_task.execute(context=context)


def trigger_jenkins_train_job(**context):
    """
    Appelle Jenkins via l’API pour lancer le job de ré-entrainement
    et attend la fin du build.
    """
    # 🔐 Connexion Jenkins à configurer dans Airflow :
    # Conn Id : jenkins_api
    conn = BaseHook.get_connection("jenkins_api")
    username = conn.login
    password = conn.password
    base_url = conn.host.rstrip("/")

    # 1️⃣ Récupération du crumb CSRF
    crumb_resp = requests.get(
        f"{base_url}/crumbIssuer/api/json",
        auth=(username, password),
        timeout=10,
    )
    crumb_resp.raise_for_status()
    crumb_data = crumb_resp.json()

    headers = {
        crumb_data["crumbRequestField"]: crumb_data["crumb"],
        "Content-Type": "application/json",
    }

    # 2️⃣ Déclenchement du job Jenkins
    build_resp = requests.post(
        f"{base_url}/job/{JENKINS_JOB_NAME}/build",
        auth=(username, password),
        headers=headers,
        timeout=10,
    )

    if build_resp.status_code != 201:
        raise Exception(
            f"❌ Erreur lors du déclenchement du job Jenkins "
            f"{JENKINS_JOB_NAME} : {build_resp.status_code}"
        )

    queue_url = build_resp.headers.get("Location")
    if not queue_url:
        raise Exception("❌ Impossible de récupérer l’URL de queue Jenkins")

    print(f"📥 Job {JENKINS_JOB_NAME} en queue : {queue_url}")

    # 3️⃣ Récupération du numéro de build
    build_number = None
    for _ in range(30):  # ~60s max (30 * 2s)
        queue_resp = requests.get(
            f"{queue_url}api/json",
            auth=(username, password),
            timeout=10,
        )
        queue_resp.raise_for_status()
        queue_data = queue_resp.json()

        if "executable" in queue_data and "number" in queue_data["executable"]:
            build_number = queue_data["executable"]["number"]
            break

        time.sleep(2)

    if build_number is None:
        raise Exception("❌ Timeout : Jenkins n’a pas attribué de numéro de build")

    print(f"⏳ Build #{build_number} en cours sur {JENKINS_JOB_NAME}...")

    # 4️⃣ Polling du statut du build
    jenkins_build_url = f"{base_url}/job/{JENKINS_JOB_NAME}/{build_number}/"

    for _ in range(60):  # ~5 minutes (60 * 5s)
        build_info_resp = requests.get(
            f"{jenkins_build_url}api/json",
            auth=(username, password),
            timeout=10,
        )
        build_info_resp.raise_for_status()
        build_info = build_info_resp.json()

        if not build_info.get("building", False):
            result = build_info.get("result")
            print(f"✅ Résultat du build Jenkins : {result}")
            print(f"🔗 Voir le build sur Jenkins : {jenkins_build_url}")
            if result != "SUCCESS":
                raise Exception(f"❌ Le job Jenkins a échoué : {result}")
            break

        time.sleep(5)

    return f"✔️ Build #{build_number} terminé avec succès ({JENKINS_JOB_NAME})"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="trigger_jenkins_fraud_retrain",
    default_args=default_args,
    start_date=datetime(2025, 11, 17),
    schedule_interval=None,   # déclenché à la main ou via un autre DAG (drift)
    catchup=False,
    tags=["jenkins", "fraud", "retrain"],
    on_failure_callback=send_failure_email,
) as dag:

    trigger_train = PythonOperator(
        task_id="trigger_jenkins_fraud_train_build",
        python_callable=trigger_jenkins_train_job,
    )