# model_utils.py
import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_NAME = os.getenv("REGISTERED_NAME", "fraud-detection-xgb-pipeline")

def set_mlflow_uri() -> None:
    """
    Configure mlflow.set_tracking_uri depuis MLFLOW_TRACKING_URI.
    - lit d'abord l'env (docker --env-file .env),
    - sinon tente de charger un .env si présent.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        # on essaie quelques emplacements possibles du .env
        candidates = [
            Path("/app/.env"),
            Path(__file__).resolve().parents[0] / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(dotenv_path=p, override=True)
                uri = os.getenv("MLFLOW_TRACKING_URI")
                if uri:
                    break
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI manquant (variable d'env ou .env).")
    mlflow.set_tracking_uri(uri)


def load_champion_pipeline_and_threshold(
    registered_name: str | None = None,
) -> Tuple[object, float, object]:
    """
    Charge le pipeline 'champion' depuis le Model Registry MLflow et récupère le seuil 'best_threshold'.
    1) alias @champion
    2) tag champion:true
    3) stage Production
    """
    set_mlflow_uri()
    client = MlflowClient()
    name = registered_name or REGISTERED_NAME

    # 1) alias champion
    try:
        if hasattr(client, "get_model_version_by_alias"):
            mv = client.get_model_version_by_alias(name, "champion")
            pipe = mlflow.sklearn.load_model(f"models:/{name}@champion")
            thr = float(mv.tags["best_threshold"])
            return pipe, thr, mv
    except Exception:
        pass

    # 2) tag champion:true
    mv = None
    versions = client.search_model_versions(f"name='{name}'")
    for v in versions:
        if v.tags.get("champion", "").lower() == "true":
            mv = v
            break

    # 3) stage Production
    if mv is None:
        latest = client.get_latest_versions(name, stages=["Production"])
        if not latest:
            raise RuntimeError(
                f"Aucune version trouvée pour '{name}' (alias/tag/stage)."
            )
        mv = latest[0]

    # chargement par stage ou run_id
    try:
        uri = f"models:/{name}/{mv.current_stage}" if mv.current_stage else f"runs:/{mv.run_id}/model"
        pipe = mlflow.sklearn.load_model(uri)
    except Exception:
        pipe = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/model")

    if "best_threshold" not in mv.tags:
        raise RuntimeError(
            f"Tag 'best_threshold' absent sur la version champion (name={name}, version={mv.version})."
        )
    thr = float(mv.tags["best_threshold"])
    return pipe, thr, mv