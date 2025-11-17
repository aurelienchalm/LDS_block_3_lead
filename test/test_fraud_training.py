# test/test_fraud_training.py

import os
from pathlib import Path
import sys

# Ajouter le chemin du projet (racine) dans sys.path
# /app/test/test_fraud_training.py → on remonte d'un niveau vers /app
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fraud_training import main

def test_fraud_training_runs():
    """
    Test d'intégration simple :
    - Vérifie que le script de ré-entrainement s'exécute sans lever d'exception.
    - Si ça se termine sans erreur → test OK.
    """
    main()


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