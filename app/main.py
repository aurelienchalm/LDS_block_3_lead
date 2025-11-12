import os
from typing import Any, Dict, List, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, RootModel

from model_utils import load_champion_pipeline_and_threshold, set_mlflow_uri, REGISTERED_NAME

from typing import Any, Dict, List, Union, Annotated
from fastapi import Body

EXAMPLE_RECORD = {
    "merchant": "fraud_Kilback LLC",
    "category": "gas_transport",
    "amt": 83.01,
    "gender": "F",
    "state": "TX",
    "job": "Film/video editor",
    "city_pop": 72011,
    "distance_km": 98.52,
    "age": 58,
    "year": 2020,
    "month": 12,
    "day_of_week": 5,
    "hour": 19,
    "is_weekend": 1
}

EXAMPLE_BATCH = {
    "data": [
        {
            "merchant": "fraud_Haley Group",
            "category": "misc_pos",
            "amt": 60.05,
            "gender": "M",
            "state": "FL",
            "job": "Set designer",
            "city_pop": 54767,
            "distance_km": 27.69,
            "age": 33,
            "year": 2020,
            "month": 6,
            "day_of_week": 6,
            "hour": 12,
            "is_weekend": 1
        },
        {
            "merchant": "fraud_Swaniawski, Nitzsche and Welch",
            "category": "health_fitness",
            "amt": 41.28,
            "gender": "F",
            "state": "NY",
            "job": "Librarian, public",
            "city_pop": 34496,
            "distance_km": 59.08,
            "age": 50,
            "year": 2020,
            "month": 6,
            "day_of_week": 6,
            "hour": 12,
            "is_weekend": 1
        }
    ]
}

class Record(RootModel[Dict[str, Any]]):
    model_config = {"json_schema_extra": {"example": EXAMPLE_RECORD}}

class Batch(BaseModel):
    data: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = {"json_schema_extra": {"example": EXAMPLE_BATCH}}

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="Serveur FastAPI pour le modèle champion MLflow de détection de fraude.",
)

PIPELINE = None
BEST_THRESHOLD: float = 0.5
MODEL_VERSION = None


@app.on_event("startup")
def startup_event():
    global PIPELINE, BEST_THRESHOLD, MODEL_VERSION
    set_mlflow_uri()
    PIPELINE, BEST_THRESHOLD, MODEL_VERSION = load_champion_pipeline_and_threshold(REGISTERED_NAME)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": REGISTERED_NAME,
        "version": getattr(MODEL_VERSION, "version", None),
        "stage": getattr(MODEL_VERSION, "current_stage", None),
        "threshold": BEST_THRESHOLD,
    }


@app.post("/reload")
def reload_model():
    """Recharge le champion (utile après promotion d'un nouveau modèle)."""
    global PIPELINE, BEST_THRESHOLD, MODEL_VERSION
    try:
        PIPELINE, BEST_THRESHOLD, MODEL_VERSION = load_champion_pipeline_and_threshold(REGISTERED_NAME)
        return {
            "status": "reloaded",
            "model_name": REGISTERED_NAME,
            "version": getattr(MODEL_VERSION, "version", None),
            "stage": getattr(MODEL_VERSION, "current_stage", None),
            "threshold": BEST_THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload error: {e}")


@app.post("/predict")
def predict(
    payload: Annotated[
        Union[Record, Batch],
        Body(
            examples={
                "single_record": {
                    "summary": "Une transaction",
                    "value": EXAMPLE_RECORD,
                },
                "batch_of_two": {
                    "summary": "Batch de 2 transactions",
                    "value": EXAMPLE_BATCH,
                },
            }
        ),
    ]
):
    try:
        #Pydantic v2 : accéder au contenu RootModel via .root
        if isinstance(payload, Record):
            df = pd.DataFrame([payload.root])
        else:
            if not payload.data:
                raise HTTPException(status_code=400, detail="payload.data est vide")
            df = pd.DataFrame(payload.data)

        probs = PIPELINE.predict_proba(df)[:, 1]
        preds = (probs >= BEST_THRESHOLD).astype(int)

        results = [
            {
                "is_fraud": int(p),
                "probability": float(proba),
                "threshold": float(BEST_THRESHOLD),
                "model_name": REGISTERED_NAME,
                "model_version": getattr(MODEL_VERSION, "version", None),
                "stage": getattr(MODEL_VERSION, "current_stage", None),
            }
            for p, proba in zip(preds, probs)
        ]
        return results if len(results) > 1 else results[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=bool(os.getenv("RELOAD", "")))