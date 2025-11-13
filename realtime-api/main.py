import os, random
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List
from threading import Lock

from fastapi import FastAPI, APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Config
PAYMENTS_CSV = os.getenv(
    "PAYMENTS_CSV",
    "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv",
)
BATCH_DEFAULT = int(os.getenv("BATCH_DEFAULT", "200"))

# ── App & Static
app = FastAPI(title="Real-time Payments API (local)", version="1.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ── État en mémoire
_RT_DF: pd.DataFrame | None = None
_RT_IDX: int = 0
_RT_LOCK = Lock()

RT_COLUMNS = [
    "merchant","category","amt","gender","state","job","city_pop",
    "lat","long","merch_lat","merch_long","dob",
    "trans_date_trans_time","unix_time","cc_num","trans_num"
]

def _load_df():
    global _RT_DF
    df = pd.read_csv(PAYMENTS_CSV)
    # tri chronologique
    if "unix_time" in df.columns:
        df = df.sort_values("unix_time")
    elif "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df = df.sort_values("trans_date_trans_time")
    df = df.reset_index(drop=True)
    keep = [c for c in RT_COLUMNS if c in df.columns]
    if keep:
        df = df[keep]
    _RT_DF = df

@app.on_event("startup")
def startup():
    _load_df()

# ── Endpoints "compat" (1 seul record, comme l’API d’origine)
@app.get("/current-transactions")
@limiter.limit("5/minute")
async def current_transactions(request: Request):
    """
    Return one current transaction in .to_json(orient="split") format.
    """
    p = 0.001
    filename = "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"

    df = pd.read_csv(
        filename,
        header=0,
        index_col=[0],
        skiprows=lambda i: i > 0 and random.random() > p,
    )

    # On prend UNE ligne au hasard pour simuler du temps réel...
    df = df.sample(1)

    df = df.loc[:, ~df.columns.isin(["fraud"])]

    return df.to_json(orient="split", date_format="iso")

# ── Endpoints batch pour Airflow
class RTBatch(BaseModel):
    data: List[Dict[str, Any]]

realtime = APIRouter(prefix="/realtime", tags=["realtime"])

@realtime.get("/health")
def health():
    with _RT_LOCK:
        n = 0 if _RT_DF is None else len(_RT_DF)
        return {"status": "ok", "rows": n, "next_idx": _RT_IDX, "csv": PAYMENTS_CSV}

@realtime.post("/reset")
def reset(idx: int = 0):
    global _RT_IDX
    with _RT_LOCK:
        if _RT_DF is None:
            raise HTTPException(500, "Dataframe not loaded")
        if idx < 0 or idx >= len(_RT_DF):
            raise HTTPException(400, "invalid idx")
        _RT_IDX = idx
        return {"status": "reset", "idx": _RT_IDX}

@realtime.get("/payments/peek")
def peek(n: int = Query(10, ge=1, le=2000)):
    with _RT_LOCK:
        if _RT_DF is None:
            raise HTTPException(500, "Dataframe not loaded")
        i, j = _RT_IDX, min(_RT_IDX + n, len(_RT_DF))
        return {"idx": i, "preview": _RT_DF.iloc[i:j].to_dict(orient="records")}

@realtime.get("/payments/next", response_model=RTBatch)
def next_batch(n: int = Query(BATCH_DEFAULT, ge=1, le=5000)):
    global _RT_IDX
    with _RT_LOCK:
        if _RT_DF is None:
            raise HTTPException(500, "Dataframe not loaded")
        i = _RT_IDX
        if i >= len(_RT_DF):
            return RTBatch(data=[])
        j = min(i + n, len(_RT_DF))
        data = _RT_DF.iloc[i:j].to_dict(orient="records")
        _RT_IDX = j
        return RTBatch(data=data)

app.include_router(realtime)