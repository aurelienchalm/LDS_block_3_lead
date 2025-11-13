import os
import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 1. Chargement du .env et connexion NeonDB
# ─────────────────────────────────────────────
load_dotenv()  # .env à la racine

DATABASE_URL = os.getenv("POSTGRES_DATABASE")

if not DATABASE_URL:
    st.error("❌ Variable d'environnement POSTGRES_DATABASE manquante dans le .env")
    st.stop()


@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)


engine = get_engine()

# ─────────────────────────────────────────────
# 2. Fonctions utilitaires
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des transactions...")
def load_predictions_for_date(pred_date: dt.date, fraud_filter: str) -> pd.DataFrame:
    """
    Charge les lignes de fraud_realtime_predictions pour une date donnée
    en filtrant éventuellement sur la fraude.
    """
    base_query = """
        SELECT *
        FROM fraud_realtime_predictions
        WHERE DATE(prediction_time) = :pred_date
    """

    params = {"pred_date": pred_date}

    if fraud_filter == "Fraude uniquement":
        base_query += " AND is_fraud = TRUE"
    elif fraud_filter == "Transactions non frauduleuses":
        base_query += " AND is_fraud = FALSE"
    # sinon "Toutes" → pas de filtre supplémentaire

    with engine.connect() as conn:
        df = pd.read_sql(text(base_query), conn, params=params)

    return df


# ─────────────────────────────────────────────
# 3. UI Streamlit
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Monitoring Fraude – Temps réel",
    layout="wide",
)

st.title("🧾 Monitoring des transactions & détection de fraude")

st.markdown(
    """
Cette interface affiche les prédictions de fraude issues de l'API de scoring
et stockées dans la table **`fraud_realtime_predictions`** de NeonDB.
"""
)

# ── Filtres dans une sidebar
st.sidebar.header("🎛️ Filtres")

# Date de prédiction → par défaut : aujourd'hui
today = dt.date.today()
selected_date = st.sidebar.date_input(
    "Date de prédiction (prediction_time)",
    value=today,
    max_value=today,
)

fraud_filter = st.sidebar.selectbox(
    "Filtre fraude",
    [
        "Toutes",
        "Fraude uniquement",
        "Transactions non frauduleuses",
    ],
)

# Bouton pour recharger
if st.sidebar.button("🔄 Recharger"):
    st.cache_data.clear()

# ─────────────────────────────────────────────
# 4. Chargement des données
# ─────────────────────────────────────────────
df = load_predictions_for_date(selected_date, fraud_filter)

st.subheader("📅 Résumé des transactions")

if df.empty:
    st.warning(
        f"Aucune transaction trouvée pour la date **{selected_date}** "
        f"avec le filtre **{fraud_filter}**."
    )
else:
    # ── Quelques métriques en haut
    col1, col2, col3 = st.columns(3)

    total_tx = len(df)
    nb_fraudes = df["is_fraud"].sum() if "is_fraud" in df.columns else 0
    taux_fraude = nb_fraudes / total_tx * 100 if total_tx > 0 else 0.0

    col1.metric("Nombre de transactions", f"{total_tx}")
    col2.metric("Nombre de fraudes détectées", f"{nb_fraudes}")
    col3.metric("Taux de fraude", f"{taux_fraude:.2f} %")

    st.markdown("---")

    # ── Mise en forme du tableau
    # On essaie d'afficher les colonnes les plus utiles en premier
    colonnes_preferees = [
        "prediction_time",
        "transaction_time",
        "merchant",
        "category",
        "amt",
        "state",
        "city_pop",
        "proba_fraud",
        "is_fraud",
    ]

    cols_presentes = [c for c in colonnes_preferees if c in df.columns]
    autres_cols = [c for c in df.columns if c not in cols_presentes]

    df_affiche = df[cols_presentes + autres_cols]

    # Conversion pour un affichage plus propre
    if "prediction_time" in df_affiche.columns:
        df_affiche["prediction_time"] = pd.to_datetime(df_affiche["prediction_time"])

    if "transaction_time" in df_affiche.columns:
        df_affiche["transaction_time"] = pd.to_datetime(df_affiche["transaction_time"])

    if "proba_fraud" in df_affiche.columns:
        df_affiche["proba_fraud"] = df_affiche["proba_fraud"].astype(float)

    st.subheader("📊 Détail des transactions")

    st.dataframe(
        df_affiche,
        use_container_width=True,
        height=600,
    )