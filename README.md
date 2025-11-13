# 🥷 Détection Automatique de Fraudes — MLflow & API Temps Réel

## 🎯 Objectif du projet
Ce projet vise à **détecter automatiquement les transactions frauduleuses** à partir de données financières et d’un flux de paiements temps réel.

Le pipeline complet combine :
- un **entraînement de modèle XGBoost** avec pipeline de preprocessing sklearn,
- une **expérimentation et versioning via MLflow**,
- une **API FastAPI de prédiction** déployable en container Docker,
- une **API temps réel de simulation de transactions**,
- et une intégration prête pour Airflow (future automatisation ETL).

---

## 🧱 Architecture du projet

```
fraud-detection/
│
├── data/
│   └── fraudTest.csv             # Dataset de référence
│
├── app/
│   ├── main.py                   # API FastAPI (endpoint /predict & /realtime-predict)
│   ├── model_utils.py            # Fonctions de chargement du modèle champion depuis MLflow
│   ├── requirements.txt          # Dépendances FastAPI + MLflow + XGBoost
│   └── Dockerfile                # Image Docker pour déploiement FastAPI
│
├── realtime-api/
│   ├── main.py                   # API simulant le flux de paiements en temps réel
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   └── style.css
│   ├── requirements.txt          # Dépendances pour SlowAPI + Jinja2
│   └── Dockerfile
│
├── csv_to_neondb.ipynb     # chargement en db du csv de train
├── fraud_detection.ipynb  # Entraînement + logging MLflow
│
├── .env                         # Variables d'environnement (MLflow, NeonDB, etc.)
└── README.md
```

---

## 🧠 Entraînement du modèle

### Dataset : `fraudTest.csv`
Ce dataset contient des transactions avec la variable cible :
```python
is_fraud ∈ {0, 1}
```
- 0 → transaction normale  
- 1 → transaction frauduleuse (≈ 0.39% du total)

### Étapes de préparation :
- Nettoyage des doublons et NaN  
- Conversion des dates (`trans_date_trans_time`, `dob`)  
- Création de variables dérivées :
  - `age`
  - `year`, `month`, `day_of_week`, `hour`, `is_weekend`
  - `distance_km` (calcul Haversine entre client et marchand)
- Encodage catégoriel et normalisation numérique via `ColumnTransformer`

### Modèle utilisé : `XGBClassifier`
```python
XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio_neg_to_pos,
    eval_metric="logloss",
)
```

- Recherche du **seuil optimal** (`best_threshold`) maximisant le F1-score
- Sauvegarde du pipeline complet dans **MLflow** (préprocessing + modèle)
- Ajout de tags :
  - `champion: true`
  - `best_threshold: X.XX`
- Alias MLflow : `@champion`

---

## 📦 Intégration MLflow

### Tracking configuré via `.env`
```
MLFLOW_TRACKING_URI=http://<ec2-mlflow>:5000
```

### Exemple de log
```python
with mlflow.start_run(run_name="xgboost_fraud_v1") as run:
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model",
        registered_model_name="fraud-detection-xgb-pipeline")
```

### Chargement du modèle champion :
```python
from model_utils import load_champion_pipeline_and_threshold
PIPELINE, BEST_THRESHOLD = load_champion_pipeline_and_threshold()
```

---

## 🚀 API FastAPI — Prédiction

### Endpoints disponibles

#### `GET /health`
→ Vérifie que le service est en ligne.

#### `POST /predict`
→ Fait une prédiction de fraude sur un enregistrement fourni manuellement (JSON).

#### `GET /realtime-predict`
→ Récupère une transaction depuis l’API temps réel et renvoie la prédiction.

### Exemple de réponse :
```json
{
  "merchant": "fraud_Weimann-Lockman",
  "amount": 56.9,
  "proba_fraud": 0.9123,
  "is_fraud": true,
  "threshold": 0.94
}
```

---

## 🌐 API Temps Réel (simulation)

**Nom :** `realtime-api`

Cette API renvoie une transaction simulée (prélevée aléatoirement du dataset d’origine).

- Endpoint : `GET /current-transactions`
- Format JSON (type “split”) :
```json
{
  "columns": ["merchant","category","amt","gender","state","job","city_pop","lat","long","merch_lat","merch_long","dob","cc_num","trans_num","current_time"],
  "data": [["fraud_Kirlin and Sons","personal_care",2.86,"M","CA","Therapist, art",351,34.99,-106.06,34.01,-106.56,"1977-03-23",1234567890123,"a1b2c3d4",1762965719]]
}
```

Utilisation :
```bash
curl http://localhost:7871/current-transactions
```

**URL :** `http://localhost:7871/docs`

---

## 🐳 Docker — Build & Run

### 1️⃣ API Temps réel
#### en local
```bash
docker build -t realtime-api \
    -f realtime-api/Dockerfile \
    realtime-api
docker rm -f realtime-api 2>/dev/null || true
docker run -p 7871:8001 --name realtime-api realtime-api
```

#### sur EC2 
```bash
docker network create fraud-net 2>/dev/null || true
docker build -t realtime-api \
    -f realtime-api/Dockerfile \
    realtime-api
docker rm -f realtime-api 2>/dev/null || true
docker run -d \
  --name realtime-api \
  --restart always \
  --network fraud-net \
  -p 7871:8001 \
  realtime-api
```

**URL :** `http://ip:7871/docs`

### 2️⃣ API de prédiction
#### en local
```bash
docker build -t fraud-app \
    -f app/Dockerfile \
    app
docker rm -f fraud-app 2>/dev/null || true
docker run --env-file .env -p 7860:8000 --name fraud-app fraud-app
```

#### sur EC2 
```bash
docker build -t fraud-app \
    -f app/Dockerfile \
    app
docker rm -f fraud-app 2>/dev/null || true
docker run -d \
  --name fraud-app \
  --restart always \
  --network fraud-net \
  --env-file .env \
  -p 7860:8000 \
  fraud-app
```

**URL :** `http://ip:7860/docs`

---

## 🔗 Intégration entre les deux APIs
Le `REALTIME_API_URL` est passé dans `.env` en local :
```
REALTIME_API_URL=http://host.docker.internal:7871/current-transactions
```

Sur EC2 : 
```
REALTIME_API_URL=http://realtime-api:8001/current-transactions
```

---

## 🚀 Streamlit : interface de consulatation des fraudes

### 🐳 Docker — Build & Run

```bash
docker build -t fraud-streamlit \
    -f app_streamlit/Dockerfile \
    app_streamlit

docker rm -f fraud-streamlit 2>/dev/null || true

docker run -d \
  --name fraud-streamlit \
  --restart always \
  --env-file .env \
  -p 8501:8501 \
  fraud-streamlit
```

Sur EC2 pour arrêter tout : 

```bash
docker rm -f realtime-api
docker rm -f fraud-app
docker rm -f fraud-streamlit
```

---

## 📊 Résultats du modèle (XGBoost)

| Jeu | F1-score | Precision | Recall | Threshold |
|-----|-----------|------------|---------|------------|
| Train | 0.97 | 0.94 | 0.99 | 0.94 |
| Test  | 0.86 | 0.88 | 0.83 | 0.94 |

---

## 🧩 Prochaines étapes
- Intégration dans **Airflow** (déclenchement automatique via DAG)
- Monitoring des performances avec **Evidently**
- Stockage NeonDB des prédictions temps réel
- Ajout d’une interface Streamlit pour visualiser les alertes fraude

---

## 👨‍💻 Auteur
**Aurélien Chalm**  
Lead Data Science & Engineering — Certification Jedha  
Projet : *Automatic Fraud Detection (ETL with Airflow)*  
Stack : `Python • Scikit-learn • XGBoost • MLflow • FastAPI • Docker • NeonDB • Airflow`
