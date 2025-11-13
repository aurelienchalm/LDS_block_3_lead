FROM python:3.11-slim

# 1) Dossier de travail
WORKDIR /app_streamlit

# 2) Installer dépendances système nécessaires à psycopg2 & co
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3) Copier requirements et installer
COPY requirements.txt /app_streamlit/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copier l'ensemble du code Streamlit
COPY . /app_streamlit

# 5) Exposer le port Streamlit
EXPOSE 8501

# 6) Commande de lancement
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]