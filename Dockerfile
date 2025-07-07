FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer un lien symbolique pour python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Installation de llama-cpp-python pré-compilé pour CUDA 12.1
RUN echo "Installing llama-cpp-python with CUDA support..." && \
    pip install llama-cpp-python==0.2.90 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Installation des dépendances FastAPI et métriques
RUN echo "Installing FastAPI and dependencies..." && \
    pip install fastapi uvicorn[standard] pydantic httpx \
    prometheus-client psutil pynvml

# Créer le répertoire de travail
WORKDIR /app

# Créer le répertoire pour le modèle (vide pour l'instant)
RUN mkdir -p /app/models

# Copier l'application
COPY app.py /app/

# Informations sur l'image ### MODIFIÉ ###
RUN echo "=== Docker image build completed ===" && \
    echo "Model: Mixtral-8x7B-Instruct will be downloaded on first start" && \
    echo "Download size: ~32.9 GB" && \
    echo "API will be available on port 8000 after model download" && \
    echo "Metrics available at /metrics endpoint"

# Exposer le port
EXPOSE 8000

# Commande de démarrage ### MODIFIÉ ###
CMD echo "Starting Mixtral-8x7B API server..." && \
    echo "Note: Model will be downloaded on first start (~32.9 GB)" && \
    echo "This may take 20-40 minutes depending on connection speed" && \
    uvicorn app:app --host 0.0.0.0 --port 8000