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

# Créer le répertoire pour le modèle (utilise /workspace/models pour compatibilité RunPod)
RUN mkdir -p /workspace/models

# Copier l'application
COPY app.py /app/

# Informations sur l'image
RUN echo "=== Docker image build completed ===" && \
    echo "Model: Qwen2.5-32B-Instruct-Q8_0 will be downloaded on first start" && \
    echo "Download size: ~34 GB (Q8_0 - highest quality)" && \
    echo "API will be available on port 8000 after model download" && \
    echo "Optimized for NVIDIA L40 48GB GPU" && \
    echo "Metrics available at /metrics endpoint"

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD echo "Starting Qwen2.5-32B API server..." && \
    echo "GPU: Checking for NVIDIA GPU..." && \
    nvidia-smi || echo "No GPU detected, will run on CPU" && \
    echo "" && \
    echo "Note: Model will be downloaded on first start (~34 GB)" && \
    echo "This may take 20-30 minutes depending on connection speed" && \
    echo "Subsequent starts will be much faster (model cached)" && \
    echo "" && \
    uvicorn app:app --host 0.0.0.0 --port 8000