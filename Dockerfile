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
RUN mkdir -p /workspace/models

# Copier l'application
COPY app.py /app/

# Script de nettoyage
RUN echo '#!/bin/bash\n\
echo "🧹 Nettoyage des anciens modèles..."\n\
if [ -d "/workspace/models" ]; then\n\
    echo "📊 Contenu actuel du dossier models:"\n\
    ls -lah /workspace/models/\n\
    echo ""\n\
    # Supprimer tous les fichiers .gguf sauf Qwen2.5-72B\n\
    find /workspace/models -name "*.gguf" ! -name "*Qwen2.5-72B*" -type f -exec rm -v {} \;\n\
    echo ""\n\
    echo "💾 Espace disque après nettoyage:"\n\
    df -h /workspace\n\
fi\n\
echo "✅ Nettoyage terminé"\n\
echo ""' > /app/cleanup.sh && chmod +x /app/cleanup.sh

# Informations sur l'image
RUN echo "=== Docker image build completed ===" && \
    echo "Model: Qwen2.5-72B-Instruct-Q3_K_M will be downloaded on first start" && \
    echo "Download size: ~40 GB (Q3_K_M quantization)" && \
    echo "API will be available on port 8000 after model download" && \
    echo "Metrics available at /metrics endpoint" && \
    echo "Auto-cleanup of old models enabled!"

# Exposer le port
EXPOSE 8000

# Commande de démarrage avec nettoyage
CMD echo "Starting Qwen2.5-72B API server..." && \
    /app/cleanup.sh && \
    echo "Note: Model will be downloaded on first start (~40 GB)" && \
    echo "This may take 15-30 minutes depending on connection speed" && \
    uvicorn app:app --host 0.0.0.0 --port 8000