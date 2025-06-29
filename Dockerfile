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

# Installation des autres dépendances
RUN echo "Installing FastAPI and dependencies..." && \
    pip install fastapi uvicorn[standard] pydantic httpx

# Créer le répertoire de travail
WORKDIR /app

# Créer le répertoire pour le modèle
RUN mkdir -p /app/models

# Télécharger Gemma-2-9B - Excellent pour JSON et catégorisation
# Q5_K_M pour meilleure qualité sur les tâches de classification
RUN echo "=== Downloading Gemma-2-9B-IT Q5_K_M ===" && \
    echo "Model optimized for structured outputs and categorization" && \
    echo "File size: ~6.5 GB" && \
    echo "This may take 5-15 minutes depending on connection speed..." && \
    wget --progress=dot:giga \
         --show-progress \
         --timeout=300 \
         --tries=3 \
         -O /app/models/gemma-2-9b-it-Q5_K_M.gguf \
         https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q5_K_M.gguf && \
    echo "=== Download completed successfully ===" && \
    echo "Model size:" && \
    ls -lh /app/models/gemma-2-9b-it-Q5_K_M.gguf

# Vérifier l'intégrité du fichier
RUN echo "=== Verifying model file ===" && \
    if [ ! -f /app/models/gemma-2-9b-it-Q5_K_M.gguf ]; then \
        echo "ERROR: Model file not found!" && exit 1; \
    fi && \
    FILE_SIZE=$(stat -c%s /app/models/gemma-2-9b-it-Q5_K_M.gguf) && \
    echo "File size in bytes: $FILE_SIZE" && \
    if [ $FILE_SIZE -lt 1000000000 ]; then \
        echo "ERROR: File seems too small, download may have failed!" && exit 1; \
    fi && \
    echo "=== Model file verified successfully ==="

# Copier l'application
COPY app.py /app/

# Afficher les informations finales
RUN echo "=== Docker image build completed ===" && \
    echo "Model: Gemma-2-9B-IT Q5_K_M" && \
    echo "Optimized for: JSON outputs, categorization, date extraction" && \
    echo "Location: /app/models/gemma-2-9b-it-Q5_K_M.gguf" && \
    echo "API will be available on port 8000"

# Exposer le port
EXPOSE 8000

# Commande de démarrage avec logs
CMD echo "Starting Gemma-2-9B API server..." && \
    echo "Model loading may take 20-40 seconds..." && \
    echo "Optimized for categorization and JSON outputs" && \
    uvicorn app:app --host 0.0.0.0 --port 8000