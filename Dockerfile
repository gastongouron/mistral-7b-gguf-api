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

# Créer le répertoire de travail ET le répertoire des modèles
WORKDIR /app
RUN mkdir -p /app/models && \
    echo "Created directory: /app/models" && \
    ls -la /app/

# Télécharger Qwen2.5-32B-Instruct - Version GGUF par bartowski
# Q4_K_M pour un bon équilibre qualité/taille
RUN echo "=== Downloading Qwen2.5-32B-Instruct Q4_K_M ===" && \
    echo "Model optimized for structured JSON output and multilingual support" && \
    echo "File size: ~18.4 GB" && \
    echo "This may take 10-20 minutes depending on connection speed..." && \
    cd /app/models && \
    wget --progress=dot:giga \
         --show-progress \
         --timeout=600 \
         --tries=3 \
         -O Qwen2.5-32B-Instruct-Q4_K_M.gguf \
         "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf" && \
    echo "=== Download completed successfully ===" && \
    echo "Model size:" && \
    ls -lh /app/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf

# Vérifier l'intégrité du fichier
RUN echo "=== Verifying model file ===" && \
    if [ ! -f /app/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf ]; then \
        echo "ERROR: Model file not found!" && exit 1; \
    fi && \
    FILE_SIZE=$(stat -c%s /app/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf) && \
    echo "File size in bytes: $FILE_SIZE" && \
    if [ $FILE_SIZE -lt 10000000000 ]; then \
        echo "ERROR: File seems too small, download may have failed!" && exit 1; \
    fi && \
    echo "=== Model file verified successfully ==="

# Copier l'application
COPY app.py /app/

# Afficher les informations finales
RUN echo "=== Docker image build completed ===" && \
    echo "Model: Qwen2.5-32B-Instruct Q4_K_M (bartowski)" && \
    echo "Optimized for: Structured JSON output, French medical conversations" && \
    echo "Location: /app/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf" && \
    echo "API will be available on port 8000" && \
    echo "Metrics available at /metrics endpoint" && \
    echo "" && \
    echo "Directory contents:" && \
    ls -la /app/models/

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD echo "Starting Qwen2.5-32B API server..." && \
    echo "Model loading may take 30-60 seconds..." && \
    echo "Optimized for medical French conversations with JSON output" && \
    echo "Recommended GPU: 24GB+ VRAM" && \
    echo "" && \
    echo "Checking model file..." && \
    ls -lh /app/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf && \
    echo "" && \
    uvicorn app:app --host 0.0.0.0 --port 8000