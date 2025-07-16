FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ------------------------------------------------------------------
# Env de base
# ------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ------------------------------------------------------------------
# Paquets système
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv python3-distutils \
    git wget curl ca-certificates gnupg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Lien python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# ------------------------------------------------------------------
# Dépendances Python
# NOTE: Si tu préfères, copie requirements.txt et fais pip install -r.
# ------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.2.90" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    httpx \
    prometheus-client \
    psutil \
    pynvml \
    "huggingface_hub[cli]" \
    typing_extensions

# ------------------------------------------------------------------
# Dirs
# ------------------------------------------------------------------
WORKDIR /app
RUN mkdir -p /workspace/models

# ------------------------------------------------------------------
# Copier l'application (utilise *ton* fichier proxy optimisé)
# Nom de fichier: app.py pour que 'uvicorn app:app' fonctionne.
# ------------------------------------------------------------------
COPY app.py /app/

# ------------------------------------------------------------------
# Infos build
# ------------------------------------------------------------------
RUN echo '=== Docker image build completed ===' && \
    echo 'Model: Qwen2.5-32B-Instruct-Q6_K (≈25 GB) will be downloaded on first start' && \
    echo 'Set HF_TOKEN at runtime for authenticated/accelerated download (optional)' && \
    echo 'API: port 8000' && \
    echo 'Metrics: /metrics' && \
    echo 'Uvicorn workers=1 (single shared model)'

EXPOSE 8000

# ------------------------------------------------------------------
# Entrée
# ------------------------------------------------------------------
CMD echo 'Starting Qwen2.5-32B Proxy API server...' && \
    (nvidia-smi || echo 'No GPU detected; fallback CPU (slow)') && \
    echo '' && \
    echo 'Note: Model (~25 GB) downloaded/cached under /workspace/models on first run.' && \
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
