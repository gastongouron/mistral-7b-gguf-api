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
RUN pip install llama-cpp-python==0.2.90 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Installation des autres dépendances
RUN pip install fastapi uvicorn[standard] pydantic httpx

# Créer le répertoire de travail
WORKDIR /app

# Créer le répertoire pour le modèle
RUN mkdir -p /app/models

# CHANGEMENT ICI : Télécharger Qwen2.5-14B au lieu de Mistral
# Note: Le fichier Q4_K_M est divisé en 2 parties
RUN wget -O /app/models/qwen2.5-14b-instruct-q4_k_m-00001-of-00002.gguf \
    https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00001-of-00002.gguf && \
    wget -O /app/models/qwen2.5-14b-instruct-q4_k_m-00002-of-00002.gguf \
    https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00002-of-00002.gguf

# Installer llama-gguf-split pour fusionner les fichiers
RUN git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    make llama-gguf-split && \
    cp llama-gguf-split /usr/local/bin/ && \
    rm -rf /tmp/llama.cpp

# Fusionner les fichiers GGUF
RUN cd /app/models && \
    llama-gguf-split --merge qwen2.5-14b-instruct-q4_k_m-00001-of-00002.gguf qwen2.5-14b-instruct-q4_k_m.gguf && \
    rm qwen2.5-14b-instruct-q4_k_m-00001-of-00002.gguf qwen2.5-14b-instruct-q4_k_m-00002-of-00002.gguf

# Copier l'application
COPY app.py /app/

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]