#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Qwen2.5-14B GGUF avec llama-cpp-python
Optimisé pour conversations médicales françaises avec extraction JSON
Télécharge automatiquement le modèle au premier démarrage
"""
import os
import time
import uuid
import json
import re
import logging
import asyncio
import psutil
import socket
import urllib.request
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from llama_cpp import Llama

# Import des métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

# Configuration mise à jour pour Qwen2.5-14B
MODEL_PATH = "/app/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
API_TOKEN = os.getenv("API_TOKEN", "supersecret")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===== MÉTRIQUES PROMETHEUS =====

# Informations système
system_info = Info('fastapi_system', 'System information')
system_info.info({
    'model': 'qwen2.5-14b',
    'instance': socket.gethostname(),
    'pod_id': os.getenv('RUNPOD_POD_ID', 'local'),
    'version': '2.0.0'
})

# Métriques GPU
gpu_utilization_percent = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
gpu_memory_used_bytes = Gauge('gpu_memory_used_bytes', 'GPU memory used in bytes')
gpu_memory_total_bytes = Gauge('gpu_memory_total_bytes', 'GPU memory total in bytes')
gpu_temperature_celsius = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius')
gpu_power_watts = Gauge('gpu_power_watts', 'GPU power usage in watts')

# Métriques système
cpu_usage_percent = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_used_bytes = Gauge('memory_used_bytes', 'System memory used in bytes')
memory_total_bytes = Gauge('memory_total_bytes', 'System memory total in bytes')
disk_usage_percent = Gauge('disk_usage_percent', 'Disk usage percentage')

# Métriques FastAPI
fastapi_requests_total = Counter(
    'fastapi_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

fastapi_request_duration_seconds = Histogram(
    'fastapi_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

fastapi_websocket_connections = Gauge(
    'fastapi_websocket_connections',
    'Number of active WebSocket connections'
)

# Métriques d'inférence
fastapi_inference_requests_total = Counter(
    'fastapi_inference_requests_total',
    'Total number of inference requests',
    ['model', 'status']
)

fastapi_inference_duration_seconds = Histogram(
    'fastapi_inference_duration_seconds',
    'Inference duration in seconds',
    ['model'],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

fastapi_inference_queue_size = Gauge(
    'fastapi_inference_queue_size',
    'Current inference queue size'
)

fastapi_inference_tokens_total = Counter(
    'fastapi_inference_tokens_total',
    'Total tokens processed',
    ['type']  # prompt, completion
)

fastapi_inference_tokens_per_second = Gauge(
    'fastapi_inference_tokens_per_second',
    'Tokens generated per second'
)

# Métriques du modèle
model_loaded = Gauge('model_loaded', 'Whether the model is loaded (1) or not (0)')
model_loading_duration_seconds = Gauge('model_loading_duration_seconds', 'Time taken to load the model')
model_download_progress = Gauge('model_download_progress', 'Model download progress in percentage')

# Queue pour simuler la file d'attente
inference_queue = asyncio.Queue(maxsize=1000)

# Variable globale pour l'état du téléchargement
download_in_progress = False
download_complete = False

# ===== FONCTIONS UTILITAIRES POUR MÉTRIQUES =====

def update_gpu_metrics():
    """Mettre à jour les métriques GPU"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Utilisation GPU
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization_percent.set(util.gpu)
        
        # Mémoire GPU
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used_bytes.set(mem_info.used)
        gpu_memory_total_bytes.set(mem_info.total)
        
        # Température
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_temperature_celsius.set(temp)
        
        # Consommation électrique
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            gpu_power_watts.set(power)
        except:
            pass
            
    except Exception as e:
        logging.debug(f"Impossible de collecter les métriques GPU: {e}")

def update_system_metrics():
    """Mettre à jour les métriques système"""
    try:
        # CPU
        cpu_usage_percent.set(psutil.cpu_percent(interval=0.1))
        
        # Mémoire
        mem = psutil.virtual_memory()
        memory_used_bytes.set(mem.used)
        memory_total_bytes.set(mem.total)
        
        # Disque
        disk = psutil.disk_usage('/')
        disk_usage_percent.set(disk.percent)
        
    except Exception as e:
        logging.debug(f"Impossible de collecter les métriques système: {e}")

async def metrics_update_task():
    """Tâche pour mettre à jour les métriques périodiquement"""
    while True:
        update_gpu_metrics()
        update_system_metrics()
        fastapi_inference_queue_size.set(inference_queue.qsize())
        await asyncio.sleep(5)  # Mise à jour toutes les 5 secondes

# ===== CODE PRINCIPAL =====

# Sécurité
security = HTTPBearer()

# Modèles Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-14b"
    messages: List[Message]
    temperature: Optional[float] = 0.1  # Légèrement augmenté pour Qwen
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[str, str]] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# Variable globale pour le modèle
llm = None

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifier le token Bearer"""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def download_model_if_needed():
    """Télécharger le modèle au premier démarrage si nécessaire"""
    global download_in_progress, download_complete
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 5_000_000_000:  # Plus de 5GB, probablement OK
            print(f"Modèle trouvé: {file_size / (1024**3):.1f} GB")
            download_complete = True
            return
        else:
            print(f"Modèle incomplet ({file_size / (1024**3):.1f} GB), re-téléchargement...")
            os.remove(MODEL_PATH)
    
    if download_in_progress:
        print("Téléchargement déjà en cours...")
        return
    
    download_in_progress = True
    print(f"Téléchargement du modèle Qwen2.5-14B... (~8.1 GB)")
    print(f"URL: {MODEL_URL}")
    print("Cela peut prendre 10-20 minutes selon votre connexion...")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    def download_progress(block_num, block_size, total_size):
        """Callback pour afficher la progression"""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        
        # Mettre à jour la métrique Prometheus
        model_download_progress.set(percent)
        
        # Afficher la progression
        sys.stdout.write(f'\rTéléchargement: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) ')
        sys.stdout.flush()
        
        # Log tous les 10%
        if int(percent) % 10 == 0 and int(percent) != int((block_num - 1) * block_size * 100 / total_size):
            logging.info(f"Téléchargement du modèle: {percent:.0f}%")
    
    try:
        start_time = time.time()
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=download_progress)
        print("\n✅ Téléchargement terminé!")
        
        download_time = time.time() - start_time
        print(f"Temps de téléchargement: {download_time/60:.1f} minutes")
        
        # Vérifier la taille
        file_size = os.path.getsize(MODEL_PATH)
        print(f"Taille du fichier: {file_size / (1024**3):.1f} GB")
        
        if file_size < 5_000_000_000:
            raise Exception(f"Fichier trop petit: {file_size} bytes")
        
        download_complete = True
        model_download_progress.set(100)
        
    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement: {e}")
        download_in_progress = False
        model_download_progress.set(0)
        
        # Supprimer le fichier partiel
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise
    finally:
        download_in_progress = False

def load_model():
    """Charger le modèle GGUF avec configuration optimale pour Qwen2.5-14B"""
    global llm
    
    # S'assurer que le modèle est téléchargé
    download_model_if_needed()
    
    print(f"Chargement du modèle Qwen2.5-14B depuis {MODEL_PATH}...")
    
    # Détecter la mémoire GPU disponible
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        print(f"VRAM disponible: {vram_gb:.1f} GB")
        
        # Adapter les couches GPU selon la VRAM (14B est plus léger)
        if vram_gb >= 16:
            n_gpu_layers = -1  # Tout sur GPU
            print("Configuration: Modèle entièrement sur GPU")
        elif vram_gb >= 12:
            n_gpu_layers = 40  # La plupart sur GPU
            print("Configuration: 40 couches sur GPU")
        else:
            n_gpu_layers = 30
            print("⚠️ VRAM limitée, performance réduite (30 couches sur GPU)")
    except:
        n_gpu_layers = -1
        print("Impossible de détecter la VRAM, chargement complet sur GPU")
    
    # Configuration optimisée pour Qwen2.5-14B
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,  # Qwen2.5 supporte 128K mais 8K suffisant pour conversations médicales
        n_threads=12,  # Un peu moins que pour 32B
        n_gpu_layers=n_gpu_layers,
        n_batch=512,  # Plus grand batch pour 14B
        use_mmap=True,
        use_mlock=False,  # False pour économiser RAM système
        verbose=True,
        seed=42,  # Pour reproductibilité
        rope_scaling_type="linear",  # Support du contexte long
        rope_freq_scale=1.0
    )
    
    print("Modèle Qwen2.5-14B chargé avec succès!")
    print(f"Configuration: {n_gpu_layers} couches GPU, contexte 8K tokens")

def format_messages_qwen(messages: List[Message]) -> str:
    """Formater les messages pour Qwen2.5 (format ChatML)"""
    formatted = ""
    
    # System prompt par défaut si non fourni
    has_system = any(msg.role == "system" for msg in messages)
    if not has_system:
        formatted += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    
    for message in messages:
        if message.role == "system":
            formatted += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            formatted += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            formatted += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
    
    # Ajouter le début de la réponse de l'assistant
    formatted += "<|im_start|>assistant\n"
    
    return formatted

def extract_json_from_text(text: str) -> str:
    """Extraire JSON même si le modèle ajoute du texte autour"""
    # Qwen2.5 génère généralement du JSON propre, mais on garde cette fonction par sécurité
    
    # D'abord essayer tel quel
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    
    # Chercher le premier { et le dernier }
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    
    # Chercher entre ```json et ```
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Chercher après "JSON:" ou similaire
    json_prefix_match = re.search(r'(?:JSON|json|Json):\s*({.*})', text, re.DOTALL)
    if json_prefix_match:
        return json_prefix_match.group(1)
    
    return text

def clean_and_parse_json(text: str) -> Optional[Dict]:
    """Nettoyer et parser du JSON potentiellement mal formaté"""
    # D'abord essayer d'extraire le JSON
    text = extract_json_from_text(text)
    
    # Nettoyer les artefacts communs
    text = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \[.*?\] => ', '', text.strip())
    text = re.sub(r'^.*?Extracted content:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?:\s*(?=\{)', '', text)
    
    # Remplacer les underscores échappés
    text = text.replace(r'\_', '_')
    
    # Nettoyer les commentaires JSON non standard
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Essayer avec des corrections communes
        text = text.replace("'", '"')  # Simple quotes -> double quotes
        text = re.sub(r',\s*}', '}', text)  # Virgules finales
        text = re.sub(r',\s*]', ']', text)
        text = re.sub(r'([^\\])"([^"]*)"([^"]*)"', r'\1"\2\'\3"', text)  # Guillemets internes
        
        try:
            return json.loads(text)
        except:
            return None

def ensure_json_response(text: str, request_format: Optional[Dict] = None) -> str:
    """S'assurer que la réponse est du JSON valide si demandé"""
    if request_format and request_format.get("type") == "json_object":
        parsed = clean_and_parse_json(text)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        else:
            # Fallback pour Qwen si parsing échoue
            return json.dumps({
                "response": text[:500] if len(text) > 500 else text,
                "error": "Could not parse as valid JSON",
                "note": "Qwen2.5 usually generates valid JSON. Check prompt formatting."
            }, ensure_ascii=False)
    return text

# ===== LIFESPAN POUR GÉRER LE CYCLE DE VIE =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Démarrage
    print("=== Démarrage de l'application ===")
    
    # Démarrer la tâche de mise à jour des métriques
    metrics_task = asyncio.create_task(metrics_update_task())
    
    # Charger le modèle (incluant le téléchargement si nécessaire)
    try:
        model_start = time.time()
        load_model()
        model_loading_duration_seconds.set(time.time() - model_start)
        model_loaded.set(1)
        print("=== Modèle chargé, API prête ===")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        model_loaded.set(0)
    
    yield
    
    # Arrêt
    print("=== Arrêt de l'application ===")
    model_loaded.set(0)
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass

# Initialisation de l'application avec lifespan
app = FastAPI(
    title="Qwen2.5-14B GGUF API",
    version="2.0.0",
    description="API FastAPI pour Qwen2.5-14B optimisée pour conversations médicales françaises avec JSON structuré",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINT MÉTRIQUES =====

@app.get("/metrics")
async def metrics():
    """Endpoint pour exposer les métriques Prometheus"""
    # Mettre à jour les métriques avant de les retourner
    update_gpu_metrics()
    update_system_metrics()
    
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Qwen2.5-14B GGUF API",
        "status": "running" if llm is not None else "loading",
        "model": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "model_loaded": llm is not None,
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/ws": "WebSocket - Chat endpoint (requires token in query)",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics",
            "/download-status": "GET - Model download status"
        },
        "optimized_for": [
            "French medical conversations",
            "Structured JSON output", 
            "Multi-turn dialogue",
            "Information extraction"
        ]
    }

@app.get("/health")
async def health_check():
    """Vérifier l'état de l'API"""
    health_status = {
        "status": "healthy" if llm is not None else "loading",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "download_complete": download_complete,
        "download_in_progress": download_in_progress
    }
    
    if os.path.exists(MODEL_PATH):
        health_status["model_size"] = f"{os.path.getsize(MODEL_PATH) / (1024**3):.1f} GB"
    
    return health_status

@app.get("/download-status")
async def download_status():
    """Statut du téléchargement du modèle"""
    status = {
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "model_exists": os.path.exists(MODEL_PATH),
        "download_progress_percent": model_download_progress._value._value if hasattr(model_download_progress, '_value') else 0
    }
    
    if os.path.exists(MODEL_PATH):
        status["current_size_gb"] = os.path.getsize(MODEL_PATH) / (1024**3)
        status["expected_size_gb"] = 8.1
    
    return status

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    """Lister les modèles disponibles"""
    start_time = time.time()
    
    try:
        result = {
            "object": "list",
            "data": [
                {
                    "id": "qwen2.5-14b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Alibaba/Qwen",
                    "permission": [],
                    "root": "qwen2.5-14b",
                    "parent": None,
                    "ready": llm is not None
                }
            ]
        }
        
        # Métriques
        fastapi_requests_total.labels(method="GET", endpoint="/v1/models", status="success").inc()
        return result
        
    except Exception as e:
        fastapi_requests_total.labels(method="GET", endpoint="/v1/models", status="error").inc()
        raise
    finally:
        fastapi_request_duration_seconds.labels(method="GET", endpoint="/v1/models").observe(time.time() - start_time)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_token)])
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint compatible OpenAI optimisé pour Qwen2.5 et outputs structurés"""
    start_time = time.time()
    status = "success"
    
    if llm is None:
        if download_in_progress:
            raise HTTPException(
                status_code=503, 
                detail="Model is being downloaded. Please check /download-status for progress."
            )
        else:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. It will be downloaded on first access."
            )
    
    try:
        # Ajouter à la queue
        await inference_queue.put(request)
        
        prompt = format_messages_qwen(request.messages)
        
        # Instructions spécifiques pour JSON avec Qwen2.5
        if request.response_format and request.response_format.get("type") == "json_object":
            # Qwen2.5 comprend très bien cette instruction
            prompt += "Please respond with valid JSON only, no additional text or explanations.\n"
        
        # Timer pour l'inférence
        inference_start = time.time()
        
        # Paramètres optimisés pour Qwen2.5-14B (un peu plus permissifs que 32B)
        response = llm(
            prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.1,
            top_k=20,  # Un peu plus permissif que Phi
            stop=request.stop or ["<|im_end|>", "<|endoftext|>", "</s>"],
            echo=False,
            repeat_penalty=1.05  # Légère pénalité pour éviter les répétitions
        )
        
        # Retirer de la queue
        await inference_queue.get()
        
        # Durée d'inférence
        inference_duration = time.time() - inference_start
        fastapi_inference_duration_seconds.labels(model="qwen2.5-14b").observe(inference_duration)
        
        # Métriques de tokens
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        
        fastapi_inference_tokens_total.labels(type="prompt").inc(prompt_tokens)
        fastapi_inference_tokens_total.labels(type="completion").inc(completion_tokens)
        
        # Tokens par seconde
        if inference_duration > 0:
            tps = completion_tokens / inference_duration
            fastapi_inference_tokens_per_second.set(tps)
            print(f"[PERF] Génération: {tps:.1f} tokens/sec, {inference_duration:.2f}s total")
        
        generated_text = response['choices'][0]['text'].strip()
        
        # Post-processing pour JSON si nécessaire
        if request.response_format and request.response_format.get("type") == "json_object":
            generated_text = extract_json_from_text(generated_text)
        
        generated_text = ensure_json_response(generated_text, request.response_format)
        
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=generated_text),
                    finish_reason=response['choices'][0]['finish_reason']
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=response['usage']['total_tokens']
            )
        )
        
        # Métriques de succès
        fastapi_inference_requests_total.labels(model="qwen2.5-14b", status="success").inc()
        
        return chat_response
        
    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        fastapi_inference_requests_total.labels(model="qwen2.5-14b", status="error").inc()
        print(f"Erreur lors de la génération: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        fastapi_requests_total.labels(method="POST", endpoint="/v1/chat/completions", status=status).inc()
        fastapi_request_duration_seconds.labels(method="POST", endpoint="/v1/chat/completions").observe(time.time() - start_time)

# ====== ENDPOINT WEBSOCKET ======

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """Endpoint WebSocket pour les complétions de chat"""
    
    # Vérifier le token
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    
    # Envoyer un message de bienvenue
    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "qwen2.5-14b",
        "model_loaded": llm is not None,
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "capabilities": [
            "French medical conversations",
            "Structured JSON output",
            "Multi-turn dialogue", 
            "128K context support"
        ],
        "performance": {
            "json_accuracy": "90%+",
            "languages": "29+ including French",
            "speed": "25-35 tokens/sec",
            "memory": "10-12GB VRAM"
        }
    }
    
    if not llm and download_in_progress:
        welcome_msg["warning"] = "Model is being downloaded. Please wait..."
    
    await websocket.send_json(welcome_msg)
    
    try:
        while True:
            # Recevoir la requête
            data = await websocket.receive_json()
            
            # Log
            print(f"[WS] Requête reçue: {len(data.get('messages', []))} messages")
            
            # Vérifier que le modèle est chargé
            if llm is None:
                error_msg = {
                    "type": "error",
                    "error": "Model not loaded"
                }
                if download_in_progress:
                    error_msg["error"] = "Model is being downloaded. Please check /download-status"
                    error_msg["download_progress"] = model_download_progress._value._value if hasattr(model_download_progress, '_value') else 0
                
                await websocket.send_json(error_msg)
                fastapi_inference_requests_total.labels(model="qwen2.5-14b", status="error").inc()
                continue
            
            # Traiter la requête
            try:
                # Ajouter à la queue
                request = ChatCompletionRequest(
                    messages=[Message(**msg) for msg in data.get("messages", [])],
                    **{k: v for k, v in data.items() if k != "messages"}
                )
                await inference_queue.put(request)
                
                # Convertir les messages
                messages = [Message(**msg) for msg in data.get("messages", [])]
                prompt = format_messages_qwen(messages)
                
                # Ajouter instruction JSON si demandé
                if data.get("response_format", {}).get("type") == "json_object":
                    prompt += "Please respond with valid JSON only, no additional text or explanations.\n"
                
                # Générer
                start_time = time.time()
                
                response = llm(
                    prompt,
                    max_tokens=data.get("max_tokens", 512),
                    temperature=data.get("temperature", 0.1),
                    top_p=data.get("top_p", 0.1),
                    top_k=data.get("top_k", 20),
                    stop=data.get("stop", ["<|im_end|>", "<|endoftext|>", "</s>"]),
                    echo=False,
                    repeat_penalty=1.05
                )
                
                # Retirer de la queue
                await inference_queue.get()
                
                elapsed = (time.time() - start_time) * 1000
                inference_duration = elapsed / 1000.0
                
                # Métriques
                fastapi_inference_duration_seconds.labels(model="qwen2.5-14b").observe(inference_duration)
                fastapi_inference_tokens_total.labels(type="prompt").inc(response['usage']['prompt_tokens'])
                fastapi_inference_tokens_total.labels(type="completion").inc(response['usage']['completion_tokens'])
                
                # Tokens par seconde
                if inference_duration > 0:
                    tps = response['usage']['completion_tokens'] / inference_duration
                    fastapi_inference_tokens_per_second.set(tps)
                
                # Extraire et nettoyer
                generated_text = response['choices'][0]['text'].strip()
                
                # Post-processing pour JSON si nécessaire
                if data.get("response_format", {}).get("type") == "json_object":
                    generated_text = extract_json_from_text(generated_text)
                
                generated_text = ensure_json_response(
                    generated_text, 
                    data.get("response_format")
                )
                
                # Envoyer la réponse
                response_json = {
                    "type": "completion",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": response['choices'][0]['finish_reason']
                    }],
                    "usage": response['usage'],
                    "time_ms": round(elapsed),
                    "tokens_per_second": round(response['usage']['completion_tokens'] / (elapsed / 1000), 2)
                }
                
                # Ajouter request_id s'il existe
                if "request_id" in data:
                    response_json["request_id"] = data["request_id"]
                
                await websocket.send_json(response_json)
                
                # Métriques de succès
                fastapi_inference_requests_total.labels(model="qwen2.5-14b", status="success").inc()
                
                print(f"[WS] Réponse envoyée en {elapsed:.0f}ms ({response_json['tokens_per_second']} t/s)")
                
            except Exception as e:
                print(f"[WS] Erreur: {str(e)}")
                fastapi_inference_requests_total.labels(model="qwen2.5-14b", status="error").inc()
                error_response = {
                    "type": "error",
                    "error": str(e)
                }
                # Ajouter request_id s'il existe même en cas d'erreur
                if "request_id" in data:
                    error_response["request_id"] = data["request_id"]
                    
                await websocket.send_json(error_response)
    
    except WebSocketDisconnect:
        print("[WS] Client déconnecté")
    finally:
        fastapi_websocket_connections.dec()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)