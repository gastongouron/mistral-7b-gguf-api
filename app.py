#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Mixtral-8x7B GGUF avec llama-cpp-python
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

### MODIFIÉ ### Configuration pour Mixtral-8x7B
MODEL_PATH = "/app/models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
API_TOKEN = os.getenv("API_TOKEN", "supersecret")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===== MÉTRIQUES PROMETHEUS =====

# Informations système ### MODIFIÉ ###
system_info = Info('fastapi_system', 'System information')
system_info.info({
    'model': 'mixtral-8x7b',
    'instance': socket.gethostname(),
    'pod_id': os.getenv('RUNPOD_POD_ID', 'local'),
    'version': '3.0.0' # Version applicative
})

# ... (le reste des métriques reste identique)
gpu_utilization_percent = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
gpu_memory_used_bytes = Gauge('gpu_memory_used_bytes', 'GPU memory used in bytes')
gpu_memory_total_bytes = Gauge('gpu_memory_total_bytes', 'GPU memory total in bytes')
gpu_temperature_celsius = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius')
gpu_power_watts = Gauge('gpu_power_watts', 'GPU power usage in watts')
cpu_usage_percent = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_used_bytes = Gauge('memory_used_bytes', 'System memory used in bytes')
memory_total_bytes = Gauge('memory_total_bytes', 'System memory total in bytes')
disk_usage_percent = Gauge('disk_usage_percent', 'Disk usage percentage')
fastapi_requests_total = Counter('fastapi_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
fastapi_request_duration_seconds = Histogram('fastapi_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])
fastapi_websocket_connections = Gauge('fastapi_websocket_connections', 'Number of active WebSocket connections')
fastapi_inference_requests_total = Counter('fastapi_inference_requests_total', 'Total number of inference requests', ['model', 'status'])
fastapi_inference_duration_seconds = Histogram('fastapi_inference_duration_seconds', 'Inference duration in seconds', ['model'], buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0])
fastapi_inference_queue_size = Gauge('fastapi_inference_queue_size', 'Current inference queue size')
fastapi_inference_tokens_total = Counter('fastapi_inference_tokens_total', 'Total tokens processed', ['type'])  # prompt, completion
fastapi_inference_tokens_per_second = Gauge('fastapi_inference_tokens_per_second', 'Tokens generated per second')
model_loaded = Gauge('model_loaded', 'Whether the model is loaded (1) or not (0)')
model_loading_duration_seconds = Gauge('model_loading_duration_seconds', 'Time taken to load the model')
model_download_progress = Gauge('model_download_progress', 'Model download progress in percentage')
inference_queue = asyncio.Queue(maxsize=1000)
download_in_progress = False
download_complete = False

# ... (les fonctions de métriques update_gpu_metrics, update_system_metrics, metrics_update_task restent identiques)
def update_gpu_metrics():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization_percent.set(util.gpu)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used_bytes.set(mem_info.used)
        gpu_memory_total_bytes.set(mem_info.total)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_temperature_celsius.set(temp)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            gpu_power_watts.set(power)
        except:
            pass
    except Exception as e:
        logging.debug(f"Impossible de collecter les métriques GPU: {e}")

def update_system_metrics():
    try:
        cpu_usage_percent.set(psutil.cpu_percent(interval=0.1))
        mem = psutil.virtual_memory()
        memory_used_bytes.set(mem.used)
        memory_total_bytes.set(mem.total)
        disk = psutil.disk_usage('/')
        disk_usage_percent.set(disk.percent)
    except Exception as e:
        logging.debug(f"Impossible de collecter les métriques système: {e}")

async def metrics_update_task():
    while True:
        update_gpu_metrics()
        update_system_metrics()
        fastapi_inference_queue_size.set(inference_queue.qsize())
        await asyncio.sleep(5)

# ===== CODE PRINCIPAL =====

# Sécurité
security = HTTPBearer()

# Modèles Pydantic ### MODIFIÉ ###
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mixtral-8x7b"
    messages: List[Message]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = 0.9
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
        ### MODIFIÉ ### Vérification de la taille pour Mixtral (>30GB)
        if file_size > 30_000_000_000:
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
    ### MODIFIÉ ### Messages pour Mixtral
    print(f"Téléchargement du modèle Mixtral-8x7B... (~32.9 GB)")
    print(f"URL: {MODEL_URL}")
    print("Cela peut prendre 20-40 minutes selon votre connexion...")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        model_download_progress.set(percent)
        sys.stdout.write(f'\rTéléchargement: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) ')
        sys.stdout.flush()
        if int(percent) % 10 == 0 and int(percent) != int((block_num - 1) * block_size * 100 / total_size):
            logging.info(f"Téléchargement du modèle: {percent:.0f}%")
    
    try:
        start_time = time.time()
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=download_progress)
        print("\n✅ Téléchargement terminé!")
        
        download_time = time.time() - start_time
        print(f"Temps de téléchargement: {download_time/60:.1f} minutes")
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"Taille du fichier: {file_size / (1024**3):.1f} GB")
        
        ### MODIFIÉ ### Vérification de la taille pour Mixtral
        if file_size < 30_000_000_000:
            raise Exception(f"Fichier trop petit: {file_size} bytes")
        
        download_complete = True
        model_download_progress.set(100)
        
    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement: {e}")
        download_in_progress = False
        model_download_progress.set(0)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise
    finally:
        download_in_progress = False

### MODIFIÉ ### Fonction de chargement pour Mixtral-8x7B
def load_model():
    """Charger le modèle GGUF avec configuration optimale pour Mixtral-8x7B"""
    global llm
    
    download_model_if_needed()
    
    print(f"Chargement du modèle Mixtral-8x7B depuis {MODEL_PATH}...")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        print(f"VRAM disponible: {vram_gb:.1f} GB")
        
        # Adapter les couches GPU pour Mixtral (plus gourmand)
        if vram_gb >= 40:
            n_gpu_layers = -1  # Tout sur GPU
            print("Configuration: Modèle entièrement sur GPU (recommandé)")
        elif vram_gb >= 24:
            n_gpu_layers = 28  # Bon compromis pour 24GB de VRAM (ex: RTX 4090)
            print("Configuration: 28 couches sur GPU")
        else:
            n_gpu_layers = 16
            print(f"⚠️ VRAM limitée ({vram_gb:.1f}GB), performance réduite (16 couches sur GPU)")
    except Exception as e:
        n_gpu_layers = -1
        print(f"Impossible de détecter la VRAM ({e}), tentative de chargement complet sur GPU")
    
    # Configuration pour Mixtral
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,          # Mixtral supporte 32k de contexte
        n_threads=12,         # Adapter au nombre de coeurs CPU disponibles
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        use_mmap=True,
        use_mlock=False,      # Mieux pour les gros modèles
        verbose=True,
        seed=42
    )
    
    print("Modèle Mixtral-8x7B chargé avec succès!")
    print(f"Configuration: {n_gpu_layers} couches GPU, contexte 32K tokens")

### MODIFIÉ ### Nouvelle fonction de formatage pour Mistral
def format_messages_mistral(messages: List[Message]) -> str:
    """Formater les messages pour Mistral (format [INST])"""
    prompt_parts = ["<s>"]
    has_system_prompt = False
    
    # Gérer le system prompt (Mistral le préfère au début, avant le premier [INST])
    if messages and messages[0].role == "system":
        prompt_parts.append(f"{messages[0].content}\n\n")
        messages = messages[1:]
        has_system_prompt = True

    for i, message in enumerate(messages):
        if message.role == "user":
            # Si c'est le premier message utilisateur, le format est légèrement différent
            if i == 0 and not has_system_prompt:
                 prompt_parts.append(f"[INST] {message.content} [/INST]")
            else:
                prompt_parts.append(f"<s>[INST] {message.content} [/INST]")
        elif message.role == "assistant":
            # La réponse de l'assistant suit directement le [/INST]
            prompt_parts.append(f" {message.content}</s>")
            
    # S'assurer que le prompt se termine prêt pour la réponse de l'assistant
    if not prompt_parts[-1].strip().endswith("</s>"):
         prompt_parts.append(" ") # On attend la réponse de l'assistant

    return "".join(prompt_parts)

# ... (les fonctions extract_json_from_text, clean_and_parse_json, ensure_json_response restent identiques)
def extract_json_from_text(text: str) -> str:
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    json_prefix_match = re.search(r'(?:JSON|json|Json):\s*({.*})', text, re.DOTALL)
    if json_prefix_match:
        return json_prefix_match.group(1)
    return text

def clean_and_parse_json(text: str) -> Optional[Dict]:
    text = extract_json_from_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def ensure_json_response(text: str, request_format: Optional[Dict] = None) -> str:
    if request_format and request_format.get("type") == "json_object":
        parsed = clean_and_parse_json(text)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "response": text,
                "error": "Could not parse as valid JSON."
            }, ensure_ascii=False)
    return text


# ===== LIFESPAN POUR GÉRER LE CYCLE DE VIE =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    print("=== Démarrage de l'application ===")
    metrics_task = asyncio.create_task(metrics_update_task())
    try:
        model_start = time.time()
        load_model()
        model_loading_duration_seconds.set(time.time() - model_start)
        model_loaded.set(1)
        print("=== Modèle chargé, API prête ===")
    except Exception as e:
        print(f"Erreur fatale lors du chargement du modèle: {e}")
        model_loaded.set(0)
    
    yield
    
    print("=== Arrêt de l'application ===")
    model_loaded.set(0)
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass

# Initialisation de l'application avec lifespan ### MODIFIÉ ###
app = FastAPI(
    title="Mixtral-8x7B GGUF API",
    version="3.0.0",
    description="API FastAPI pour Mixtral-8x7B optimisée pour conversations médicales françaises avec JSON structuré",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (L'endpoint /metrics reste identique)
@app.get("/metrics")
async def metrics():
    update_gpu_metrics()
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    ### MODIFIÉ ###
    return {
        "message": "Mixtral-8x7B GGUF API",
        "status": "running" if llm is not None else "loading",
        "model": "Mixtral-8x7B-Instruct-v0.1.Q5_K_M.gguf",
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
        }
    }

# ... (Les endpoints /health, /download-status restent fonctionnellement identiques)
@app.get("/health")
async def health_check():
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
    status = {
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "model_exists": os.path.exists(MODEL_PATH),
        "download_progress_percent": model_download_progress._value._value if hasattr(model_download_progress, '_value') else 0
    }
    if os.path.exists(MODEL_PATH):
        status["current_size_gb"] = os.path.getsize(MODEL_PATH) / (1024**3)
        status["expected_size_gb"] = 32.9
    return status

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    start_time = time.time()
    try:
        ### MODIFIÉ ###
        result = {
            "object": "list",
            "data": [
                {
                    "id": "mixtral-8x7b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Mistral AI",
                    "permission": [],
                    "root": "mixtral-8x7b",
                    "parent": None,
                    "ready": llm is not None
                }
            ]
        }
        fastapi_requests_total.labels(method="GET", endpoint="/v1/models", status="success").inc()
        return result
    except Exception as e:
        fastapi_requests_total.labels(method="GET", endpoint="/v1/models", status="error").inc()
        raise
    finally:
        fastapi_request_duration_seconds.labels(method="GET", endpoint="/v1/models").observe(time.time() - start_time)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_token)])
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint compatible OpenAI optimisé pour Mixtral et outputs structurés"""
    start_time = time.time()
    status = "success"
    
    if llm is None:
        if download_in_progress:
            raise HTTPException(status_code=503, detail="Model is being downloaded. Please check /download-status for progress.")
        else:
            raise HTTPException(status_code=503, detail="Model not loaded. It will be downloaded on first access.")
    
    try:
        await inference_queue.put(request)
        
        ### MODIFIÉ ### Utilisation du format Mistral
        prompt = format_messages_mistral(request.messages)
        
        # Le format JSON est mieux géré par une instruction claire dans le prompt pour Mistral
        # que par un paramètre `grammar`. On peut l'ajouter au dernier message utilisateur.
        
        inference_start = time.time()
        
        ### MODIFIÉ ### Paramètres optimisés pour Mixtral
        response = llm(
            prompt,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.9, # Mistral fonctionne bien avec un top_p plus élevé
            top_k=40,
            stop=request.stop or ["</s>", "[INST]", "[/INST]"], # Tokens d'arrêt pour Mistral
            echo=False,
            repeat_penalty=1.1
        )
        
        await inference_queue.get()
        
        inference_duration = time.time() - inference_start
        fastapi_inference_duration_seconds.labels(model="mixtral-8x7b").observe(inference_duration)
        
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        
        fastapi_inference_tokens_total.labels(type="prompt").inc(prompt_tokens)
        fastapi_inference_tokens_total.labels(type="completion").inc(completion_tokens)
        
        if inference_duration > 0:
            tps = completion_tokens / inference_duration
            fastapi_inference_tokens_per_second.set(tps)
            print(f"[PERF] Génération: {tps:.1f} tokens/sec, {inference_duration:.2f}s total")
        
        generated_text = response['choices'][0]['text'].strip()
        generated_text = ensure_json_response(generated_text, request.response_format)
        
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(index=0, message=Message(role="assistant", content=generated_text), finish_reason=response['choices'][0]['finish_reason'])],
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=response['usage']['total_tokens'])
        )
        
        fastapi_inference_requests_total.labels(model="mixtral-8x7b", status="success").inc()
        return chat_response
        
    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        fastapi_inference_requests_total.labels(model="mixtral-8x7b", status="error").inc()
        logging.error(f"Erreur lors de la génération: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        fastapi_requests_total.labels(method="POST", endpoint="/v1/chat/completions", status=status).inc()
        fastapi_request_duration_seconds.labels(method="POST", endpoint="/v1/chat/completions").observe(time.time() - start_time)

# ====== ENDPOINT WEBSOCKET ======
# ... Le WebSocket reste identique en structure mais les appels internes sont modifiés.

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """Endpoint WebSocket pour les complétions de chat avec Mixtral"""
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    
    ### MODIFIÉ ### Message de bienvenue pour Mixtral
    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "mixtral-8x7b",
        "model_loaded": llm is not None,
        "capabilities": ["French medical conversations", "Structured JSON output", "32K context"]
    }
    await websocket.send_json(welcome_msg)
    
    try:
        while True:
            data = await websocket.receive_json()
            if llm is None:
                await websocket.send_json({"type": "error", "error": "Model not loaded"})
                fastapi_inference_requests_total.labels(model="mixtral-8x7b", status="error").inc()
                continue
            
            try:
                request = ChatCompletionRequest(messages=[Message(**msg) for msg in data.get("messages", [])], **{k: v for k, v in data.items() if k != "messages"})
                await inference_queue.put(request)
                
                ### MODIFIÉ ###
                prompt = format_messages_mistral(request.messages)
                
                start_time = time.time()
                response = llm(
                    prompt,
                    max_tokens=data.get("max_tokens", 4096),
                    temperature=data.get("temperature", 0.1),
                    top_p=data.get("top_p", 0.9),
                    top_k=data.get("top_k", 40),
                    stop=data.get("stop", ["</s>", "[INST]", "[/INST]"]),
                    echo=False,
                    repeat_penalty=1.1
                )
                await inference_queue.get()
                elapsed = (time.time() - start_time) * 1000
                inference_duration = elapsed / 1000.0

                fastapi_inference_duration_seconds.labels(model="mixtral-8x7b").observe(inference_duration)
                fastapi_inference_tokens_total.labels(type="prompt").inc(response['usage']['prompt_tokens'])
                fastapi_inference_tokens_total.labels(type="completion").inc(response['usage']['completion_tokens'])

                if inference_duration > 0:
                    tps = response['usage']['completion_tokens'] / inference_duration
                    fastapi_inference_tokens_per_second.set(tps)

                generated_text = response['choices'][0]['text'].strip()
                generated_text = ensure_json_response(generated_text, data.get("response_format"))

                response_json = {
                    "type": "completion",
                    "choices": [{"message": {"role": "assistant", "content": generated_text}, "finish_reason": response['choices'][0]['finish_reason']}],
                    "usage": response['usage'],
                    "time_ms": round(elapsed),
                    "tokens_per_second": round(response['usage']['completion_tokens'] / (elapsed / 1000), 2)
                }
                if "request_id" in data:
                    response_json["request_id"] = data["request_id"]
                
                await websocket.send_json(response_json)
                fastapi_inference_requests_total.labels(model="mixtral-8x7b", status="success").inc()
                
            except Exception as e:
                logging.error(f"[WS] Erreur: {str(e)}", exc_info=True)
                fastapi_inference_requests_total.labels(model="mixtral-8x7b", status="error").inc()
                error_response = {"type": "error", "error": str(e)}
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