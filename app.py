#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Mixtral-8x7B GGUF avec llama-cpp-python
Optimisé pour conversations médicales françaises avec extraction JSON AMÉLIORÉE
Version avec parsing JSON intelligent et nettoyage automatique
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
from typing import List, Optional, Dict, Any, Tuple
from llama_cpp import Llama

# Import des métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

# Configuration pour Mixtral-8x7B
MODEL_PATH = "/workspace/models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
API_TOKEN = os.getenv("API_TOKEN", "supersecret")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===== MÉTRIQUES PROMETHEUS =====
system_info = Info('fastapi_system', 'System information')
system_info.info({
    'model': 'mixtral-8x7b',
    'instance': socket.gethostname(),
    'pod_id': os.getenv('RUNPOD_POD_ID', 'local'),
    'version': '4.0.0'  # Version avec JSON amélioré
})

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
fastapi_inference_tokens_total = Counter('fastapi_inference_tokens_total', 'Total tokens processed', ['type'])
fastapi_inference_tokens_per_second = Gauge('fastapi_inference_tokens_per_second', 'Tokens generated per second')
model_loaded = Gauge('model_loaded', 'Whether the model is loaded (1) or not (0)')
model_loading_duration_seconds = Gauge('model_loading_duration_seconds', 'Time taken to load the model')
model_download_progress = Gauge('model_download_progress', 'Model download progress in percentage')
json_parse_success_total = Counter('json_parse_success_total', 'Number of successful JSON parses')
json_parse_failure_total = Counter('json_parse_failure_total', 'Number of failed JSON parses')

inference_queue = asyncio.Queue(maxsize=1000)
download_in_progress = False
download_complete = False

# ===== FONCTIONS DE MÉTRIQUES =====
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

# ===== PARSING JSON AMÉLIORÉ =====

def clean_escaped_json(text: str) -> str:
    """Nettoie les caractères d'échappement dans le JSON"""
    # Remplace les underscores échappés
    text = text.replace(r'\_', '_')
    # Remplace les doubles backslashes
    text = text.replace('\\\\', '\\')
    # Nettoie les espaces et retours à la ligne excessifs
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

def extract_json_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrait le JSON d'un texte et retourne (json_str, remaining_text)
    Version améliorée qui gère plusieurs formats
    """
    text = text.strip()
    
    # Nettoie d'abord les échappements
    text = clean_escaped_json(text)
    
    # Cas 1: Le texte commence et finit par des accolades
    if text.startswith('{') and text.endswith('}'):
        return text, None
    
    # Cas 2: JSON dans des balises code
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        remaining = text[:json_match.start()] + text[json_match.end():]
        return json_str, remaining.strip() if remaining.strip() else None
    
    # Cas 3: JSON avec un préfixe (JSON:, json:, etc.)
    json_prefix_match = re.search(r'(?:JSON|json|Json):\s*({.*?})', text, re.DOTALL)
    if json_prefix_match:
        json_str = json_prefix_match.group(1)
        remaining = text[:json_prefix_match.start()] + text[json_prefix_match.end():]
        return json_str, remaining.strip() if remaining.strip() else None
    
    # Cas 4: Cherche la première accolade ouvrante et la dernière fermante
    start = text.find('{')
    if start != -1:
        # Compte les accolades pour trouver la fin correcte
        count = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    end = i
                    break
        
        if end != -1:
            json_str = text[start:end+1]
            remaining = text[:start] + text[end+1:]
            return json_str, remaining.strip() if remaining.strip() else None
    
    # Cas 5: Pas de JSON trouvé
    return None, text

def smart_json_parse(text: str) -> Dict[str, Any]:
    """
    Parse intelligent du JSON avec plusieurs stratégies de récupération
    """
    original_text = text
    
    # Étape 1: Extraction du JSON
    json_str, remaining_text = extract_json_from_text(text)
    
    if not json_str:
        logging.warning("Aucun JSON trouvé dans la réponse")
        json_parse_failure_total.inc()
        return {
            "error": "No JSON found in response",
            "original_response": original_text
        }
    
    # Étape 2: Tentative de parsing direct
    try:
        result = json.loads(json_str)
        json_parse_success_total.inc()
        
        # Si il y a du texte supplémentaire, on peut l'ajouter comme metadata
        if remaining_text:
            result["_metadata"] = {
                "additional_text": remaining_text,
                "json_extracted": True
            }
        
        return result
    except json.JSONDecodeError as e:
        logging.warning(f"Première tentative de parsing échouée: {e}")
    
    # Étape 3: Nettoyage et nouvelle tentative
    # Retire les commentaires style //
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    # Retire les virgules trailing
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    try:
        result = json.loads(json_str)
        json_parse_success_total.inc()
        
        if remaining_text:
            result["_metadata"] = {
                "additional_text": remaining_text,
                "json_extracted": True,
                "required_cleanup": True
            }
        
        return result
    except json.JSONDecodeError as e:
        logging.warning(f"Deuxième tentative échouée: {e}")
    
    # Étape 4: Tentative de réparation avec regex
    # Essaie de corriger les clés sans guillemets
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    # Corrige les valeurs true/false/null
    json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
    json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
    json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
    
    try:
        result = json.loads(json_str)
        json_parse_success_total.inc()
        
        if remaining_text:
            result["_metadata"] = {
                "additional_text": remaining_text,
                "json_extracted": True,
                "heavy_cleanup": True
            }
        
        return result
    except json.JSONDecodeError as e:
        logging.error(f"Toutes les tentatives de parsing ont échoué: {e}")
        json_parse_failure_total.inc()
        
        # Retourne une structure d'erreur
        return {
            "error": "JSON parsing failed after all attempts",
            "original_response": original_text,
            "attempted_json": json_str,
            "parse_error": str(e)
        }

def ensure_json_response(text: str, request_format: Optional[Dict] = None) -> str:
    """
    S'assure que la réponse est un JSON valide
    """
    if request_format and request_format.get("type") == "json_object":
        parsed = smart_json_parse(text)
        
        # Nettoie les metadata si elles existent
        if "_metadata" in parsed and not os.getenv("DEBUG_MODE"):
            del parsed["_metadata"]
        
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    
    return text

# ===== MODÈLES PYDANTIC =====
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
    json_schema: Optional[Dict[str, Any]] = None  # Pour forcer un schéma spécifique

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

# ===== AUTHENTIFICATION =====
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifier le token Bearer"""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ===== GESTION DU MODÈLE =====
def download_model_if_needed():
    """Télécharger le modèle au premier démarrage si nécessaire"""
    global download_in_progress, download_complete
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
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
        
        if vram_gb >= 40:
            n_gpu_layers = -1
            print("Configuration: Modèle entièrement sur GPU (recommandé)")
        elif vram_gb >= 24:
            n_gpu_layers = 28
            print("Configuration: 28 couches sur GPU")
        else:
            n_gpu_layers = 16
            print(f"⚠️ VRAM limitée ({vram_gb:.1f}GB), performance réduite (16 couches sur GPU)")
    except Exception as e:
        n_gpu_layers = -1
        print(f"Impossible de détecter la VRAM ({e}), tentative de chargement complet sur GPU")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=12,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        use_mmap=True,
        use_mlock=False,
        verbose=True,
        seed=42
    )
    
    print("Modèle Mixtral-8x7B chargé avec succès!")
    print(f"Configuration: {n_gpu_layers} couches GPU, contexte 32K tokens")

def format_messages_mistral(messages: List[Message]) -> str:
    """Formater les messages pour Mistral avec support JSON amélioré"""
    prompt_parts = ["<s>"]
    has_system_prompt = False
    
    # Ajoute automatiquement une instruction pour le JSON si nécessaire
    json_instruction = """You are a helpful assistant that ALWAYS responds with valid JSON.
Your response must be ONLY the JSON object, with no additional text before or after.
Do not include any explanations, comments, or markdown formatting.
Just return the raw JSON object."""
    
    # Gérer le system prompt
    if messages and messages[0].role == "system":
        # Combine le system prompt existant avec l'instruction JSON
        system_content = messages[0].content
        if "json" in system_content.lower() or "JSON" in system_content:
            prompt_parts.append(f"{system_content}\n\n")
        else:
            prompt_parts.append(f"{system_content}\n\n{json_instruction}\n\n")
        messages = messages[1:]
        has_system_prompt = True
    else:
        # Ajoute l'instruction JSON comme system prompt
        prompt_parts.append(f"{json_instruction}\n\n")
        has_system_prompt = True

    for i, message in enumerate(messages):
        if message.role == "user":
            if i == 0 and not has_system_prompt:
                prompt_parts.append(f"[INST] {message.content} [/INST]")
            else:
                prompt_parts.append(f"<s>[INST] {message.content} [/INST]")
        elif message.role == "assistant":
            prompt_parts.append(f" {message.content}</s>")
            
    if not prompt_parts[-1].strip().endswith("</s>"):
        prompt_parts.append(" ")

    return "".join(prompt_parts)

# ===== LIFESPAN =====
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

# ===== APPLICATION FASTAPI =====
app = FastAPI(
    title="Mixtral-8x7B GGUF API",
    version="4.0.0",
    description="API FastAPI pour Mixtral-8x7B avec parsing JSON intelligent",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS =====

@app.get("/metrics")
async def metrics():
    update_gpu_metrics()
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    return {
        "message": "Mixtral-8x7B GGUF API with Smart JSON Parsing",
        "status": "running" if llm is not None else "loading",
        "model": "Mixtral-8x7B-Instruct-v0.1.Q5_K_M.gguf",
        "model_loaded": llm is not None,
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "features": [
            "Intelligent JSON parsing",
            "Automatic escape character cleanup",
            "Multiple JSON extraction strategies",
            "Structured error responses"
        ],
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/ws": "WebSocket - Chat endpoint (requires token in query)",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics",
            "/download-status": "GET - Model download status"
        }
    }

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy" if llm is not None else "loading",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "json_parse_stats": {
            "success": json_parse_success_total._value._value if hasattr(json_parse_success_total, '_value') else 0,
            "failure": json_parse_failure_total._value._value if hasattr(json_parse_failure_total, '_value') else 0
        }
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
    """Endpoint compatible OpenAI avec parsing JSON intelligent"""
    start_time = time.time()
    status = "success"
    
    if llm is None:
        if download_in_progress:
            raise HTTPException(status_code=503, detail="Model is being downloaded. Please check /download-status for progress.")
        else:
            raise HTTPException(status_code=503, detail="Model not loaded. It will be downloaded on first access.")
    
    try:
        await inference_queue.put(request)
        
        # Formatte le prompt avec support JSON amélioré
        prompt = format_messages_mistral(request.messages)
        
        # Si un schéma JSON est fourni, l'ajoute au prompt
        if request.json_schema:
            schema_instruction = f"\n\nYour response must conform to this JSON schema:\n{json.dumps(request.json_schema, indent=2)}"
            prompt = prompt.rstrip() + schema_instruction + "\n\nResponse (JSON only):"
        
        inference_start = time.time()
        
        response = llm(
            prompt,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.9,
            top_k=40,
            stop=request.stop or ["</s>", "[INST]", "[/INST]"],
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
            logging.info(f"[PERF] Génération: {tps:.1f} tokens/sec, {inference_duration:.2f}s total")
        
        # Utilise le parsing JSON intelligent
        generated_text = response['choices'][0]['text'].strip()
        
        # Si on attend du JSON, applique le parsing intelligent
        if request.response_format and request.response_format.get("type") == "json_object":
            generated_text = ensure_json_response(generated_text, request.response_format)
            
            # Vérifie si le parsing a réussi
            try:
                parsed = json.loads(generated_text)
                if "error" in parsed and "original_response" in parsed:
                    logging.warning(f"JSON parsing failed, returning error structure: {parsed['error']}")
            except:
                pass
        
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=generated_text),
                finish_reason=response['choices'][0]['finish_reason']
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=response['usage']['total_tokens']
            )
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """Endpoint WebSocket avec parsing JSON intelligent"""
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    
    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "mixtral-8x7b",
        "model_loaded": llm is not None,
        "capabilities": [
            "French medical conversations",
            "Structured JSON output",
            "32K context",
            "Intelligent JSON parsing"
        ]
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
                request = ChatCompletionRequest(
                    messages=[Message(**msg) for msg in data.get("messages", [])],
                    **{k: v for k, v in data.items() if k != "messages"}
                )
                await inference_queue.put(request)
                
                prompt = format_messages_mistral(request.messages)
                
                # Ajoute le schéma JSON si fourni
                if data.get("json_schema"):
                    schema_instruction = f"\n\nYour response must conform to this JSON schema:\n{json.dumps(data['json_schema'], indent=2)}"
                    prompt = prompt.rstrip() + schema_instruction + "\n\nResponse (JSON only):"
                
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
                
                # Applique le parsing JSON intelligent si nécessaire
                if data.get("response_format") and data["response_format"].get("type") == "json_object":
                    generated_text = ensure_json_response(generated_text, data.get("response_format"))

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
        logging.info("[WS] Client déconnecté")
    finally:
        fastapi_websocket_connections.dec()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)