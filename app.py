"""
API FastAPI pour servir le modèle Qwen2.5-32B GGUF avec llama-cpp-python
Optimisé pour Q6_K avec support multi-utilisateurs et streaming interruptible
Version 8.0.0 - Production Ready
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
import subprocess
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator, Union
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
import weakref
from datetime import datetime, timedelta
import hashlib

# Import des métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

# Configuration pour Qwen2.5-32B Q6_K (optimisé multi-users)
MODEL_PATH = "/workspace/models/Qwen2.5-32B-Instruct-Q6_K.gguf"
MODEL_URL = "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q6_K.gguf"

# Configuration environnement
API_TOKEN = os.getenv("API_TOKEN", "supersecret")
MAX_CONCURRENT_USERS = int(os.getenv("MAX_CONCURRENT_USERS", "9"))
MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOKENS_PER_REQUEST", "150"))
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== MÉTRIQUES PROMETHEUS =====
system_info = Info('fastapi_system', 'System information')
system_info.info({
    'model': 'qwen2.5-32b-q6k',
    'instance': socket.gethostname(),
    'pod_id': os.getenv('RUNPOD_POD_ID', 'local'),
    'version': '8.0.0',
    'max_concurrent_users': str(MAX_CONCURRENT_USERS)
})

# GPU Metrics
gpu_utilization_percent = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
gpu_memory_used_bytes = Gauge('gpu_memory_used_bytes', 'GPU memory used in bytes')
gpu_memory_total_bytes = Gauge('gpu_memory_total_bytes', 'GPU memory total in bytes')
gpu_temperature_celsius = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius')
gpu_power_watts = Gauge('gpu_power_watts', 'GPU power usage in watts')
gpu_layers_offloaded = Gauge('gpu_layers_offloaded', 'Number of layers offloaded to GPU')

# System Metrics
cpu_usage_percent = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_used_bytes = Gauge('memory_used_bytes', 'System memory used in bytes')
memory_total_bytes = Gauge('memory_total_bytes', 'System memory total in bytes')
disk_usage_percent = Gauge('disk_usage_percent', 'Disk usage percentage')

# Request Metrics
fastapi_requests_total = Counter('fastapi_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
fastapi_request_duration_seconds = Histogram('fastapi_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])
fastapi_websocket_connections = Gauge('fastapi_websocket_connections', 'Number of active WebSocket connections')
fastapi_concurrent_users = Gauge('fastapi_concurrent_users', 'Number of concurrent users')

# Inference Metrics
fastapi_inference_requests_total = Counter('fastapi_inference_requests_total', 'Total number of inference requests', ['model', 'status'])
fastapi_inference_duration_seconds = Histogram('fastapi_inference_duration_seconds', 'Inference duration in seconds', ['model'], 
                                                buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0])
fastapi_inference_queue_size = Gauge('fastapi_inference_queue_size', 'Current inference queue size')
fastapi_inference_tokens_total = Counter('fastapi_inference_tokens_total', 'Total tokens processed', ['type'])
fastapi_inference_tokens_per_second = Gauge('fastapi_inference_tokens_per_second', 'Tokens generated per second')

# Model Metrics
model_loaded = Gauge('model_loaded', 'Whether the model is loaded (1) or not (0)')
model_loading_duration_seconds = Gauge('model_loading_duration_seconds', 'Time taken to load the model')
model_download_progress = Gauge('model_download_progress', 'Model download progress in percentage')

# JSON Parsing Metrics
json_parse_success_total = Counter('json_parse_success_total', 'Number of successful JSON parses')
json_parse_failure_total = Counter('json_parse_failure_total', 'Number of failed JSON parses')

# Stream Metrics
stream_cancellation_total = Counter('stream_cancellation_total', 'Number of stream cancellations')
stream_cancellation_latency_seconds = Histogram('stream_cancellation_latency_seconds', 'Time from cancellation request to actual stop')

# Rate Limiting Metrics
rate_limit_exceeded_total = Counter('rate_limit_exceeded_total', 'Number of rate limit exceeded events', ['user_id'])

# ===== MODÈLES PYDANTIC =====
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-32b"
    messages: List[Message]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 200
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[str, str]] = None
    json_schema: Optional[Dict[str, Any]] = None
    user: Optional[str] = None  # Pour tracking par utilisateur

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

class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]

# ===== RATE LIMITER =====
class RateLimiter:
    """Rate limiter simple par utilisateur"""
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.requests = {}
        self._lock = threading.Lock()
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Vérifie si l'utilisateur a dépassé la limite"""
        if not ENABLE_RATE_LIMITING:
            return True
            
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        with self._lock:
            # Nettoyer les anciennes entrées
            if user_id in self.requests:
                self.requests[user_id] = [
                    req_time for req_time in self.requests[user_id] 
                    if req_time > minute_ago
                ]
            else:
                self.requests[user_id] = []
            
            # Vérifier la limite
            if len(self.requests[user_id]) >= self.max_requests:
                rate_limit_exceeded_total.labels(user_id=user_id).inc()
                return False
            
            # Ajouter la nouvelle requête
            self.requests[user_id].append(now)
            return True

# ===== GESTIONNAIRE DE RESSOURCES =====
class ResourceManager:
    """Gère les ressources et limites du système"""
    def __init__(self):
        self.active_users = set()
        self.user_streams = {}  # user_id -> set of stream_ids
        self._lock = threading.Lock()
    
    def can_accept_user(self, user_id: str) -> bool:
        """Vérifie si on peut accepter un nouvel utilisateur"""
        with self._lock:
            if user_id in self.active_users:
                return True  # Déjà actif
            
            if len(self.active_users) >= MAX_CONCURRENT_USERS:
                return False
            
            self.active_users.add(user_id)
            self.user_streams[user_id] = set()
            fastapi_concurrent_users.set(len(self.active_users))
            return True
    
    def register_stream(self, user_id: str, stream_id: str):
        """Enregistre un stream pour un utilisateur"""
        with self._lock:
            if user_id in self.user_streams:
                self.user_streams[user_id].add(stream_id)
    
    def unregister_stream(self, user_id: str, stream_id: str):
        """Désenregistre un stream"""
        with self._lock:
            if user_id in self.user_streams:
                self.user_streams[user_id].discard(stream_id)
                
                # Si l'utilisateur n'a plus de streams actifs
                if not self.user_streams[user_id]:
                    self.active_users.discard(user_id)
                    del self.user_streams[user_id]
                    fastapi_concurrent_users.set(len(self.active_users))
    
    def get_user_from_stream(self, stream_id: str) -> Optional[str]:
        """Retrouve l'utilisateur d'un stream"""
        with self._lock:
            for user_id, streams in self.user_streams.items():
                if stream_id in streams:
                    return user_id
        return None

# ===== GESTIONNAIRE DE STREAMS OPTIMISÉ =====
class OptimizedStreamManager:
    """Version optimisée du gestionnaire de streams pour multi-users"""
    def __init__(self):
        self.active_streams = {}
        self.executor = ThreadPoolExecutor(max_workers=8)  # Plus de workers
        self._lock = threading.Lock()
        self.resource_manager = ResourceManager()
        
        # Configuration optimisée pour conversation
        self.chunk_size = 15  # Petits chunks pour réactivité
        self.max_tokens_per_chunk = 30
        self.stream_timeout = 10  # Timeout agressif
    
    def register_stream(self, request_id: str, user_id: str) -> threading.Event:
        """Enregistre un nouveau stream avec gestion utilisateur"""
        cancel_event = threading.Event()
        
        with self._lock:
            self.active_streams[request_id] = {
                'cancel_event': cancel_event,
                'start_time': time.time(),
                'tokens_generated': 0,
                'user_id': user_id,
                'chunks_generated': 0
            }
        
        self.resource_manager.register_stream(user_id, request_id)
        return cancel_event
    
    def cancel_stream(self, request_id: str) -> bool:
        """Annule un stream actif avec métriques"""
        with self._lock:
            if request_id in self.active_streams:
                stream_info = self.active_streams[request_id]
                stream_info['cancel_event'].set()
                
                # Métriques
                duration = time.time() - stream_info['start_time']
                stream_cancellation_total.inc()
                stream_cancellation_latency_seconds.observe(duration)
                
                logger.info(f"[STREAM] Cancelled {request_id} after {stream_info['tokens_generated']} tokens")
                return True
        return False
    
    def unregister_stream(self, request_id: str):
        """Retire un stream et libère les ressources"""
        with self._lock:
            if request_id in self.active_streams:
                user_id = self.active_streams[request_id]['user_id']
                del self.active_streams[request_id]
                
        # Libérer les ressources utilisateur
        if user_id:
            self.resource_manager.unregister_stream(user_id, request_id)
    
    async def generate_async(self, llm_model, prompt: str, request_id: str, 
                           user_id: str, **kwargs) -> AsyncGenerator[Dict, None]:
        """Génération optimisée avec chunks et interruption"""
        cancel_event = self.register_stream(request_id, user_id)
        
        # Limiter les tokens pour la conversation
        original_max_tokens = min(kwargs.get('max_tokens', 200), MAX_TOKENS_PER_REQUEST)
        
        loop = asyncio.get_event_loop()
        tokens_generated = 0
        full_text = ""
        
        try:
            # Générer par chunks pour permettre l'interruption
            remaining_tokens = original_max_tokens
            current_prompt = prompt
            
            while remaining_tokens > 0 and not cancel_event.is_set():
                # Taille du chunk actuel
                chunk_tokens = min(self.chunk_size, remaining_tokens)
                kwargs['max_tokens'] = chunk_tokens
                
                # Générer le chunk de manière synchrone
                def generate_chunk():
                    return llm_model(current_prompt, stream=True, **kwargs)
                
                # Exécuter dans le thread pool avec timeout
                try:
                    future = loop.run_in_executor(self.executor, generate_chunk)
                    stream = await asyncio.wait_for(future, timeout=self.stream_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"[STREAM] Timeout for {request_id}")
                    break
                
                # Parcourir les tokens du chunk
                chunk_text = ""
                for output in stream:
                    if cancel_event.is_set():
                        break
                    
                    token = output['choices'][0]['text']
                    chunk_text += token
                    full_text += token
                    tokens_generated += 1
                    
                    with self._lock:
                        if request_id in self.active_streams:
                            self.active_streams[request_id]['tokens_generated'] = tokens_generated
                    
                    yield output
                    
                    # Micro-pause pour permettre l'interruption
                    await asyncio.sleep(0)
                
                # Préparer le prochain chunk
                remaining_tokens -= chunk_tokens
                current_prompt = prompt + full_text
                
                # Vérifier si on doit continuer (pas de fin naturelle)
                if chunk_text.strip().endswith(('.', '!', '?', '"', '\n')):
                    break  # Fin naturelle de phrase
                    
        except Exception as e:
            logger.error(f"[STREAM] Error in {request_id}: {str(e)}")
            raise
        finally:
            self.unregister_stream(request_id)

# Variables globales
llm = None
stream_manager = OptimizedStreamManager()
rate_limiter = RateLimiter()
download_in_progress = False
download_complete = False

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

def get_user_id(request: Request, token: str = Depends(verify_token)) -> str:
    """Extrait l'ID utilisateur de la requête"""
    # Priorité : header X-User-ID > paramètre user > hash du token
    user_id = request.headers.get('X-User-ID')
    if not user_id and hasattr(request, 'json_body') and request.json_body:
        user_id = request.json_body.get('user')
    if not user_id:
        # Générer un ID basé sur le token pour la session
        user_id = hashlib.md5(f"{token}:{request.client.host}".encode()).hexdigest()[:8]
    return user_id

# ===== MÉTRIQUES SYSTÈME =====
def update_gpu_metrics():
    """Met à jour les métriques GPU"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization_percent.set(util.gpu)
        
        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used_bytes.set(mem_info.used)
        gpu_memory_total_bytes.set(mem_info.total)
        
        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_temperature_celsius.set(temp)
        
        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            gpu_power_watts.set(power)
        except:
            pass
            
    except Exception as e:
        logger.debug(f"Could not collect GPU metrics: {e}")

def update_system_metrics():
    """Met à jour les métriques système"""
    try:
        cpu_usage_percent.set(psutil.cpu_percent(interval=0.1))
        mem = psutil.virtual_memory()
        memory_used_bytes.set(mem.used)
        memory_total_bytes.set(mem.total)
        disk = psutil.disk_usage('/')
        disk_usage_percent.set(disk.percent)
    except Exception as e:
        logger.debug(f"Could not collect system metrics: {e}")

async def metrics_update_task():
    """Tâche de mise à jour des métriques"""
    while True:
        update_gpu_metrics()
        update_system_metrics()
        await asyncio.sleep(5)

# ===== FORMATTERS =====
def format_messages_qwen(messages: List[Message]) -> str:
    """Format ChatML pour Qwen2.5"""
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

# ===== PARSING JSON AMÉLIORÉ =====
def clean_escaped_json(text: str) -> str:
    """Nettoie les caractères d'échappement dans le JSON"""
    text = text.replace(r'\_', '_')
    text = text.replace('\\\\', '\\')
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

def extract_json_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extrait le JSON d'un texte et retourne (json_str, remaining_text)"""
    text = text.strip()
    text = clean_escaped_json(text)
    
    # Cas 1: JSON pur
    if text.startswith('{') and text.endswith('}'):
        return text, None
    
    # Cas 2: JSON dans code block
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        remaining = text[:json_match.start()] + text[json_match.end():]
        return json_str, remaining.strip() if remaining.strip() else None
    
    # Cas 3: JSON avec préfixe
    json_prefix_match = re.search(r'(?:JSON|json|Json):\s*({.*?})', text, re.DOTALL)
    if json_prefix_match:
        json_str = json_prefix_match.group(1)
        remaining = text[:json_prefix_match.start()] + text[json_prefix_match.end():]
        return json_str, remaining.strip() if remaining.strip() else None
    
    # Cas 4: Chercher les accolades
    start = text.find('{')
    if start != -1:
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
    
    return None, text

def smart_json_parse(text: str) -> Dict[str, Any]:
    """Parse intelligent du JSON avec récupération d'erreurs"""
    original_text = text
    json_str, remaining_text = extract_json_from_text(text)
    
    if not json_str:
        logger.warning("No JSON found in response")
        json_parse_failure_total.inc()
        return {
            "error": "No JSON found in response",
            "original_response": original_text
        }
    
    # Tentative 1: Parse direct
    try:
        result = json.loads(json_str)
        json_parse_success_total.inc()
        
        if remaining_text:
            result["_metadata"] = {
                "additional_text": remaining_text,
                "json_extracted": True
            }
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"First parse attempt failed: {e}")
    
    # Tentative 2: Nettoyage basique
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
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
        logger.warning(f"Second parse attempt failed: {e}")
    
    # Tentative 3: Réparation agressive
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
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
        logger.error(f"All parse attempts failed: {e}")
        json_parse_failure_total.inc()
        
        return {
            "error": "JSON parsing failed after all attempts",
            "original_response": original_text,
            "attempted_json": json_str,
            "parse_error": str(e)
        }

def ensure_json_response(text: str, request_format: Optional[Dict] = None) -> str:
    """S'assure que la réponse est un JSON valide"""
    if request_format and request_format.get("type") == "json_object":
        parsed = smart_json_parse(text)
        
        if "_metadata" in parsed and not os.getenv("DEBUG_MODE"):
            del parsed["_metadata"]
        
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    
    return text

# ===== GESTION DU MODÈLE =====
def download_with_retry(url: str, dest: str, max_retries: int = 3, delay: int = 5):
    """Télécharge avec retry et gestion des erreurs"""
    import urllib.error
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        print("⚠️  WARNING: No HuggingFace token found!")
        print("   Set HF_TOKEN environment variable to avoid 429 errors")
    
    # Essayer huggingface-cli en premier
    if hf_token and subprocess.run(["which", "huggingface-cli"], capture_output=True).returncode == 0:
        print("🔑 HuggingFace token detected, using huggingface-cli...")
        try:
            subprocess.run(["huggingface-cli", "login", "--token", hf_token], 
                         capture_output=True, check=True)
            
            hf_cmd = [
                "huggingface-cli", "download",
                "bartowski/Qwen2.5-32B-Instruct-GGUF",
                "Qwen2.5-32B-Instruct-Q6_K.gguf",
                "--local-dir", os.path.dirname(dest),
                "--local-dir-use-symlinks", "False"
            ]
            
            print("📥 Downloading with huggingface-cli...")
            result = subprocess.run(hf_cmd, text=True)
            if result.returncode == 0:
                print("✅ Download successful!")
                return True
        except Exception as e:
            print(f"⚠️ huggingface-cli failed: {e}")
    
    # Fallback sur téléchargement direct
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = delay * (2 ** attempt)
                print(f"\n⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            
            print(f"\n📥 Attempt {attempt + 1}/{max_retries}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            if hf_token:
                headers['Authorization'] = f'Bearer {hf_token}'
            
            request = urllib.request.Request(url, headers=headers)
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                model_download_progress.set(percent)
                sys.stdout.write(f'\rDownload: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) ')
                sys.stdout.flush()
            
            urllib.request.urlretrieve(request.full_url, dest, reporthook=download_progress)
            print("\n✅ Download complete!")
            return True
            
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"\n⚠️ Error 429: Rate limit exceeded")
                if not hf_token:
                    print("💡 Tip: Set HF_TOKEN to avoid this error")
                if attempt < max_retries - 1:
                    continue
            else:
                print(f"\n❌ HTTP Error {e.code}: {e.reason}")
                if attempt < max_retries - 1:
                    continue
                raise
    
    return False

def cleanup_old_models():
    """Nettoie les anciens modèles avant téléchargement"""
    models_dir = "/workspace/models"
    
    print(f"\n{'='*60}")
    print("🧹 CLEANING OLD MODELS")
    print(f"{'='*60}")
    
    try:
        import glob
        old_models = glob.glob(os.path.join(models_dir, "*.gguf"))
        
        if not old_models:
            print("✅ No old models found")
            return
        
        # Calculer l'espace utilisé
        total_size = sum(os.path.getsize(f) for f in old_models)
        print(f"📊 Space used by old models: {total_size / (1024**3):.1f} GB")
        
        # Supprimer les anciens modèles sauf Q6_K
        target_model = os.path.basename(MODEL_PATH)
        for model_file in old_models:
            if os.path.basename(model_file) != target_model:
                print(f"🗑️  Deleting: {os.path.basename(model_file)}")
                try:
                    os.remove(model_file)
                    print(f"   ✅ Deleted")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
        
        # Vérifier l'espace libre
        import shutil
        stat = shutil.disk_usage(models_dir)
        free_gb = stat.free / (1024**3)
        print(f"\n💾 Free space after cleanup: {free_gb:.1f} GB")
        
        # Q6_K ~25GB
        required_gb = 30
        if free_gb < required_gb:
            print(f"⚠️  WARNING: Only {free_gb:.1f} GB available")
            print(f"   Q6_K model requires ~25 GB")
        else:
            print(f"✅ Sufficient space for new model")
            
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
    
    print(f"{'='*60}\n")

def download_model_if_needed():
    """Télécharge le modèle si nécessaire"""
    global download_in_progress, download_complete
    
    cleanup_old_models()
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        expected_size = 25_000_000_000  # ~25 GB pour Q6_K
        if file_size > expected_size * 0.95:
            print(f"✅ Model found: {file_size / (1024**3):.1f} GB")
            download_complete = True
            return
        else:
            print(f"⚠️ Incomplete model ({file_size / (1024**3):.1f} GB), resuming download...")
    
    if download_in_progress:
        print("⏳ Download already in progress...")
        return
    
    download_in_progress = True
    
    print(f"\n{'='*60}")
    print(f"📥 DOWNLOADING QWEN2.5-32B Q6_K")
    print(f"{'='*60}")
    print(f"📦 Size: ~25 GB (Q6_K - optimal for multi-users)")
    print(f"📍 Destination: {MODEL_PATH}")
    print(f"⏱️ Estimated time: 15-20 minutes")
    print(f"{'='*60}\n")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    try:
        start_time = time.time()
        
        if not download_with_retry(MODEL_URL, MODEL_PATH):
            raise Exception("Download failed after all attempts")
        
        download_time = time.time() - start_time
        print(f"\n✅ Download completed in {download_time/60:.1f} minutes")
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"📦 File size: {file_size / (1024**3):.1f} GB")
        
        if file_size < 24_000_000_000:
            raise Exception(f"File too small: {file_size} bytes")
        
        download_complete = True
        model_download_progress.set(100)
        
    except Exception as e:
        print(f"\n❌ Download error: {e}")
        download_in_progress = False
        model_download_progress.set(0)
        raise
    finally:
        download_in_progress = False

def load_model():
    """Charge le modèle Q6_K avec configuration optimale"""
    global llm
    
    download_model_if_needed()
    
    print(f"\n{'='*60}")
    print(f"🚀 LOADING QWEN2.5-32B Q6_K")
    print(f"{'='*60}")
    
    try:
        # Détection GPU
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        vram_free_gb = mem_info.free / (1024**3)
        
        print(f"📊 GPU Detection:")
        print(f"   Total VRAM: {vram_gb:.1f} GB")
        print(f"   Free VRAM: {vram_free_gb:.1f} GB")
        
        # FORCER tout sur GPU
        n_gpu_layers = -1
        print(f"✅ Configuration: ALL layers on GPU (forced)")
        
    except Exception as e:
        print(f"⚠️ GPU detection failed: {e}")
        print("✅ Forcing GPU usage anyway...")
        n_gpu_layers = -1
    
    print(f"\n📋 Model Configuration:")
    print(f"   Model: Qwen2.5-32B Q6_K")
    print(f"   Context: 8192 tokens (optimized for multi-users)")
    print(f"   GPU Layers: ALL")
    print(f"   Batch Size: 512")
    print(f"   Max Users: {MAX_CONCURRENT_USERS}")
    
    start_load = time.time()
    
    # Configuration optimisée pour Q6_K multi-users
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,  # Context réduit pour plus d'utilisateurs
        n_threads=24,  # Threads CPU réduits
        n_gpu_layers=n_gpu_layers,  # TOUT sur GPU
        n_batch=3072,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
        seed=42,
        # Paramètres Qwen
        rope_freq_base=1000000,
        rope_freq_scale=1.0,
        # Optimisations GPU
        f16_kv=True,
        logits_all=False,
        vocab_only=False,
        embedding=False,  # Pas d'embeddings
        low_vram=False,   # On a assez de VRAM
        # Optimisation multi-users
        n_threads_batch=12,  # Threads pour batching
               
        # Gestion mémoire GPU
        tensor_split=None,  # Pas de split si 1 seul GPU
        main_gpu=0,  # GPU principal
    )
    
    load_time = time.time() - start_load
    print(f"\n✅ Model loaded in {load_time:.1f} seconds")
    model_loaded.set(1)
    model_loading_duration_seconds.set(load_time)
    
    # Mettre à jour la métrique des couches GPU
    gpu_layers_offloaded.set(-1)  # -1 signifie toutes les couches
    
    # Test de performance
    print("\n🧪 Performance test...")
    test_start = time.time()
    test_prompt = "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"
    
    result = llm(test_prompt, max_tokens=10, temperature=0.1)
    test_time = time.time() - test_start
    
    print(f"⏱️ Time for 10 tokens: {test_time:.2f}s")
    print(f"📊 Speed: {10/test_time:.1f} tokens/second")
    
    if test_time > 1:
        print("\n⚠️ Performance seems limited, checking GPU usage...")
    else:
        print("\n✅ Excellent GPU performance!")
    
    print(f"\n{'='*60}")
    print("✅ QWEN2.5-32B Q6_K READY FOR PRODUCTION")
    print(f"{'='*60}\n")

# ===== LIFESPAN =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    print("=== Starting FastAPI Application ===")
    
    # Démarrer les tâches de fond
    metrics_task = asyncio.create_task(metrics_update_task())
    
    try:
        # Charger le modèle
        model_start = time.time()
        load_model()
        print("=== Model loaded, API ready ===")
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        model_loaded.set(0)
    
    yield
    
    # Nettoyage
    print("=== Shutting down ===")
    model_loaded.set(0)
    metrics_task.cancel()
    
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    # Fermer le thread pool
    stream_manager.executor.shutdown(wait=True)

# ===== APPLICATION FASTAPI =====
app = FastAPI(
    title="Qwen2.5-32B Q6_K Multi-User API",
    version="8.0.0",
    description="Production-ready API with streaming, interruption, and multi-user support",
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
    """Endpoint Prometheus metrics"""
    update_gpu_metrics()
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint avec informations système"""
    return {
        "message": "Qwen2.5-32B Q6_K Multi-User API",
        "status": "running" if llm is not None else "loading",
        "model": {
            "name": "Qwen2.5-32B-Instruct-Q6_K.gguf",
            "loaded": llm is not None,
            "size": "~25GB",
            "context": "4096 tokens",
            "gpu_layers": "all"
        },
        "system": {
            "max_concurrent_users": MAX_CONCURRENT_USERS,
            "active_users": len(stream_manager.resource_manager.active_users),
            "active_streams": len(stream_manager.active_streams),
            "rate_limiting": ENABLE_RATE_LIMITING
        },
        "features": [
            "Multi-user support (15+ concurrent)",
            "Streaming with async interruption",
            "Intelligent JSON parsing",
            "Rate limiting per user",
            "Resource management",
            "Optimized for conversational AI"
        ],
        "endpoints": {
            "/v1/chat/completions": "OpenAI-compatible chat endpoint",
            "/ws": "WebSocket streaming with interruption",
            "/v1/summary": "Extract structured summary",
            "/v1/models": "List available models",
            "/health": "Health check with detailed status",
            "/metrics": "Prometheus metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check détaillé"""
    health_status = {
        "status": "healthy" if llm is not None else "loading",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": llm is not None,
            "path": MODEL_PATH,
            "exists": os.path.exists(MODEL_PATH),
            "size": f"{os.path.getsize(MODEL_PATH) / (1024**3):.1f} GB" if os.path.exists(MODEL_PATH) else None
        },
        "system": {
            "active_users": len(stream_manager.resource_manager.active_users),
            "max_users": MAX_CONCURRENT_USERS,
            "active_streams": len(stream_manager.active_streams),
            "active_stream_ids": list(stream_manager.active_streams.keys())[:5]  # Top 5
        },
        "performance": {
            "json_parse_success": json_parse_success_total._value._value if hasattr(json_parse_success_total, '_value') else 0,
            "json_parse_failure": json_parse_failure_total._value._value if hasattr(json_parse_failure_total, '_value') else 0,
            "stream_cancellations": stream_cancellation_total._value._value if hasattr(stream_cancellation_total, '_value') else 0
        }
    }
    
    # Vérifier si on est en surcharge
    if len(stream_manager.resource_manager.active_users) >= MAX_CONCURRENT_USERS:
        health_status["status"] = "overloaded"
        health_status["message"] = "Maximum concurrent users reached"
    
    return health_status

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    """Liste les modèles disponibles"""
    start_time = time.time()
    try:
        result = {
            "object": "list",
            "data": [
                {
                    "id": "qwen2.5-32b-q6k",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Qwen Team",
                    "permission": [],
                    "root": "qwen2.5-32b",
                    "parent": None,
                    "ready": llm is not None,
                    "context_length": 4096,
                    "quantization": "Q6_K"
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

@app.post("/v1/chat/completions", response_model=Union[ChatCompletionResponse, None])
async def chat_completions(
    request: ChatCompletionRequest,
    user_id: str = Depends(get_user_id)
):
    """Endpoint compatible OpenAI avec gestion multi-users"""
    start_time = time.time()
    status = "success"
    
    # Vérifier le modèle
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Vérifier le rate limit
    if not rate_limiter.check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Vérifier les ressources
    if not stream_manager.resource_manager.can_accept_user(user_id):
        raise HTTPException(
            status_code=503, 
            detail=f"Server at capacity ({MAX_CONCURRENT_USERS} concurrent users)"
        )
    
    try:
        # Logging
        logger.info(f"[CHAT] User {user_id}: {len(request.messages)} messages")
        
        # Formatter le prompt
        prompt = format_messages_qwen(request.messages)
        
        # Gérer le streaming
        if request.stream:
            # Retourner un générateur pour le streaming
            return await handle_streaming_response(
                prompt, request, user_id, start_time
            )
        
        # Réponse non-streaming
        inference_start = time.time()
        
        response = llm(
            prompt,
            max_tokens=min(request.max_tokens or 200, MAX_TOKENS_PER_REQUEST),
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 40,
            stop=request.stop or ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            echo=False,
            repeat_penalty=1.1
        )
        
        inference_duration = time.time() - inference_start
        fastapi_inference_duration_seconds.labels(model="qwen2.5-32b-q6k").observe(inference_duration)
        
        # Métriques tokens
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        
        fastapi_inference_tokens_total.labels(type="prompt").inc(prompt_tokens)
        fastapi_inference_tokens_total.labels(type="completion").inc(completion_tokens)
        
        if inference_duration > 0:
            tps = completion_tokens / inference_duration
            fastapi_inference_tokens_per_second.set(tps)
            logger.info(f"[PERF] User {user_id}: {tps:.1f} tokens/sec")
        
        # Gérer la réponse JSON si demandée
        generated_text = response['choices'][0]['text'].strip()
        
        if request.response_format and request.response_format.get("type") == "json_object":
            generated_text = ensure_json_response(generated_text, request.response_format)
        
        # Construire la réponse
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
        
        fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="success").inc()
        return chat_response
        
    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="error").inc()
        logger.error(f"Generation error for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        fastapi_requests_total.labels(method="POST", endpoint="/v1/chat/completions", status=status).inc()
        fastapi_request_duration_seconds.labels(method="POST", endpoint="/v1/chat/completions").observe(time.time() - start_time)

async def handle_streaming_response(prompt: str, request: ChatCompletionRequest, 
                                  user_id: str, start_time: float):
    """Gère la réponse en streaming"""
    request_id = f"stream_{uuid.uuid4().hex[:8]}"
    
    async def generate():
        try:
            # Headers SSE
            yield "data: [DONE]\n\n".encode('utf-8')  # Pour initialiser le stream
            
            tokens_count = 0
            async for output in stream_manager.generate_async(
                llm,
                prompt,
                request_id,
                user_id,
                max_tokens=min(request.max_tokens or 200, MAX_TOKENS_PER_REQUEST),
                temperature=request.temperature or 0.1,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 40,
                stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
            ):
                token = output['choices'][0]['text']
                tokens_count += 1
                
                # Créer le chunk de stream
                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[StreamChoice(
                        index=0,
                        delta=StreamDelta(content=token),
                        finish_reason=None
                    )]
                )
                
                yield f"data: {chunk.json()}\n\n".encode('utf-8')
            
            # Chunk final
            final_chunk = ChatCompletionChunk(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[StreamChoice(
                    index=0,
                    delta=StreamDelta(),
                    finish_reason="stop"
                )]
            )
            
            yield f"data: {final_chunk.json()}\n\n".encode('utf-8')
            yield "data: [DONE]\n\n".encode('utf-8')
            
            # Métriques
            duration = time.time() - start_time
            if duration > 0:
                tps = tokens_count / duration
                fastapi_inference_tokens_per_second.set(tps)
            
            fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="success").inc()
            
        except Exception as e:
            logger.error(f"Streaming error for {user_id}: {str(e)}")
            fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="error").inc()
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode('utf-8')
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/v1/summary", dependencies=[Depends(verify_token)])
async def create_summary(
    request: dict,
    user_id: str = Depends(get_user_id)
):
    """Endpoint pour créer un résumé structuré"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Rate limiting
    if not rate_limiter.check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        messages = request.get("messages", [])
        validated_messages = [Message(**msg) for msg in messages]
        
        prompt = format_messages_qwen(validated_messages)
        
        logger.info(f"[SUMMARY] User {user_id}: {len(validated_messages)} messages")
        
        response = llm(
            prompt,
            max_tokens=500,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        )
        
        result_text = response['choices'][0]['text'].strip()
        parsed_result = smart_json_parse(result_text)
        
        return {
            "status": "success",
            "extraction": parsed_result,
            "usage": response['usage']
        }
        
    except Exception as e:
        logger.error(f"Summary error for {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/warmup", dependencies=[Depends(verify_token)])
async def warmup():
    """Endpoint pour préchauffer le modèle"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        warmup_prompt = "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"
        
        start_time = time.time()
        response = llm(
            warmup_prompt,
            max_tokens=10,
            temperature=0.1,
            echo=False
        )
        duration = time.time() - start_time
        
        return {
            "status": "success",
            "warmup_time": f"{duration:.2f}s",
            "model_ready": True,
            "response": response['choices'][0]['text'].strip()
        }
        
    except Exception as e:
        logger.error(f"Warmup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """WebSocket endpoint avec support multi-users et interruption"""
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    # Extraire l'ID utilisateur
    user_id = websocket.headers.get('X-User-ID', 
                                    hashlib.md5(f"{token}:{websocket.client.host}".encode()).hexdigest()[:8])
    
    # Vérifier les ressources
    if not stream_manager.resource_manager.can_accept_user(user_id):
        await websocket.close(code=1008, reason="Server at capacity")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    
    # Message de bienvenue
    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "qwen2.5-32b-q6k",
        "model_loaded": llm is not None,
        "user_id": user_id,
        "capabilities": [
            "French medical conversations",
            "Streaming responses",
            "2K context window",
            "Async stream cancellation",
            "Multi-user support"
        ]
    }
    await websocket.send_json(welcome_msg)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Gérer l'annulation
            if data.get("type") == "cancel_stream":
                request_id = data.get("request_id")
                if request_id:
                    cancelled = stream_manager.cancel_stream(request_id)
                    await websocket.send_json({
                        "type": "stream_cancelled",
                        "request_id": request_id,
                        "success": cancelled
                    })
                    logger.info(f"[WS] User {user_id} cancelled stream {request_id}")
                continue
            
            # Vérifier le modèle
            if llm is None:
                await websocket.send_json({"type": "error", "error": "Model not loaded"})
                continue
            
            # Vérifier le rate limit
            if not rate_limiter.check_rate_limit(user_id):
                await websocket.send_json({
                    "type": "error", 
                    "error": "Rate limit exceeded"
                })
                continue
            
            try:
                messages = [Message(**msg) for msg in data.get("messages", [])]
                request_id = data.get("request_id") or f"ws_{uuid.uuid4().hex[:8]}"
                
                logger.info(f"[WS] User {user_id} stream {request_id}: {len(messages)} messages")
                
                # Formatter le prompt
                prompt = format_messages_qwen(messages)
                
                await websocket.send_json({
                    "type": "stream_start",
                    "request_id": request_id
                })
                
                # Streaming avec gestion d'interruption
                full_response = ""
                tokens_count = 0
                start_time = time.time()
                cancelled = False
                
                try:
                    async for output in stream_manager.generate_async(
                        llm,
                        prompt,
                        request_id,
                        user_id,
                        max_tokens=min(data.get("max_tokens", 200), MAX_TOKENS_PER_REQUEST),
                        temperature=data.get("temperature", 0.1),
                        top_p=data.get("top_p", 0.9),
                        top_k=data.get("top_k", 40),
                        stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
                    ):
                        token = output['choices'][0]['text']
                        full_response += token
                        tokens_count += 1
                        
                        # Vérifier la connexion
                        if websocket.client_state.value != 1:  # 1 = CONNECTED
                            logger.info(f"[WS] User {user_id} disconnected during stream")
                            cancelled = True
                            break
                        
                        try:
                            await websocket.send_json({
                                "type": "stream_token",
                                "token": token,
                                "request_id": request_id
                            })
                        except Exception as e:
                            logger.info(f"[WS] Send error for {user_id}: {e}")
                            cancelled = True
                            break
                        
                except asyncio.CancelledError:
                    cancelled = True
                    logger.info(f"[WS] Stream {request_id} cancelled")
                
                # Calculer les métriques
                duration = time.time() - start_time
                tps = tokens_count / duration if duration > 0 else 0
                
                # Stream end seulement si connecté
                if websocket.client_state.value == 1:
                    try:
                        await websocket.send_json({
                            "type": "stream_end",
                            "full_response": full_response,
                            "tokens": tokens_count,
                            "request_id": request_id,
                            "cancelled": cancelled,
                            "duration": duration,
                            "tokens_per_second": tps
                        })
                    except:
                        pass
                
                logger.info(f"[WS] User {user_id} stream {request_id}: "
                          f"{tokens_count} tokens in {duration:.2f}s "
                          f"({tps:.1f} tps, cancelled: {cancelled})")
                
                # Métriques
                fastapi_inference_requests_total.labels(
                    model="qwen2.5-32b-q6k",
                    status="cancelled" if cancelled else "success"
                ).inc()
                
            except Exception as e:
                logger.error(f"[WS] Error for {user_id}: {str(e)}", exc_info=True)
                if websocket.client_state.value == 1:
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "error": str(e),
                            "request_id": data.get("request_id")
                        })
                    except:
                        pass
    
    except WebSocketDisconnect:
        logger.info(f"[WS] User {user_id} disconnected")
    finally:
        fastapi_websocket_connections.dec()
        # Libérer les ressources utilisateur
        stream_manager.resource_manager.active_users.discard(user_id)
        fastapi_concurrent_users.set(len(stream_manager.resource_manager.active_users))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Important: 1 seul worker pour partager le modèle
        log_level="info"
    )