#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Mixtral-8x7B GGUF avec llama-cpp-python
Version 6.0.0 - Production Ready avec Streaming Optimisé
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
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# ===== MÉTRIQUES PROMETHEUS =====
system_info = Info('fastapi_system', 'System information')
system_info.info({
    'model': 'mixtral-8x7b',
    'instance': socket.gethostname(),
    'pod_id': os.getenv('RUNPOD_POD_ID', 'local'),
    'version': '6.0.0'  # Version avec streaming optimisé
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
fastapi_streaming_latency_seconds = Histogram('fastapi_streaming_latency_seconds', 'Latency between token generation and sending')
model_loaded = Gauge('model_loaded', 'Whether the model is loaded (1) or not (0)')
model_loading_duration_seconds = Gauge('model_loading_duration_seconds', 'Time taken to load the model')
model_download_progress = Gauge('model_download_progress', 'Model download progress in percentage')
json_parse_success_total = Counter('json_parse_success_total', 'Number of successful JSON parses')
json_parse_failure_total = Counter('json_parse_failure_total', 'Number of failed JSON parses')

inference_queue = asyncio.Queue(maxsize=1000)
download_in_progress = False
download_complete = False

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
    json_schema: Optional[Dict[str, Any]] = None

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
def format_messages_mistral_conversational(messages: List[Message]) -> str:
    """Formater les messages pour une conversation naturelle sans JSON"""
    prompt_parts = []
    
    for i, message in enumerate(messages):
        if message.role == "system":
            # Premier message système sans <s> au début
            if i == 0:
                prompt_parts.append(f"<s>[INST] {message.content} [/INST]")
            else:
                prompt_parts.append(f"[INST] {message.content} [/INST]")
        elif message.role == "user":
            # Ajouter <s> seulement si ce n'est pas le premier message
            if i > 0:
                prompt_parts.append(f"<s>[INST] {message.content} [/INST]")
            else:
                prompt_parts.append(f"[INST] {message.content} [/INST]")
        elif message.role == "assistant":
            prompt_parts.append(f" {message.content}</s>")
    
    # S'assurer qu'on commence bien par <s>
    if prompt_parts and not prompt_parts[0].startswith("<s>"):
        prompt_parts[0] = "<s>" + prompt_parts[0]
    
    return "".join(prompt_parts)

def clean_escaped_json(text: str) -> str:
    """Nettoie les caractères d'échappement dans le JSON"""
    text = text.replace(r'\_', '_')
    text = text.replace('\\\\', '\\')
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

def extract_json_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrait le JSON d'un texte et retourne (json_str, remaining_text)
    Version améliorée qui gère plusieurs formats
    """
    text = text.strip()
    text = clean_escaped_json(text)
    
    if text.startswith('{') and text.endswith('}'):
        return text, None
    
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        remaining = text[:json_match.start()] + text[json_match.end():]
        return json_str, remaining.strip() if remaining.strip() else None
    
    json_prefix_match = re.search(r'(?:JSON|json|Json):\s*({.*?})', text, re.DOTALL)
    if json_prefix_match:
        json_str = json_prefix_match.group(1)
        remaining = text[:json_prefix_match.start()] + text[json_prefix_match.end():]
        return json_str, remaining.strip() if remaining.strip() else None
    
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
    """
    Parse intelligent du JSON avec plusieurs stratégies de récupération
    """
    original_text = text
    
    json_str, remaining_text = extract_json_from_text(text)
    
    if not json_str:
        logging.warning("Aucun JSON trouvé dans la réponse")
        json_parse_failure_total.inc()
        return {
            "error": "No JSON found in response",
            "original_response": original_text
        }
    
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
        logging.warning(f"Première tentative de parsing échouée: {e}")
    
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
        logging.warning(f"Deuxième tentative échouée: {e}")
    
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
        logging.error(f"Toutes les tentatives de parsing ont échoué: {e}")
        json_parse_failure_total.inc()
        
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
        
        if "_metadata" in parsed and not os.getenv("DEBUG_MODE"):
            del parsed["_metadata"]
        
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    
    return text

# ===== GESTION DU MODÈLE =====
def download_with_retry(url: str, dest: str, max_retries: int = 3, delay: int = 5):
    """Télécharge avec retry et gestion des erreurs 429"""
    import urllib.error
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        print("⚠️  ATTENTION: Pas de token HuggingFace trouvé!")
        print("   Définissez la variable HF_TOKEN pour éviter les erreurs 429")
        print("   export HF_TOKEN='votre_token_ici'")
    
    if hf_token and subprocess.run(["which", "huggingface-cli"], capture_output=True).returncode == 0:
        print("🔑 Token HuggingFace détecté, utilisation de huggingface-cli...")
        try:
            subprocess.run(["huggingface-cli", "login", "--token", hf_token], 
                         capture_output=True, check=True)
            
            hf_cmd = [
                "huggingface-cli", "download",
                "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
                "--local-dir", os.path.dirname(dest),
                "--local-dir-use-symlinks", "False"
            ]
            
            print("📥 Téléchargement avec huggingface-cli...")
            result = subprocess.run(hf_cmd, text=True)
            if result.returncode == 0:
                print("✅ Téléchargement réussi avec huggingface-cli!")
                return True
            else:
                print(f"⚠️ Échec huggingface-cli, tentative avec méthodes alternatives...")
        except Exception as e:
            print(f"⚠️ Erreur avec huggingface-cli: {e}")
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = delay * (2 ** attempt)
                print(f"\n⏳ Attente de {wait_time}s avant nouvelle tentative...")
                time.sleep(wait_time)
            
            print(f"\n📥 Tentative {attempt + 1}/{max_retries}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            if hf_token:
                headers['Authorization'] = f'Bearer {hf_token}'
                print("🔑 Utilisation du token HF dans les headers")
            
            request = urllib.request.Request(url, headers=headers)
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                model_download_progress.set(percent)
                sys.stdout.write(f'\rTéléchargement: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) ')
                sys.stdout.flush()
            
            urllib.request.urlretrieve(request.full_url, dest, reporthook=download_progress)
            print("\n✅ Téléchargement réussi!")
            return True
            
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"\n⚠️ Erreur 429: Limite de taux dépassée")
                if not hf_token:
                    print("💡 Conseil: Définissez HF_TOKEN pour éviter cette erreur")
                if attempt < max_retries - 1:
                    continue
            else:
                print(f"\n❌ Erreur HTTP {e.code}: {e.reason}")
                if attempt < max_retries - 1:
                    continue
                raise
    
    print("\n🔧 Tentative finale avec wget...")
    try:
        wget_cmd = ["wget", "-c", "-O", dest, url]
        if hf_token:
            wget_cmd.extend(["--header", f"Authorization: Bearer {hf_token}"])
        subprocess.run(wget_cmd, check=True)
        return True
    except:
        print("\n❌ Toutes les tentatives ont échoué")
        if not hf_token:
            print("\n💡 Solution: Définissez la variable d'environnement HF_TOKEN")
            print("   export HF_TOKEN='votre_token_huggingface'")
        return False

def download_model_if_needed():
    """Télécharger le modèle au premier démarrage si nécessaire"""
    global download_in_progress, download_complete
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 30_000_000_000:
            print(f"✅ Modèle trouvé: {file_size / (1024**3):.1f} GB")
            download_complete = True
            return
        else:
            print(f"⚠️ Modèle incomplet ({file_size / (1024**3):.1f} GB), reprise du téléchargement...")
    
    if download_in_progress:
        print("⏳ Téléchargement déjà en cours...")
        return
    
    download_in_progress = True
    
    urls = [
        MODEL_URL,
        "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf?download=true",
    ]
    
    print(f"\n{'='*60}")
    print(f"📥 TÉLÉCHARGEMENT DU MODÈLE MIXTRAL-8X7B")
    print(f"{'='*60}")
    print(f"📦 Taille: ~32.9 GB")
    print(f"📍 Destination: {MODEL_PATH}")
    print(f"⏱️ Temps estimé: 20-40 minutes")
    print(f"{'='*60}\n")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    try:
        start_time = time.time()
        
        success = False
        for url in urls:
            print(f"\n🔗 Essai avec : {url}")
            if download_with_retry(url, MODEL_PATH):
                success = True
                break
        
        if not success:
            raise Exception("Échec du téléchargement après toutes les tentatives")
        
        print("\n✅ Téléchargement terminé!")
        
        download_time = time.time() - start_time
        print(f"⏱️ Temps de téléchargement: {download_time/60:.1f} minutes")
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"📦 Taille du fichier: {file_size / (1024**3):.1f} GB")
        
        if file_size < 30_000_000_000:
            raise Exception(f"Fichier trop petit: {file_size} bytes")
        
        download_complete = True
        model_download_progress.set(100)
        
    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement: {e}")
        download_in_progress = False
        model_download_progress.set(0)
        
        print("\n" + "="*60)
        print("📋 TÉLÉCHARGEMENT MANUEL REQUIS")
        print("="*60)
        print("\nPour contourner la limite de HuggingFace :")
        print(f"\n1. Téléchargez le modèle avec votre navigateur :")
        print(f"   {MODEL_URL}")
        print(f"\n2. Ou utilisez wget avec reprise :")
        print(f"   wget -c '{MODEL_URL}' -O {MODEL_PATH}")
        print(f"\n3. Ou utilisez huggingface-cli :")
        print(f"   pip install huggingface-hub")
        print(f"   huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf --local-dir /workspace/models/")
        print("\n" + "="*60)
        
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
        n_batch=512,  # Gardé à 512 pour performance optimale
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
    system_mentions_json = False
    
    # Parcourir tous les messages pour construire le prompt
    for i, message in enumerate(messages):
        if message.role == "system":
            # Utiliser le prompt système fourni par VoxEngine
            system_content = message.content
            prompt_parts.append(f"[INST] {system_content} [/INST]")
            has_system_prompt = True
            # Vérifier si le système mentionne JSON
            if "json" in system_content.lower() or "JSON" in system_content:
                system_mentions_json = True
                
        elif message.role == "user":
            if i == 0 and not has_system_prompt:
                # Premier message utilisateur sans prompt système
                prompt_parts.append(f"[INST] {message.content} [/INST]")
            else:
                # Messages utilisateur suivants
                prompt_parts.append(f"<s>[INST] {message.content} [/INST]")
                
        elif message.role == "assistant":
            # Réponses de l'assistant
            prompt_parts.append(f" {message.content}</s>")
    
    # S'assurer que le prompt se termine correctement
    if not prompt_parts[-1].strip().endswith("</s>") and not prompt_parts[-1].strip().endswith("[/INST]"):
        prompt_parts.append(" ")
    
    return "".join(prompt_parts)

# ===== CLASSE POUR STREAMING OPTIMISÉ =====
class StreamingBuffer:
    """Buffer intelligent pour optimiser le streaming des tokens"""
    
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.buffer = []
        self.last_send_time = 0
        self.min_tokens = 3  # Minimum de tokens avant envoi
        self.max_delay_ms = 100  # Maximum 100ms d'attente
        self.punctuation = {'.', '!', '?', ',', ';', ':', ')', ']', '}'}
        
    async def add_token(self, token: str, request_id: str):
        """Ajouter un token au buffer et envoyer si nécessaire"""
        self.buffer.append(token)
        
        # Conditions pour envoyer
        should_send = False
        
        # 1. Assez de tokens
        if len(self.buffer) >= self.min_tokens:
            should_send = True
            
        # 2. Ponctuation trouvée
        if any(p in token for p in self.punctuation):
            should_send = True
            
        # 3. Timeout dépassé
        if self.last_send_time > 0:
            elapsed_ms = (time.time() - self.last_send_time) * 1000
            if elapsed_ms >= self.max_delay_ms and self.buffer:
                should_send = True
        
        if should_send:
            await self.flush(request_id)
    
    async def flush(self, request_id: str):
        """Envoyer tous les tokens bufferisés"""
        if not self.buffer:
            return
            
        combined_tokens = "".join(self.buffer)
        
        await self.ws.send_json({
            "type": "stream_token",
            "token": combined_tokens,
            "request_id": request_id,
            "timestamp": time.time(),
            "batch_size": len(self.buffer)
        })
        
        # Log pour debug streaming
        logging.debug(f"[STREAM] Envoyé {len(self.buffer)} tokens: '{combined_tokens}'")
        
        self.buffer = []
        self.last_send_time = time.time()
        
        # Force le flush du WebSocket
        await asyncio.sleep(0)

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
    version="6.0.0",
    description="API FastAPI pour Mixtral-8x7B avec streaming optimisé",
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
        "message": "Mixtral-8x7B GGUF API with Optimized Streaming",
        "version": "6.0.0",
        "status": "running" if llm is not None else "loading",
        "model": "Mixtral-8x7B-Instruct-v0.1.Q5_K_M.gguf",
        "model_loaded": llm is not None,
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "features": [
            "Optimized streaming with intelligent buffering",
            "Intelligent JSON parsing",
            "Natural conversation mode",
            "Summary extraction endpoint",
            "Prometheus metrics"
        ],
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/ws": "WebSocket - Streaming chat endpoint with buffer optimization (requires token in query)",
            "/ws/test": "WebSocket - Test streaming latency",
            "/v1/summary": "POST - Extract structured summary from conversation",
            "/v1/models": "GET - List available models",
            "/v1/warmup": "POST - Warmup model for faster first response",
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
        "streaming_optimization": "intelligent_buffering",
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

@app.post("/v1/summary", dependencies=[Depends(verify_token)])
async def create_summary(request: dict):
    """Endpoint pour créer un résumé structuré de la conversation"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        messages = request.get("messages", [])
        
        extraction_prompt = f"""Analyse cette conversation médicale et extrais UNIQUEMENT les informations explicitement fournies.
Retourne un JSON avec ces champs (null si non fourni) :

{{
  "nom": "valeur ou null",
  "prenom": "valeur ou null", 
  "dateNaissance": "format JJ/MM/AAAA ou null",
  "dejaPatient": "oui/non/null",
  "praticien": "Dr Nom ou null",
  "motif": "description ou null",
  "resume": "résumé court de la demande",
  "categorie": "appointment_create/emergency/etc"
}}

Conversation à analyser:
{chr(10).join([f"{m['role']}: {m['content']}" for m in messages])}

Retourne UNIQUEMENT le JSON, sans texte avant ou après."""

        response = llm(
            extraction_prompt,
            max_tokens=500,
            temperature=0.1,
            top_p=0.9,
            stop=["</s>", "[INST]", "[/INST]"]
        )
        
        result_text = response['choices'][0]['text'].strip()
        parsed_result = smart_json_parse(result_text)
        
        return {
            "status": "success",
            "extraction": parsed_result,
            "usage": response['usage']
        }
        
    except Exception as e:
        logging.error(f"Erreur création résumé: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Log pour debug
        logging.info(f"[CHAT] Messages reçus: {[(m.role, m.content[:50] + '...' if len(m.content) > 50 else m.content) for m in request.messages]}")
        
        # Déterminer le format à utiliser
        # Si le premier message est un prompt système qui mentionne JSON ou si response_format demande JSON
        needs_json_format = False
        if request.response_format and request.response_format.get("type") == "json_object":
            needs_json_format = True
        elif request.messages and request.messages[0].role == "system":
            system_content = request.messages[0].content.lower()
            if "json" in system_content or "extraction" in system_content:
                needs_json_format = True
        
        # Utiliser le bon formatteur
        if needs_json_format:
            prompt = format_messages_mistral(request.messages)
        else:
            # Pour les conversations naturelles
            prompt = format_messages_mistral_conversational(request.messages)
        
        # Ajouter le schema JSON si fourni
        if request.json_schema:
            schema_instruction = f"\n\nYour response must conform to this JSON schema:\n{json.dumps(request.json_schema, indent=2)}"
            prompt = prompt.rstrip() + schema_instruction + "\n\nResponse (JSON only):"
        
        logging.info(f"[CHAT] Format utilisé: {'JSON' if needs_json_format else 'Conversational'}")
        logging.debug(f"[CHAT] Prompt final (200 premiers chars): {prompt[:200]}...")
        
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
        
        generated_text = response['choices'][0]['text'].strip()
        
        # Si on attend du JSON et que response_format le demande
        if request.response_format and request.response_format.get("type") == "json_object":
            generated_text = ensure_json_response(generated_text, request.response_format)
            
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

@app.post("/v1/warmup", dependencies=[Depends(verify_token)])
async def warmup():
    """Endpoint pour préchauffer le modèle"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Faire une petite inférence pour warmup
        warmup_prompt = "<s>[INST] Bonjour [/INST]"
        
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
        logging.error(f"Erreur warmup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/debug/prompt", dependencies=[Depends(verify_token)])
async def debug_prompt(request: ChatCompletionRequest):
    """Endpoint de debug pour voir le prompt généré sans appeler le modèle"""
    try:
        # Déterminer le format
        needs_json_format = False
        if request.response_format and request.response_format.get("type") == "json_object":
            needs_json_format = True
        elif request.messages and request.messages[0].role == "system":
            system_content = request.messages[0].content.lower()
            if "json" in system_content or "extraction" in system_content:
                needs_json_format = True
        
        # Générer les deux formats
        prompt_conversational = format_messages_mistral_conversational(request.messages)
        prompt_json = format_messages_mistral(request.messages)
        
        return {
            "messages_received": [
                {
                    "role": m.role,
                    "content": m.content[:100] + "..." if len(m.content) > 100 else m.content
                } for m in request.messages
            ],
            "detected_format": "json" if needs_json_format else "conversational",
            "prompt_conversational": prompt_conversational[:500] + "..." if len(prompt_conversational) > 500 else prompt_conversational,
            "prompt_json": prompt_json[:500] + "..." if len(prompt_json) > 500 else prompt_json,
            "prompt_used": (prompt_json if needs_json_format else prompt_conversational)[:500] + "...",
            "model_config": {
                "temperature": request.temperature or 0.1,
                "max_tokens": request.max_tokens or 4096,
                "top_p": request.top_p or 0.9
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """Endpoint WebSocket avec streaming optimisé par buffer intelligent"""
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    
    # Désactiver Nagle's algorithm pour réduire la latence
    try:
        import socket as sock
        transport = websocket._connection._transport
        raw_socket = transport.get_extra_info('socket')
        if raw_socket:
            raw_socket.setsockopt(sock.IPPROTO_TCP, sock.TCP_NODELAY, 1)
            logging.info("[WS] TCP_NODELAY activé pour réduire la latence")
    except Exception as e:
        logging.warning(f"[WS] Impossible d'activer TCP_NODELAY: {e}")
    
    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "mixtral-8x7b",
        "model_loaded": llm is not None,
        "streaming_mode": "intelligent_buffering",
        "buffer_config": {
            "min_tokens": 3,
            "max_delay_ms": 100,
            "punctuation_split": True
        },
        "capabilities": [
            "French medical conversations",
            "Optimized streaming responses",
            "32K context"
        ]
    }
    await websocket.send_json(welcome_msg)
    
    try:
        while True:
            data = await websocket.receive_json()
            if llm is None:
                await websocket.send_json({"type": "error", "error": "Model not loaded"})
                continue
            
            try:
                messages = [Message(**msg) for msg in data.get("messages", [])]
                
                # Log pour debug
                logging.info(f"[WS] Messages reçus: {[(m.role, len(m.content)) for m in messages]}")
                
                # Utiliser le formatteur conversationnel par défaut pour WebSocket
                # Sauf si explicitement demandé autrement
                use_json_format = data.get("format") == "json"
                
                if use_json_format:
                    prompt = format_messages_mistral(messages)
                else:
                    prompt = format_messages_mistral_conversational(messages)
                
                logging.info(f"[WS] Format utilisé: {'JSON' if use_json_format else 'Conversational'}")
                
                request_id = data.get("request_id", str(uuid.uuid4()))
                
                await websocket.send_json({
                    "type": "stream_start",
                    "request_id": request_id,
                    "timestamp": time.time()
                })
                
                # Créer le buffer intelligent
                buffer = StreamingBuffer(websocket)
                
                # Créer une tâche qui force l'envoi périodique du buffer
                async def periodic_flush():
                    while True:
                        await asyncio.sleep(0.1)  # Vérifier toutes les 100ms
                        if buffer.buffer and (time.time() - buffer.last_send_time) > 0.1:
                            await buffer.flush(request_id)
                
                flush_task = asyncio.create_task(periodic_flush())
                
                try:
                    stream = llm(
                        prompt,
                        max_tokens=data.get("max_tokens", 200),
                        temperature=data.get("temperature", 0.7),
                        top_p=data.get("top_p", 0.9),
                        stop=["</s>", "[INST]", "[/INST]"],
                        stream=True
                    )
                    
                    full_response = ""
                    tokens_count = 0
                    stream_start_time = time.time()
                    
                    for output in stream:
                        token = output['choices'][0]['text']
                        if token:  # Ignorer les tokens vides
                            full_response += token
                            tokens_count += 1
                            
                            # Ajouter au buffer (envoi automatique selon les règles)
                            await buffer.add_token(token, request_id)
                    
                    # Envoyer les derniers tokens
                    await buffer.flush(request_id)
                    
                finally:
                    flush_task.cancel()
                    try:
                        await flush_task
                    except asyncio.CancelledError:
                        pass
                
                # Statistiques finales
                stream_duration = time.time() - stream_start_time
                tokens_per_second = tokens_count / stream_duration if stream_duration > 0 else 0
                
                await websocket.send_json({
                    "type": "stream_end",
                    "full_response": full_response,
                    "tokens": tokens_count,
                    "request_id": request_id,
                    "duration": stream_duration,
                    "tokens_per_second": tokens_per_second,
                    "timestamp": time.time()
                })
                
                logging.info(f"[WS] Stream terminé: {tokens_count} tokens en {stream_duration:.2f}s ({tokens_per_second:.1f} t/s)")
                
                fastapi_inference_requests_total.labels(model="mixtral-8x7b", status="success").inc()
                
            except Exception as e:
                logging.error(f"[WS] Erreur: {str(e)}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "request_id": data.get("request_id")
                })
    
    except WebSocketDisconnect:
        logging.info("[WS] Client déconnecté")
    finally:
        fastapi_websocket_connections.dec()

@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """WebSocket de test pour vérifier le streaming et mesurer la latence"""
    await websocket.accept()
    
    try:
        # Message de bienvenue
        await websocket.send_json({
            "type": "info",
            "message": "Test de streaming - Les tokens seront envoyés progressivement"
        })
        
        # Simuler un streaming parfait pour tester la latence réseau
        test_tokens = ["Bonjour", ",", " voici", " un", " test", " de", " streaming", " fluide", " et", " naturel", "."]
        
        await websocket.send_json({
            "type": "stream_start",
            "request_id": "test",
            "timestamp": time.time()
        })
        
        start_time = time.time()
        for i, token in enumerate(test_tokens):
            await websocket.send_json({
                "type": "stream_token",
                "token": token,
                "index": i,
                "timestamp": time.time(),
                "elapsed": time.time() - start_time
            })
            await asyncio.sleep(0.05)  # 50ms entre tokens pour simuler la génération
        
        await websocket.send_json({
            "type": "stream_end",
            "request_id": "test",
            "timestamp": time.time(),
            "total_duration": time.time() - start_time,
            "message": "Test terminé - Vérifiez que les tokens sont arrivés progressivement"
        })
        
    except WebSocketDisconnect:
        pass

@app.get("/test/streaming-latency", dependencies=[Depends(verify_token)])
async def test_streaming_latency():
    """Test la latence de génération token par token"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prompt = "<s>[INST] Compte de 1 à 5 [/INST]"
    
    # Test avec streaming
    stream_start = time.time()
    tokens = []
    
    stream = llm(prompt, max_tokens=20, temperature=0.1, stream=True)
    
    for output in stream:
        token = output['choices'][0]['text']
        if token:
            tokens.append({
                "token": token,
                "time": time.time() - stream_start
            })
    
    return {
        "prompt": prompt,
        "tokens": tokens,
        "total_tokens": len(tokens),
        "total_time": time.time() - stream_start,
        "config": {
            "n_batch": 512,
            "streaming": True
        },
        "analysis": {
            "avg_time_between_tokens": sum(t["time"] for t in tokens[1:]) / max(1, len(tokens)-1) if len(tokens) > 1 else 0,
            "tokens_per_second": len(tokens) / (time.time() - stream_start) if tokens else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration optimisée pour le streaming
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # Options pour optimiser le streaming
        loop="uvloop" if sys.platform != "win32" else "asyncio",
        ws_ping_interval=20,  # Ping toutes les 20 secondes (réduit)
        ws_ping_timeout=20,
        limit_concurrency=1000,
        backlog=2048,
        # Log pour debug
        log_level="info"
    )