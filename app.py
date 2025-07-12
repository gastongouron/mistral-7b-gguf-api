"""
API FastAPI pour servir le modèle Qwen2.5-32B GGUF avec llama-cpp-python
Optimisé pour conversations médicales françaises avec streaming et interruption
Version 7.0.0 - Qwen Edition
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
from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
import weakref

# Import des métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

# Configuration pour Qwen2.5-32B
MODEL_PATH = "/workspace/models/Qwen2.5-72B-Instruct-Q3_K_M.gguf"
MODEL_URL = "https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF/resolve/main/Qwen2.5-72B-Instruct-Q3_K_M.gguf"

API_TOKEN = os.getenv("API_TOKEN", "supersecret")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===== MÉTRIQUES PROMETHEUS =====
system_info = Info('fastapi_system', 'System information')
system_info.info({
    'model': 'qwen2.5-32b',
    'instance': socket.gethostname(),
    'pod_id': os.getenv('RUNPOD_POD_ID', 'local'),
    'version': '7.0.0'  # Version Qwen
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
stream_cancellation_total = Counter('stream_cancellation_total', 'Number of stream cancellations')
stream_cancellation_latency_seconds = Histogram('stream_cancellation_latency_seconds', 'Time from cancellation request to actual stop')

inference_queue = asyncio.Queue(maxsize=1000)
download_in_progress = False
download_complete = False

# ===== MODÈLES PYDANTIC =====
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-32b"
    messages: List[Message]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None  # Ajout pour support top_k
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

# ===== GESTIONNAIRE DE STREAMS INTERRUPTIBLES =====
class StreamManager:
    """Gère les streams actifs et leur interruption"""
    def __init__(self):
        self.active_streams = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
    
    def register_stream(self, request_id: str) -> threading.Event:
        """Enregistre un nouveau stream et retourne son event de cancellation"""
        cancel_event = threading.Event()
        with self._lock:
            self.active_streams[request_id] = {
                'cancel_event': cancel_event,
                'start_time': time.time(),
                'tokens_generated': 0,
                'cancelled_at': None
            }
        return cancel_event
    
    def cancel_stream(self, request_id: str) -> bool:
        """Annule un stream actif"""
        with self._lock:
            if request_id in self.active_streams:
                cancel_time = time.time()
                stream_info = self.active_streams[request_id]
                stream_info['cancel_event'].set()
                stream_info['cancelled_at'] = cancel_time
                
                # Métriques
                start_time = stream_info['start_time']
                stream_cancellation_total.inc()
                stream_cancellation_latency_seconds.observe(cancel_time - start_time)
                
                logging.info(f"[STREAM] Annulation demandée pour {request_id} après {stream_info['tokens_generated']} tokens")
                return True
        return False
    
    def unregister_stream(self, request_id: str):
        """Retire un stream de la liste active"""
        with self._lock:
            if request_id in self.active_streams:
                del self.active_streams[request_id]
    
    def update_token_count(self, request_id: str, count: int):
        """Met à jour le nombre de tokens générés"""
        with self._lock:
            if request_id in self.active_streams:
                self.active_streams[request_id]['tokens_generated'] = count
    
    def is_cancelled(self, request_id: str) -> bool:
        """Vérifie si un stream est annulé"""
        with self._lock:
            if request_id in self.active_streams:
                return self.active_streams[request_id]['cancel_event'].is_set()
        return False
    
    async def generate_async(self, llm_model, prompt: str, request_id: str, **kwargs) -> AsyncGenerator[Dict, None]:
        """Génère des tokens avec interruption côté client uniquement"""
        cancel_event = self.register_stream(request_id)
        
        # Utiliser la génération streaming normale de llama-cpp
        # mais vérifier l'annulation à chaque token côté envoi
        loop = asyncio.get_event_loop()
        
        # Réduire max_tokens pour limiter la génération inutile
        original_max_tokens = kwargs.get('max_tokens', 200)
        kwargs['max_tokens'] = min(original_max_tokens, 100)  # Limiter à 100 tokens par requête
        
        def generate_sync():
            """Génération synchrone dans un thread"""
            return llm_model(prompt, stream=True, **kwargs)
        
        # Lancer la génération dans un thread executor
        future = loop.run_in_executor(self.executor, generate_sync)
        
        try:
            # Obtenir le générateur
            stream = await future
            tokens_generated = 0
            
            # Parcourir les tokens
            for output in stream:
                # Vérifier l'annulation AVANT d'envoyer
                if cancel_event.is_set():
                    logging.info(f"[STREAM] Arrêt de l'envoi pour {request_id} après {tokens_generated} tokens")
                    # On ne peut pas arrêter la génération, mais on arrête l'envoi
                    break
                
                tokens_generated += 1
                self.update_token_count(request_id, tokens_generated)
                
                # Yield le token seulement si pas annulé
                yield output
                
                # Micro-pause pour permettre la réception du cancel
                await asyncio.sleep(0)
                
                # Si on a atteint la limite, demander plus si nécessaire
                if tokens_generated >= kwargs['max_tokens'] and tokens_generated < original_max_tokens:
                    if not cancel_event.is_set():
                        # Continuer avec un nouveau chunk
                        kwargs['max_tokens'] = min(100, original_max_tokens - tokens_generated)
                        stream = llm_model(prompt + output['choices'][0]['text'], stream=True, **kwargs)
                    
        except Exception as e:
            logging.error(f"[STREAM] Erreur: {str(e)}")
            raise
        finally:
            self.unregister_stream(request_id)
            
            # Logger ce qui s'est passé
            with self._lock:
                if request_id in self.active_streams:
                    info = self.active_streams[request_id]
                    if info.get('cancelled_at'):
                        cancelled_after = info['cancelled_at'] - info['start_time']
                        logging.info(f"[STREAM] {request_id} annulé après {cancelled_after:.2f}s, {tokens_generated} tokens envoyés")

# Instance globale du gestionnaire de streams
stream_manager = StreamManager()

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

# ===== FORMATTERS POUR QWEN =====
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
    # Ajouter le début de la réponse assistant
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
    
    # Pour Qwen, essayer d'abord avec huggingface-cli
    if hf_token and subprocess.run(["which", "huggingface-cli"], capture_output=True).returncode == 0:
        print("🔑 Token HuggingFace détecté, utilisation de huggingface-cli...")
        try:
            subprocess.run(["huggingface-cli", "login", "--token", hf_token], 
                         capture_output=True, check=True)
            
            hf_cmd = [
                "huggingface-cli", "download",
                "Qwen/Qwen2.5-32B-Instruct-GGUF",
                "qwen2_5-32b-instruct-q4_k_m.gguf",
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
    
    # Fallback sur les méthodes alternatives
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
        expected_size = 18_500_000_000  # ~18.5 GB pour Q4_K_M
        if file_size > expected_size * 0.95:  # 95% de la taille attendue
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
        "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/qwen2_5-32b-instruct-q4_k_m.gguf?download=true",
    ]
    
    print(f"\n{'='*60}")
    print(f"📥 TÉLÉCHARGEMENT DU MODÈLE QWEN2.5-32B")
    print(f"{'='*60}")
    print(f"📦 Taille: ~18.5 GB (Q4_K_M quantization)")
    print(f"📍 Destination: {MODEL_PATH}")
    print(f"⏱️ Temps estimé: 10-20 minutes")
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
        
        if file_size < 18_000_000_000:
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
        print(f"   huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF qwen2_5-32b-instruct-q4_k_m.gguf --local-dir /workspace/models/")
        print("\n" + "="*60)
        
        raise
    finally:
        download_in_progress = False

def load_model():
    """Charger le modèle GGUF avec configuration optimale pour Qwen2.5-32B"""
    global llm
    
    download_model_if_needed()
    
    print(f"Chargement du modèle Qwen2.5-32B depuis {MODEL_PATH}...")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        print(f"VRAM disponible: {vram_gb:.1f} GB")
        
        # Qwen2.5-32B Q4_K_M nécessite environ 20-25GB VRAM
        if vram_gb >= 48:  # L40 ou mieux
            n_gpu_layers = -1
            print("Configuration: Modèle entièrement sur GPU (optimal)")
        elif vram_gb >= 24:
            n_gpu_layers = 40  # La plupart des couches
            print("Configuration: 40 couches sur GPU")
        else:
            n_gpu_layers = 20
            print(f"⚠️ VRAM limitée ({vram_gb:.1f}GB), performance réduite")
    except Exception as e:
        n_gpu_layers = -1
        print(f"Impossible de détecter la VRAM ({e}), tentative de chargement complet sur GPU")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,  # Contexte réduit pour meilleures performances
        n_threads=16,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        use_mmap=True,
        use_mlock=False,
        verbose=True,
        seed=42,
        # Paramètres spécifiques à Qwen
        rope_freq_base=1000000,  # Important pour Qwen
        rope_freq_scale=1.0
    )
    
    print("Modèle Qwen2.5-32B chargé avec succès!")
    print(f"Configuration: {n_gpu_layers} couches GPU, contexte 8K tokens")

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
    # Fermer le thread pool executor
    stream_manager.executor.shutdown(wait=True)

# ===== APPLICATION FASTAPI =====
app = FastAPI(
    title="Qwen2.5-32B GGUF API",
    version="7.0.0",
    description="API FastAPI pour Qwen2.5-32B avec streaming et interruption asynchrone",
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
        "message": "Qwen2.5-32B GGUF API with Async Interruption",
        "status": "running" if llm is not None else "loading",
        "model": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "model_loaded": llm is not None,
        "download_complete": download_complete,
        "download_in_progress": download_in_progress,
        "features": [
            "Streaming responses",
            "Async stream interruption",
            "Intelligent JSON parsing",
            "Natural conversation mode",
            "Summary extraction endpoint",
            "Better instruction following than Mixtral"
        ],
        "active_streams": len(stream_manager.active_streams),
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/ws": "WebSocket - Streaming chat endpoint with interruption support (requires token in query)",
            "/v1/summary": "POST - Extract structured summary from conversation",
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
        "active_streams": len(stream_manager.active_streams),
        "json_parse_stats": {
            "success": json_parse_success_total._value._value if hasattr(json_parse_success_total, '_value') else 0,
            "failure": json_parse_failure_total._value._value if hasattr(json_parse_failure_total, '_value') else 0
        },
        "stream_cancellation_stats": {
            "total": stream_cancellation_total._value._value if hasattr(stream_cancellation_total, '_value') else 0,
            "active_streams": list(stream_manager.active_streams.keys())
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
        status["expected_size_gb"] = 18.5  # Q4_K_M
    return status

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    start_time = time.time()
    try:
        result = {
            "object": "list",
            "data": [
                {
                    "id": "qwen2.5-32b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Qwen Team",
                    "permission": [],
                    "root": "qwen2.5-32b",
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
        
        # Prompt adapté pour Qwen
        extraction_prompt = f"""<|im_start|>system
Analyse cette conversation médicale et extrais UNIQUEMENT les informations explicitement fournies.
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
<|im_end|>
<|im_start|>user
Conversation à analyser:
{chr(10).join([f"{m['role']}: {m['content']}" for m in messages])}

Retourne UNIQUEMENT le JSON, sans texte avant ou après.
<|im_end|>
<|im_start|>assistant
"""

        response = llm(
            extraction_prompt,
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
        needs_json_format = False
        if request.response_format and request.response_format.get("type") == "json_object":
            needs_json_format = True
        elif request.messages and request.messages[0].role == "system":
            system_content = request.messages[0].content.lower()
            if "json" in system_content or "extraction" in system_content:
                needs_json_format = True
        
        # Toujours utiliser le formatteur Qwen
        prompt = format_messages_qwen(request.messages)
        
        # Ajouter le schema JSON si fourni
        if request.json_schema:
            schema_instruction = f"\n\nYour response must conform to this JSON schema:\n{json.dumps(request.json_schema, indent=2)}"
            prompt = prompt.rstrip() + schema_instruction + "\n\nResponse (JSON only):"
        
        logging.info(f"[CHAT] Format utilisé: Qwen ChatML")
        logging.debug(f"[CHAT] Prompt final (200 premiers chars): {prompt[:200]}...")
        
        inference_start = time.time()
        
        response = llm(
            prompt,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 40,
            stop=request.stop or ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            echo=False,
            repeat_penalty=1.1
        )
        
        await inference_queue.get()
        
        inference_duration = time.time() - inference_start
        fastapi_inference_duration_seconds.labels(model="qwen2.5-32b").observe(inference_duration)
        
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
        
        fastapi_inference_requests_total.labels(model="qwen2.5-32b", status="success").inc()
        return chat_response
        
    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        fastapi_inference_requests_total.labels(model="qwen2.5-32b", status="error").inc()
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
        logging.error(f"Erreur warmup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/debug/prompt", dependencies=[Depends(verify_token)])
async def debug_prompt(request: ChatCompletionRequest):
    """Endpoint de debug pour voir le prompt généré sans appeler le modèle"""
    try:
        # Générer le prompt Qwen
        prompt_qwen = format_messages_qwen(request.messages)
        
        return {
            "messages_received": [
                {
                    "role": m.role,
                    "content": m.content[:100] + "..." if len(m.content) > 100 else m.content
                } for m in request.messages
            ],
            "format": "ChatML (Qwen)",
            "prompt": prompt_qwen[:500] + "..." if len(prompt_qwen) > 500 else prompt_qwen,
            "model_config": {
                "temperature": request.temperature or 0.1,
                "max_tokens": request.max_tokens or 4096,
                "top_p": request.top_p or 0.9,
                "top_k": request.top_k or 40
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """Endpoint WebSocket avec streaming et interruption asynchrone"""
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    
    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "qwen2.5-32b",
        "model_loaded": llm is not None,
        "capabilities": [
            "French medical conversations",
            "Streaming responses",
            "8K context",
            "Async stream cancellation",
            "Better instruction following"
        ]
    }
    await websocket.send_json(welcome_msg)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Gérer cancel_stream
            if data.get("type") == "cancel_stream":
                request_id = data.get("request_id")
                if request_id:
                    cancelled = stream_manager.cancel_stream(request_id)
                    await websocket.send_json({
                        "type": "stream_cancelled", 
                        "request_id": request_id,
                        "success": cancelled
                    })
                    logging.info(f"[WS] Cancel stream {request_id}: {'success' if cancelled else 'not found'}")
                continue
            
            if llm is None:
                await websocket.send_json({"type": "error", "error": "Model not loaded"})
                continue
            
            try:
                messages = [Message(**msg) for msg in data.get("messages", [])]
                request_id = data.get("request_id") or f"ws_{uuid.uuid4().hex[:8]}"
                
                # Log pour debug
                logging.info(f"[WS] Nouveau stream {request_id}: {[(m.role, len(m.content)) for m in messages]}")
                
                # Toujours utiliser le formatteur Qwen
                prompt = format_messages_qwen(messages)
                
                logging.info(f"[WS] Format utilisé: Qwen ChatML")
                
                await websocket.send_json({
                    "type": "stream_start",
                    "request_id": request_id
                })
                
                # Utiliser le stream manager pour la génération asynchrone
                full_response = ""
                tokens_count = 0
                start_time = time.time()
                last_token_time = start_time
                cancelled = False
                
                try:
                    async for output in stream_manager.generate_async(
                        llm,
                        prompt,
                        request_id,
                        max_tokens=data.get("max_tokens", 200),
                        temperature=data.get("temperature", 0.1),
                        top_p=data.get("top_p", 0.9),
                        top_k=data.get("top_k", 40),
                        stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
                    ):
                        token = output['choices'][0]['text']
                        full_response += token
                        tokens_count += 1
                        last_token_time = time.time()
                        
                        # Vérifier si la connexion est toujours ouverte
                        if websocket.client_state.value != 1:  # 1 = CONNECTED
                            logging.info(f"[WS] Client déconnecté pendant stream {request_id}")
                            cancelled = True
                            break
                        
                        try:
                            await websocket.send_json({
                                "type": "stream_token",
                                "token": token,
                                "request_id": request_id
                            })
                        except Exception as e:
                            logging.info(f"[WS] Erreur envoi token: {e}")
                            cancelled = True
                            break
                        
                except asyncio.CancelledError:
                    cancelled = True
                    logging.info(f"[WS] Stream {request_id} annulé après {tokens_count} tokens")
                
                # Calculer les métriques
                duration = time.time() - start_time
                if duration > 0:
                    tps = tokens_count / duration
                    fastapi_inference_tokens_per_second.set(tps)
                
                # Le stream a été interrompu si on n'a pas atteint max_tokens
                was_cancelled = cancelled or (tokens_count < data.get("max_tokens", 200) - 10)
                
                await websocket.send_json({
                    "type": "stream_end",
                    "full_response": full_response,
                    "tokens": tokens_count,
                    "request_id": request_id,
                    "cancelled": was_cancelled,
                    "duration": duration,
                    "tokens_per_second": tps if duration > 0 else 0,
                    "time_since_last_token": time.time() - last_token_time
                })
                
                logging.info(f"[WS] Stream {request_id} terminé: {tokens_count} tokens en {duration:.2f}s (annulé: {was_cancelled})")
                
                fastapi_inference_requests_total.labels(
                    model="qwen2.5-32b", 
                    status="cancelled" if was_cancelled else "success"
                ).inc()
                
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)