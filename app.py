#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Mixtral-8x7B GGUF avec llama-cpp-python
Version avec STREAMING NATIF pour conversations médicales françaises
Production-ready avec gestion d'erreurs, métriques et resilience
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
from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator
from llama_cpp import Llama
import traceback
from datetime import datetime

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
logger = logging.getLogger(__name__)

# ===== CONFIGURATION STREAMING =====
STREAMING_CONFIG = {
    "CHUNK_SIZE": 5,  # Tokens par chunk avant envoi
    "SENTENCE_DELIMITERS": ['. ', '? ', '! ', '.\n', '?\n', '!\n', '... ', ', '],
    "MAX_BUFFER_SIZE": 100,  # Caractères max dans le buffer
    "YIELD_INTERVAL": 0.001,  # Pause entre les yields pour éviter la saturation
    "FIRST_TOKEN_PRIORITY": True,  # Envoyer le premier token immédiatement
}

# ===== MÉTRIQUES PROMETHEUS ÉTENDUES =====
system_info = Info('fastapi_system', 'System information')
system_info.info({
    'model': 'mixtral-8x7b',
    'instance': socket.gethostname(),
    'pod_id': os.getenv('RUNPOD_POD_ID', 'local'),
    'version': '5.0.0',  # Version avec streaming
    'streaming_enabled': 'true'
})

# Métriques existantes
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
fastapi_inference_requests_total = Counter('fastapi_inference_requests_total', 'Total number of inference requests', ['model', 'status', 'mode'])
fastapi_inference_duration_seconds = Histogram('fastapi_inference_duration_seconds', 'Inference duration in seconds', ['model', 'mode'], buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0])
fastapi_inference_queue_size = Gauge('fastapi_inference_queue_size', 'Current inference queue size')
fastapi_inference_tokens_total = Counter('fastapi_inference_tokens_total', 'Total tokens processed', ['type'])
fastapi_inference_tokens_per_second = Gauge('fastapi_inference_tokens_per_second', 'Tokens generated per second')
model_loaded = Gauge('model_loaded', 'Whether the model is loaded (1) or not (0)')
model_loading_duration_seconds = Gauge('model_loading_duration_seconds', 'Time taken to load the model')

# Nouvelles métriques pour le streaming
streaming_requests_total = Counter('streaming_requests_total', 'Total streaming requests')
streaming_first_token_latency = Histogram('streaming_first_token_latency_seconds', 'Time to first token in streaming mode', buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
streaming_chunks_sent = Counter('streaming_chunks_sent_total', 'Total streaming chunks sent')
streaming_errors = Counter('streaming_errors_total', 'Streaming errors', ['error_type'])
websocket_active_streams = Gauge('websocket_active_streams', 'Currently active streaming sessions')

# Gestion de la queue et du modèle
inference_queue = asyncio.Queue(maxsize=100)
active_streams = {}
llm = None

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
        logger.debug(f"Impossible de collecter les métriques GPU: {e}")

def update_system_metrics():
    try:
        cpu_usage_percent.set(psutil.cpu_percent(interval=0.1))
        mem = psutil.virtual_memory()
        memory_used_bytes.set(mem.used)
        memory_total_bytes.set(mem.total)
        disk = psutil.disk_usage('/')
        disk_usage_percent.set(disk.percent)
    except Exception as e:
        logger.debug(f"Impossible de collecter les métriques système: {e}")

async def metrics_update_task():
    while True:
        update_gpu_metrics()
        update_system_metrics()
        fastapi_inference_queue_size.set(inference_queue.qsize())
        websocket_active_streams.set(len(active_streams))
        await asyncio.sleep(5)

# ===== GESTION DU STREAMING =====
class StreamingSession:
    """Classe pour gérer une session de streaming"""
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.start_time = time.time()
        self.first_token_time = None
        self.token_count = 0
        self.chunk_count = 0
        self.buffer = ""
        self.is_active = True
        
    async def send_chunk(self, text: str, chunk_type: str = "text_chunk"):
        """Envoie un chunk avec gestion d'erreur"""
        if not self.is_active:
            return False
            
        try:
            self.chunk_count += 1
            await self.websocket.send_json({
                "type": chunk_type,
                "text": text,
                "chunk_id": self.chunk_count,
                "timestamp": int(time.time() * 1000)
            })
            streaming_chunks_sent.inc()
            return True
        except Exception as e:
            logger.error(f"[STREAM {self.session_id}] Erreur envoi chunk: {e}")
            streaming_errors.labels(error_type="send_error").inc()
            self.is_active = False
            return False
    
    def should_send_buffer(self) -> bool:
        """Détermine si le buffer doit être envoyé"""
        if not self.buffer:
            return False
            
        # Vérifier les délimiteurs de phrase
        for delimiter in STREAMING_CONFIG["SENTENCE_DELIMITERS"]:
            if delimiter in self.buffer:
                return True
                
        # Vérifier la taille du buffer
        if len(self.buffer) >= STREAMING_CONFIG["MAX_BUFFER_SIZE"]:
            return True
            
        return False
    
    async def close(self):
        """Ferme proprement la session"""
        self.is_active = False
        if self.session_id in active_streams:
            del active_streams[self.session_id]

# ===== STREAMING LLM =====
async def stream_llm_response(
    prompt: str,
    session: StreamingSession,
    temperature: float = 0.7,
    max_tokens: int = 200,
    stop_sequences: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Génère et stream la réponse du LLM
    Retourne les statistiques de génération
    """
    if not llm:
        raise ValueError("Model not loaded")
    
    stats = {
        "total_tokens": 0,
        "time_to_first_token": None,
        "total_time": 0,
        "tokens_per_second": 0,
        "full_text": ""
    }
    
    try:
        # Créer le stream
        stream = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            stop=stop_sequences or ["</s>", "[INST]", "[/INST]"],
            echo=False,
            repeat_penalty=1.1,
            stream=True
        )
        
        # Envoyer le signal de début
        await session.send_chunk("", "stream_start")
        
        # Traiter le stream
        for i, output in enumerate(stream):
            if not session.is_active:
                break
                
            # Extraire le token
            token = output['choices'][0]['text']
            if not token:
                continue
                
            session.token_count += 1
            stats["total_tokens"] += 1
            stats["full_text"] += token
            
            # Enregistrer le temps du premier token
            if session.first_token_time is None:
                session.first_token_time = time.time()
                stats["time_to_first_token"] = session.first_token_time - session.start_time
                streaming_first_token_latency.observe(stats["time_to_first_token"])
                logger.info(f"[STREAM {session.session_id}] Premier token en {stats['time_to_first_token']*1000:.0f}ms")
            
            # Ajouter au buffer
            session.buffer += token
            
            # Envoyer immédiatement le premier token si configuré
            if (i == 0 and STREAMING_CONFIG["FIRST_TOKEN_PRIORITY"]) or session.should_send_buffer():
                if await session.send_chunk(session.buffer):
                    session.buffer = ""
                else:
                    break
            
            # Yield périodique pour éviter de bloquer
            if session.token_count % 10 == 0:
                await asyncio.sleep(STREAMING_CONFIG["YIELD_INTERVAL"])
        
        # Envoyer le buffer restant
        if session.buffer and session.is_active:
            await session.send_chunk(session.buffer)
        
        # Calculer les stats finales
        stats["total_time"] = time.time() - session.start_time
        stats["tokens_per_second"] = stats["total_tokens"] / stats["total_time"] if stats["total_time"] > 0 else 0
        
        # Envoyer le signal de fin avec les stats
        if session.is_active:
            await session.websocket.send_json({
                "type": "stream_end",
                "stats": {
                    "total_tokens": stats["total_tokens"],
                    "total_time_ms": round(stats["total_time"] * 1000),
                    "time_to_first_token_ms": round(stats["time_to_first_token"] * 1000) if stats["time_to_first_token"] else None,
                    "tokens_per_second": round(stats["tokens_per_second"], 2),
                    "chunks_sent": session.chunk_count
                }
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"[STREAM {session.session_id}] Erreur génération: {str(e)}")
        streaming_errors.labels(error_type="generation_error").inc()
        
        if session.is_active:
            await session.websocket.send_json({
                "type": "error",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            })
        
        raise

# ===== MODÈLES PYDANTIC =====
class Message(BaseModel):
    role: str
    content: str

class StreamingRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 200
    stream: Optional[bool] = True
    request_id: Optional[str] = None
    mode: Optional[str] = "conversational"  # conversational, json, extraction

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
def load_model():
    """Charger le modèle GGUF avec configuration optimale pour Mixtral-8x7B"""
    global llm
    
    logger.info(f"Chargement du modèle Mixtral-8x7B depuis {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle non trouvé: {MODEL_PATH}")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        logger.info(f"VRAM disponible: {vram_gb:.1f} GB")
        
        if vram_gb >= 40:
            n_gpu_layers = -1
            logger.info("Configuration: Modèle entièrement sur GPU (optimal pour streaming)")
        elif vram_gb >= 24:
            n_gpu_layers = 28
            logger.info("Configuration: 28 couches sur GPU")
        else:
            n_gpu_layers = 16
            logger.warning(f"⚠️ VRAM limitée ({vram_gb:.1f}GB), performance streaming réduite")
    except Exception as e:
        n_gpu_layers = -1
        logger.info(f"Impossible de détecter la VRAM ({e}), chargement complet sur GPU")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=12,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        use_mmap=True,
        use_mlock=False,
        verbose=True,
        seed=42,
        # Optimisations pour le streaming
        cache=True,
        cache_type="ram",
        low_vram=False
    )
    
    logger.info("✅ Modèle Mixtral-8x7B chargé avec succès!")
    logger.info(f"Configuration streaming: {n_gpu_layers} couches GPU, contexte 32K tokens")

def format_messages_for_streaming(messages: List[Message], mode: str = "conversational") -> str:
    """
    Formater les messages pour Mistral avec différents modes
    """
    if mode == "conversational":
        # Mode conversationnel naturel pour le streaming
        system_prompt = """Tu es un assistant médical français empathique et professionnel.
Réponds de manière naturelle et conversationnelle, en phrases courtes et claires.
Sois concis mais chaleureux. Maximum 2-3 phrases par réponse."""
    else:
        # Mode normal (json, extraction, etc.)
        return format_messages_mistral(messages)
    
    # Remplacer ou ajouter le system prompt
    prompt_parts = ["<s>"]
    
    if messages and messages[0].role == "system":
        # Remplacer par notre prompt conversationnel
        messages = messages.copy()
        messages[0] = Message(role="system", content=system_prompt)
    else:
        # Ajouter notre prompt
        messages = [Message(role="system", content=system_prompt)] + messages
    
    # Formatter selon le template Mistral
    has_system = False
    for i, message in enumerate(messages):
        if message.role == "system":
            prompt_parts.append(f"{message.content}\n\n")
            has_system = True
        elif message.role == "user":
            if i == 1 and has_system:  # Premier user après system
                prompt_parts.append(f"[INST] {message.content} [/INST]")
            else:
                prompt_parts.append(f"<s>[INST] {message.content} [/INST]")
        elif message.role == "assistant":
            prompt_parts.append(f" {message.content}</s>")
    
    if not prompt_parts[-1].strip().endswith("</s>"):
        prompt_parts.append(" ")
    
    return "".join(prompt_parts)

def format_messages_mistral(messages: List[Message]) -> str:
    """Format original pour compatibilité"""
    prompt_parts = ["<s>"]
    has_system_prompt = False
    
    if messages and messages[0].role == "system":
        prompt_parts.append(f"{messages[0].content}\n\n")
        messages = messages[1:]
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
    logger.info("=== Démarrage de l'application FastAPI Streaming ===")
    metrics_task = asyncio.create_task(metrics_update_task())
    
    try:
        model_start = time.time()
        load_model()
        model_loading_duration_seconds.set(time.time() - model_start)
        model_loaded.set(1)
        logger.info("=== Modèle chargé, API prête pour le streaming ===")
    except Exception as e:
        logger.error(f"Erreur fatale lors du chargement du modèle: {e}")
        model_loaded.set(0)
    
    yield
    
    logger.info("=== Arrêt de l'application ===")
    
    # Fermer toutes les sessions de streaming actives
    for session_id, session in list(active_streams.items()):
        logger.info(f"Fermeture session streaming {session_id}")
        await session.close()
    
    model_loaded.set(0)
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass

# ===== APPLICATION FASTAPI =====
app = FastAPI(
    title="Mixtral-8x7B Streaming API",
    version="5.0.0",
    description="API FastAPI pour Mixtral-8x7B avec streaming natif pour conversations temps réel",
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

@app.get("/")
async def root():
    return {
        "message": "Mixtral-8x7B Streaming API",
        "status": "running" if llm is not None else "loading",
        "model": "Mixtral-8x7B-Instruct-v0.1.Q5_K_M.gguf",
        "model_loaded": llm is not None,
        "features": [
            "Token streaming for real-time responses",
            "Intelligent chunk batching",
            "Low latency first token",
            "Production-ready error handling",
            "Prometheus metrics",
            "Multiple conversation modes"
        ],
        "streaming_config": STREAMING_CONFIG,
        "active_streams": len(active_streams),
        "endpoints": {
            "/ws/stream": "WebSocket - Streaming conversation endpoint (recommended)",
            "/ws": "WebSocket - Legacy endpoint with streaming support",
            "/v1/chat/completions": "POST - OpenAI compatible endpoint",
            "/health": "GET - Health check with detailed status",
            "/metrics": "GET - Prometheus metrics"
        }
    }

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy" if llm is not None else "loading",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "active_streams": len(active_streams),
        "streaming_stats": {
            "total_requests": streaming_requests_total._value._value if hasattr(streaming_requests_total, '_value') else 0,
            "chunks_sent": streaming_chunks_sent._value._value if hasattr(streaming_chunks_sent, '_value') else 0,
            "errors": streaming_errors._child_samples() if hasattr(streaming_errors, '_child_samples') else {}
        },
        "system_metrics": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "inference_queue_size": inference_queue.qsize()
        }
    }
    
    if os.path.exists(MODEL_PATH):
        health_status["model_size"] = f"{os.path.getsize(MODEL_PATH) / (1024**3):.1f} GB"
    
    return health_status

@app.get("/metrics")
async def metrics():
    update_gpu_metrics()
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ===== WEBSOCKET STREAMING PRINCIPAL =====
@app.websocket("/ws/stream")
async def websocket_streaming_endpoint(websocket: WebSocket, token: str = Query(...)):
    """
    Endpoint WebSocket optimisé pour le streaming conversationnel
    Conçu spécifiquement pour l'intégration avec Voximplant
    """
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    streaming_requests_total.inc()
    
    session_id = f"stream_{uuid.uuid4().hex[:8]}"
    session = StreamingSession(session_id, websocket)
    active_streams[session_id] = session
    
    logger.info(f"[STREAM {session_id}] Nouvelle connexion WebSocket streaming")
    
    # Message de bienvenue
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "session_id": session_id,
        "model": "mixtral-8x7b",
        "streaming": True,
        "timestamp": int(time.time() * 1000)
    })
    
    try:
        while True:
            # Recevoir la requête
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON",
                    "timestamp": int(time.time() * 1000)
                })
                continue
            
            # Validation du modèle
            if llm is None:
                await websocket.send_json({
                    "type": "error",
                    "error": "Model not loaded",
                    "timestamp": int(time.time() * 1000)
                })
                streaming_errors.labels(error_type="model_not_loaded").inc()
                continue
            
            try:
                # Parser la requête
                messages = [Message(**msg) for msg in data.get("messages", [])]
                if not messages:
                    raise ValueError("No messages provided")
                
                request_id = data.get("request_id", f"req_{session.chunk_count}")
                mode = data.get("mode", "conversational")
                temperature = data.get("temperature", 0.7)
                max_tokens = data.get("max_tokens", 200)
                
                logger.info(f"[STREAM {session_id}] Requête {request_id}, mode: {mode}")
                
                # Gérer la queue d'inférence
                await inference_queue.put({"session_id": session_id, "request_id": request_id})
                
                try:
                    # Formatter le prompt selon le mode
                    if mode == "conversational":
                        prompt = format_messages_for_streaming(messages, mode)
                    else:
                        prompt = format_messages_mistral(messages)
                    
                    # Réinitialiser la session pour cette requête
                    session.start_time = time.time()
                    session.first_token_time = None
                    session.token_count = 0
                    session.buffer = ""
                    
                    # Streamer la réponse
                    stats = await stream_llm_response(
                        prompt=prompt,
                        session=session,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Métriques
                    fastapi_inference_requests_total.labels(
                        model="mixtral-8x7b", 
                        status="success", 
                        mode="streaming"
                    ).inc()
                    
                    fastapi_inference_duration_seconds.labels(
                        model="mixtral-8x7b", 
                        mode="streaming"
                    ).observe(stats["total_time"])
                    
                    fastapi_inference_tokens_total.labels(type="completion").inc(stats["total_tokens"])
                    
                    if stats["tokens_per_second"] > 0:
                        fastapi_inference_tokens_per_second.set(stats["tokens_per_second"])
                    
                    logger.info(f"[STREAM {session_id}] Génération terminée: {stats['total_tokens']} tokens, {stats['tokens_per_second']:.1f} t/s")
                    
                finally:
                    await inference_queue.get()
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[STREAM {session_id}] Erreur: {str(e)}")
                logger.error(traceback.format_exc())
                
                streaming_errors.labels(error_type="processing_error").inc()
                fastapi_inference_requests_total.labels(
                    model="mixtral-8x7b", 
                    status="error", 
                    mode="streaming"
                ).inc()
                
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "request_id": data.get("request_id"),
                    "timestamp": int(time.time() * 1000)
                })
    
    except WebSocketDisconnect:
        logger.info(f"[STREAM {session_id}] Client déconnecté")
    except Exception as e:
        logger.error(f"[STREAM {session_id}] Erreur WebSocket: {str(e)}")
        streaming_errors.labels(error_type="websocket_error").inc()
    finally:
        await session.close()
        fastapi_websocket_connections.dec()
        logger.info(f"[STREAM {session_id}] Session fermée")

# ===== WEBSOCKET HYBRIDE (streaming + JSON) =====
@app.websocket("/ws")
async def websocket_hybrid_endpoint(websocket: WebSocket, token: str = Query(...)):
    """
    Endpoint WebSocket hybride : 
    - Mode streaming pour les conversations
    - Mode JSON pour extractions/résumés
    Parfait pour Voximplant qui a besoin des deux
    """
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    fastapi_websocket_connections.inc()
    
    session_id = f"hybrid_{uuid.uuid4().hex[:8]}"
    logger.info(f"[WS-HYBRID {session_id}] Nouvelle connexion WebSocket hybride")
    
    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "mixtral-8x7b",
        "model_loaded": llm is not None,
        "capabilities": ["streaming", "json", "extraction"],
        "session_id": session_id,
        "endpoints": {
            "streaming": "Pour conversations naturelles (response_format: null ou text)",
            "json": "Pour extraction/résumé (response_format: json)"
        }
    }
    await websocket.send_json(welcome_msg)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if llm is None:
                await websocket.send_json({"type": "error", "error": "Model not loaded"})
                continue
            
            try:
                # Déterminer le mode selon response_format
                response_format = data.get("response_format", "text")
                stream_mode = data.get("stream", True)
                
                # Mode extraction/JSON : JAMAIS de streaming
                if response_format == "json" or data.get("mode") == "extraction":
                    logger.info(f"[WS-HYBRID {session_id}] Mode JSON/Extraction (pas de streaming)")
                    
                    # Créer la requête
                    messages = [Message(**msg) for msg in data.get("messages", [])]
                    
                    await inference_queue.put({"session_id": session_id, "mode": "json"})
                    
                    try:
                        # Pour l'extraction, on utilise toujours le prompt d'extraction
                        if data.get("mode") == "extraction":
                            prompt = format_messages_mistral([
                                Message(role="system", content=LLM_CONFIG.EXTRACTION_PROMPT),
                                Message(role="user", content=messages[0].content if messages else "")
                            ])
                        else:
                            prompt = format_messages_mistral(messages)
                        
                        start_time = time.time()
                        
                        # Génération SANS streaming pour avoir la réponse complète
                        response = llm(
                            prompt,
                            max_tokens=data.get("max_tokens", 500),  # Plus de tokens pour JSON
                            temperature=data.get("temperature", 0.1),  # Température basse pour JSON
                            top_p=0.9,
                            top_k=40,
                            stop=["</s>", "[INST]", "[/INST]"],
                            echo=False,
                            repeat_penalty=1.1,
                            stream=False  # PAS de streaming pour JSON
                        )
                        
                        elapsed = time.time() - start_time
                        generated_text = response['choices'][0]['text'].strip()
                        
                        # Pour le mode extraction, nettoyer et parser le JSON
                        if data.get("mode") == "extraction":
                            # Extraire le JSON de la réponse
                            json_str, _ = extract_json_from_text(generated_text)
                            if json_str:
                                try:
                                    parsed_json = json.loads(json_str)
                                    generated_text = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                                except:
                                    pass
                        
                        # Envoyer la réponse complète
                        response_msg = {
                            "type": "completion",
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": generated_text
                                },
                                "finish_reason": response['choices'][0]['finish_reason']
                            }],
                            "usage": response['usage'],
                            "time_ms": round(elapsed * 1000),
                            "tokens_per_second": round(response['usage']['completion_tokens'] / elapsed, 2),
                            "mode": "json",
                            "request_id": data.get("request_id")
                        }
                        
                        await websocket.send_json(response_msg)
                        
                        # Métriques
                        fastapi_inference_requests_total.labels(
                            model="mixtral-8x7b", 
                            status="success", 
                            mode="json"
                        ).inc()
                        
                        logger.info(f"[WS-HYBRID {session_id}] Réponse JSON générée en {elapsed:.2f}s")
                        
                    finally:
                        await inference_queue.get()
                
                # Mode conversation : streaming activé par défaut
                elif stream_mode and response_format != "json":
                    logger.info(f"[WS-HYBRID {session_id}] Mode streaming conversationnel")
                    
                    session = StreamingSession(session_id, websocket)
                    active_streams[session_id] = session
                    
                    try:
                        messages = [Message(**msg) for msg in data.get("messages", [])]
                        prompt = format_messages_for_streaming(messages, "conversational")
                        
                        await inference_queue.put({"session_id": session_id, "mode": "streaming"})
                        
                        stats = await stream_llm_response(
                            prompt=prompt,
                            session=session,
                            temperature=data.get("temperature", 0.7),
                            max_tokens=data.get("max_tokens", 200)
                        )
                        
                        # Ajouter le request_id aux stats finales
                        if "request_id" in data:
                            await websocket.send_json({
                                "type": "stream_complete",
                                "request_id": data["request_id"],
                                "stats": stats
                            })
                        
                        await inference_queue.get()
                        
                        fastapi_inference_requests_total.labels(
                            model="mixtral-8x7b", 
                            status="success", 
                            mode="streaming"
                        ).inc()
                        
                    finally:
                        await session.close()
                
                # Mode classique (compatibilité)
                else:
                    logger.info(f"[WS-HYBRID {session_id}] Mode classique (pas de streaming)")
                    
                    messages = [Message(**msg) for msg in data.get("messages", [])]
                    prompt = format_messages_mistral(messages)
                    
                    await inference_queue.put({"session_id": session_id, "mode": "classic"})
                    
                    try:
                        start_time = time.time()
                        response = llm(
                            prompt,
                            max_tokens=data.get("max_tokens", 4096),
                            temperature=data.get("temperature", 0.1),
                            top_p=data.get("top_p", 0.9),
                            top_k=40,
                            stop=["</s>", "[INST]", "[/INST]"],
                            echo=False,
                            repeat_penalty=1.1,
                            stream=False
                        )
                        
                        elapsed = time.time() - start_time
                        
                        response_json = {
                            "type": "completion",
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": response['choices'][0]['text'].strip()
                                },
                                "finish_reason": response['choices'][0]['finish_reason']
                            }],
                            "usage": response['usage'],
                            "time_ms": round(elapsed * 1000),
                            "tokens_per_second": round(response['usage']['completion_tokens'] / elapsed, 2),
                            "mode": "classic"
                        }
                        
                        if "request_id" in data:
                            response_json["request_id"] = data["request_id"]
                        
                        await websocket.send_json(response_json)
                        
                        fastapi_inference_requests_total.labels(
                            model="mixtral-8x7b", 
                            status="success", 
                            mode="classic"
                        ).inc()
                        
                    finally:
                        await inference_queue.get()
                
            except Exception as e:
                logger.error(f"[WS-HYBRID {session_id}] Erreur: {str(e)}")
                logger.error(traceback.format_exc())
                
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "request_id": data.get("request_id")
                })
                
                fastapi_inference_requests_total.labels(
                    model="mixtral-8x7b", 
                    status="error", 
                    mode="unknown"
                ).inc()
    
    except WebSocketDisconnect:
        logger.info(f"[WS-HYBRID {session_id}] Client déconnecté")
    finally:
        fastapi_websocket_connections.dec()
        if session_id in active_streams:
            await active_streams[session_id].close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)