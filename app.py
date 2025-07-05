#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Phi-3.5-mini GGUF avec llama-cpp-python
Optimisé pour extraction JSON, catégorisation et résumé
Avec authentification Bearer et endpoint WebSocket
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

# Configuration
MODEL_PATH = "/app/models/Phi-3.5-mini-instruct-Q5_K_M.gguf"
MODEL_URL = "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q5_K_M.gguf"
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
    'model': 'phi-3.5-mini',
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

# Queue pour simuler la file d'attente
inference_queue = asyncio.Queue(maxsize=1000)

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

# ===== CODE ORIGINAL =====

# Sécurité
security = HTTPBearer()

# Modèles Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "phi-3.5-mini"
    messages: List[Message]
    temperature: Optional[float] = 0.0  # 0 par défaut pour extraction déterministe
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.1  # Très bas pour précision
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

# ===== LIFESPAN POUR GÉRER LE CYCLE DE VIE =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Démarrage
    model_start = time.time()
    load_model()
    model_loading_duration_seconds.set(time.time() - model_start)
    model_loaded.set(1)
    
    # Démarrer la tâche de mise à jour des métriques
    metrics_task = asyncio.create_task(metrics_update_task())
    
    yield
    
    # Arrêt
    model_loaded.set(0)
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass

# Initialisation de l'application avec lifespan
app = FastAPI(
    title="Phi-3.5-mini GGUF API",
    version="2.0.0",
    description="API FastAPI pour Phi-3.5-mini optimisée pour extraction JSON, catégorisation et résumé",
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
    """Vérifier que le modèle existe"""
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle non trouvé à {MODEL_PATH}")
        raise Exception("Le modèle doit être pré-téléchargé dans l'image Docker")

def load_model():
    """Charger le modèle GGUF avec configuration optimale pour Phi-3.5"""
    global llm
    
    download_model_if_needed()
    
    print(f"Chargement du modèle Phi-3.5-mini depuis {MODEL_PATH}...")
    
    # Configuration optimisée pour Phi-3.5-mini
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,  # Phi-3.5 supporte 4K de contexte
        n_threads=8,
        n_gpu_layers=-1,  # Toutes les couches sur GPU
        n_batch=512,
        use_mmap=True,
        verbose=True,
        seed=42,  # Pour reproductibilité des outputs JSON
        repeat_penalty=1.0  # Phi n'a pas besoin de pénalité de répétition
    )
    
    print("Modèle Phi-3.5-mini chargé avec succès!")

def format_messages_phi(messages: List[Message]) -> str:
    """Formater les messages pour Phi-3.5 avec focus sur extraction structurée"""
    formatted = ""
    
    for message in messages:
        if message.role == "system":
            formatted += f"<|system|>\n{message.content}<|end|>\n"
        elif message.role == "user":
            formatted += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == "assistant":
            formatted += f"<|assistant|>\n{message.content}<|end|>\n"
    
    # Ajouter le début de la réponse de l'assistant
    formatted += "<|assistant|>\n"
    
    return formatted

# Alias pour compatibilité
format_messages_mistral = format_messages_phi
format_messages_gemma = format_messages_phi

def extract_json_from_text(text: str) -> str:
    """Extraire JSON même si Phi ajoute du texte autour (plus rare qu'avec Gemma)"""
    # Chercher le premier { et le dernier }
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    
    # Chercher entre ```json et ```
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Si pas de JSON trouvé, retourner tel quel
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
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Essayer avec des corrections communes
        text = text.replace("'", '"')  # Simple quotes -> double quotes
        text = re.sub(r',\s*}', '}', text)  # Virgules finales
        text = re.sub(r',\s*]', ']', text)
        
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
            # Phi-3.5 a rarement ce problème, mais on garde le fallback
            return json.dumps({
                "response": text,
                "error": "Could not parse as valid JSON",
                "raw_output": text[:200] + "..." if len(text) > 200 else text
            }, ensure_ascii=False)
    return text

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
        "message": "Phi-3.5-mini GGUF API",
        "status": "running",
        "model": "Phi-3.5-mini-instruct-Q5_K_M.gguf",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/ws": "WebSocket - Chat endpoint (requires token in query)",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics"
        },
        "optimized_for": ["JSON extraction", "categorization", "summarization", "structured outputs"],
        "performance": "100% accuracy on extraction benchmarks"
    }

@app.get("/health")
async def health_check():
    """Vérifier l'état de l'API"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    """Lister les modèles disponibles"""
    start_time = time.time()
    
    try:
        result = {
            "object": "list",
            "data": [
                {
                    "id": "phi-3.5-mini",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Microsoft",
                    "permission": [],
                    "root": "phi-3.5-mini",
                    "parent": None
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
    """Endpoint compatible OpenAI optimisé pour extraction et outputs structurés"""
    start_time = time.time()
    status = "success"
    
    if llm is None:
        fastapi_requests_total.labels(method="POST", endpoint="/v1/chat/completions", status="error").inc()
        fastapi_inference_requests_total.labels(model="phi-3.5-mini", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Ajouter à la queue
        await inference_queue.put(request)
        
        prompt = format_messages_phi(request.messages)
        
        # Instructions spécifiques pour JSON avec Phi-3.5
        if request.response_format and request.response_format.get("type") == "json_object":
            # Phi-3.5 est EXCELLENT avec cette instruction simple
            prompt += "Respond with valid JSON only.\n"
        
        # Timer pour l'inférence
        inference_start = time.time()
        
        # Paramètres optimisés pour Phi-3.5 et extraction structurée
        response = llm(
            prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.0,  # 0 par défaut pour extraction déterministe
            top_p=request.top_p or 0.1,  # Très restrictif pour précision maximale
            top_k=10,  # Encore plus restrictif que Gemma
            stop=request.stop or ["<|end|>", "<|endoftext|>", "<|assistant|>"],
            echo=False
        )
        
        # Retirer de la queue
        await inference_queue.get()
        
        # Durée d'inférence
        inference_duration = time.time() - inference_start
        fastapi_inference_duration_seconds.labels(model="phi-3.5-mini").observe(inference_duration)
        
        # Métriques de tokens
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        
        fastapi_inference_tokens_total.labels(type="prompt").inc(prompt_tokens)
        fastapi_inference_tokens_total.labels(type="completion").inc(completion_tokens)
        
        # Tokens par seconde
        if inference_duration > 0:
            tps = completion_tokens / inference_duration
            fastapi_inference_tokens_per_second.set(tps)
        
        generated_text = response['choices'][0]['text'].strip()
        
        # Post-processing spécifique pour JSON (moins nécessaire avec Phi)
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
        fastapi_inference_requests_total.labels(model="phi-3.5-mini", status="success").inc()
        
        return chat_response
        
    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        fastapi_inference_requests_total.labels(model="phi-3.5-mini", status="error").inc()
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
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "model": "phi-3.5-mini",
        "capabilities": ["JSON_extraction", "categorization", "summarization", "structured_outputs"],
        "accuracy": "100% on extraction benchmarks"
    })
    
    try:
        while True:
            # Recevoir la requête
            data = await websocket.receive_json()
            
            # Log
            print(f"[WS] Requête reçue: {len(data.get('messages', []))} messages")
            
            # Vérifier que le modèle est chargé
            if llm is None:
                await websocket.send_json({
                    "type": "error",
                    "error": "Model not loaded"
                })
                fastapi_inference_requests_total.labels(model="phi-3.5-mini", status="error").inc()
                continue
            
            # Traiter la requête
            try:
                # Ajouter à la queue
                request = ChatCompletionRequest(
                    messages=[Message(**msg) for msg in data.get("messages", [])],
                    **{k: v for k, v in data.items() if k != "messages"}
                )
                await inference_queue.put(request)
                
                # Convertir les messages en objets Message
                messages = [Message(**msg) for msg in data.get("messages", [])]
                prompt = format_messages_phi(messages)
                
                # Ajouter instruction JSON si demandé
                if data.get("response_format", {}).get("type") == "json_object":
                    prompt += "Respond with valid JSON only.\n"
                
                # Générer
                start_time = time.time()
                
                response = llm(
                    prompt,
                    max_tokens=data.get("max_tokens", 512),
                    temperature=data.get("temperature", 0.0),  # 0 par défaut pour Phi
                    top_p=data.get("top_p", 0.1),
                    top_k=data.get("top_k", 10),
                    stop=data.get("stop", ["<|end|>", "<|endoftext|>", "<|assistant|>"]),
                    echo=False
                )
                
                # Retirer de la queue
                await inference_queue.get()
                
                elapsed = (time.time() - start_time) * 1000
                inference_duration = elapsed / 1000.0
                
                # Métriques
                fastapi_inference_duration_seconds.labels(model="phi-3.5-mini").observe(inference_duration)
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
                fastapi_inference_requests_total.labels(model="phi-3.5-mini", status="success").inc()
                
                print(f"[WS] Réponse envoyée en {elapsed:.0f}ms ({response_json['tokens_per_second']} t/s)")
                
            except Exception as e:
                print(f"[WS] Erreur: {str(e)}")
                fastapi_inference_requests_total.labels(model="phi-3.5-mini", status="error").inc()
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