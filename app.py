"""
Qwen2.5-32B GGUF FastAPI *Proxy* / Test Server (Optimized Streaming)
=====================================================================

Objectif
--------
Version dérivée et *safe* de ton API actuelle pour exécution sur un pod de test.
On conserve les mêmes endpoints et la compatibilité client, tout en supprimant
les points de contention principaux identifiés :

1. **Suppression du "chunking manuel" O(n²)** — une seule génération llama_cpp
   par requête, streaming natif token-par-token.
2. **Pas de réinjection du texte déjà généré dans le prompt à chaque itération.**
3. **Gestion d'annulation propre** via `cancel_event` qui interrompt la lecture
   du flux (pas de nouvelle génération).
4. **SSE bien formé** (on n'envoie pas `[DONE]` avant d'avoir streamé les tokens).
5. **Verrou de modèle (sémaphore)** pour éviter d'appeler llama_cpp en parallèle
   si la build utilisée n'est pas thread-safe pour multi-génération simultanée.
   (Configurable : `MAX_PARALLEL_INFER`.)
6. **ThreadPoolExecutor réduit** (par défaut 2) → moins de contention CPU.
7. **Instrumentation Prometheus conservée** (noms identiques autant que possible).
8. **Configuration via variables d'environnement** pour faciliter les A/B tests.

⚠️ Cette version charge encore localement le modèle (comme ton code prod) afin de
pouvoir mesurer l'impact des changements de logique de streaming *à hardware égal*.
Si tu veux un pur proxy vers l'API prod sans rechargement du modèle, vois la section
« Mode Proxy Passif » en bas du fichier (désactivé par défaut).

Test rapide
-----------
1. Monte un pod de test avec le même volume /workspace/models (en lecture si possible).
2. Installe dépendances :

   ```bash
   pip install fastapi uvicorn[standard] pydantic prometheus-client psutil pynvml llama-cpp-python
   ```

3. Exporte ton token :

   ```bash
   export API_TOKEN="supersecret"
   ```

4. Lance :

   ```bash
   python qwen_proxy_fastapi.py --host 0.0.0.0 --port 8080
   ```

5. Test non-stream :

   ```bash
   curl -s -H "Authorization: Bearer $API_TOKEN" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"Bonjour"}]}' \
        http://localhost:8080/v1/chat/completions | jq
   ```

6. Test stream SSE :

   ```bash
   curl -N -H "Authorization: Bearer $API_TOKEN" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"Explique la relativité"}],"stream":true}' \
        http://localhost:8080/v1/chat/completions
   ```

Structure du code
-----------------
- Config & env
- Prometheus métriques (mêmes noms pour comparabilité)
- Modèles Pydantic (compat OpenAI-like)
- RateLimiter & ResourceManager (inchangés sauf petits durcissements)
- **OptimizedStreamManager** nouvelle implémentation (1-pass streaming)
- Chargement modèle (quasi identique; param env)
- Endpoints (compatibles; streaming réécrit)
- Main

Licence : même usage interne que ton code d'origine.
"""

from __future__ import annotations

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
from datetime import datetime, timedelta
import hashlib
from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator, Union

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Request,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response, StreamingResponse

from pydantic import BaseModel, Field

# llama_cpp import (NOTE: import error -> raise early for clarity)
try:
    from llama_cpp import Llama
except ImportError as e:  # pragma: no cover
    raise RuntimeError("llama-cpp-python non installé. `pip install llama-cpp-python`. ") from e

# Prometheus
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ---------------------------------------------------------------------------
# Configuration via environnement
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Qwen2.5-32B-Instruct-Q6_K.gguf")
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q6_K.gguf",
)

API_TOKEN = os.getenv("API_TOKEN", "supersecret")
MAX_CONCURRENT_USERS = int(os.getenv("MAX_CONCURRENT_USERS", "9"))
MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOKENS_PER_REQUEST", "150"))
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"

# Nombre max d'inférences en parallèle (sémaphore). 1 = safe. Augmente uniquement si build llama_cpp thread-safe.
MAX_PARALLEL_INFER = int(os.getenv("MAX_PARALLEL_INFER", "1"))

# Nombre de threads CPU utilisés par llama_cpp (prefill/token sampling). Ajuste selon ta machine.
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "24"))

# Contexte modèle. (NB: comment / doc peuvent annoncer 8192; on paramétrise ici.)
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "3072"))

# Batch prompt lors du prefill.
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", "256"))

# Executor threads pour déporter l'appel bloquant llama_cpp -> Python.
STREAM_EXECUTOR_WORKERS = int(os.getenv("STREAM_EXECUTOR_WORKERS", "2"))

# Debug JSON metadata
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Mode Proxy Passif (forward vers API existante au lieu de charger modèle local).
# Utilisation: export UPSTREAM_URL="https://prod-host:8000" -> les /v1/* sont proxifiés (sauf /metrics local).
UPSTREAM_URL = os.getenv("UPSTREAM_URL")  # None -> modèle local


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("qwen-proxy")


# ---------------------------------------------------------------------------
# Prometheus metrics (mêmes noms que prod pour comparaisons A/B)
# ---------------------------------------------------------------------------

system_info = Info("fastapi_system", "System information")
system_info.info(
    {
        "model": "qwen2.5-32b-q6k",
        "instance": socket.gethostname(),
        "pod_id": os.getenv("RUNPOD_POD_ID", "local"),
        "version": "proxy-optim-1.0.0",
        "max_concurrent_users": str(MAX_CONCURRENT_USERS),
    }
)

# GPU Metrics
gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilization percentage")
gpu_memory_used_bytes = Gauge("gpu_memory_used_bytes", "GPU memory used in bytes")
gpu_memory_total_bytes = Gauge("gpu_memory_total_bytes", "GPU memory total in bytes")
gpu_temperature_celsius = Gauge("gpu_temperature_celsius", "GPU temperature in Celsius")
gpu_power_watts = Gauge("gpu_power_watts", "GPU power usage in watts")
gpu_layers_offloaded = Gauge("gpu_layers_offloaded", "Number of layers offloaded to GPU")

# System Metrics
cpu_usage_percent = Gauge("cpu_usage_percent", "CPU usage percentage")
memory_used_bytes = Gauge("memory_used_bytes", "System memory used in bytes")
memory_total_bytes = Gauge("memory_total_bytes", "System memory total in bytes")
disk_usage_percent = Gauge("disk_usage_percent", "Disk usage percentage")

# Request Metrics
fastapi_requests_total = Counter(
    "fastapi_requests_total", "Total number of requests", ["method", "endpoint", "status"]
)
fastapi_request_duration_seconds = Histogram(
    "fastapi_request_duration_seconds", "Request duration in seconds", ["method", "endpoint"]
)
fastapi_websocket_connections = Gauge(
    "fastapi_websocket_connections", "Number of active WebSocket connections"
)
fastapi_concurrent_users = Gauge(
    "fastapi_concurrent_users", "Number of concurrent users"
)

# Inference Metrics
fastapi_inference_requests_total = Counter(
    "fastapi_inference_requests_total", "Total number of inference requests", ["model", "status"]
)
fastapi_inference_duration_seconds = Histogram(
    "fastapi_inference_duration_seconds", "Inference duration in seconds", ["model"],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0],
)
fastapi_inference_queue_size = Gauge(
    "fastapi_inference_queue_size", "Current inference queue size"
)
fastapi_inference_tokens_total = Counter(
    "fastapi_inference_tokens_total", "Total tokens processed", ["type"]
)
fastapi_inference_tokens_per_second = Gauge(
    "fastapi_inference_tokens_per_second", "Tokens generated per second"
)

# Model Metrics
model_loaded = Gauge("model_loaded", "Whether the model is loaded (1) or not (0)")
model_loading_duration_seconds = Gauge(
    "model_loading_duration_seconds", "Time taken to load the model"
)
model_download_progress = Gauge("model_download_progress", "Model download progress in percentage")

# JSON Parsing Metrics
json_parse_success_total = Counter("json_parse_success_total", "Number of successful JSON parses")
json_parse_failure_total = Counter("json_parse_failure_total", "Number of failed JSON parses")

# Stream Metrics
stream_cancellation_total = Counter("stream_cancellation_total", "Number of stream cancellations")
stream_cancellation_latency_seconds = Histogram(
    "stream_cancellation_latency_seconds", "Time from cancellation request to actual stop"
)

# Rate Limiting Metrics
rate_limit_exceeded_total = Counter(
    "rate_limit_exceeded_total", "Number of rate limit exceeded events", ["user_id"]
)


# ---------------------------------------------------------------------------
# Pydantic models (compat OpenAI-like)
# ---------------------------------------------------------------------------

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
    user: Optional[str] = None  # tracking utilisateur

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


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Rate limiter simple par utilisateur (sliding window 1 min)."""

    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

    def check_rate_limit(self, user_id: str) -> bool:
        if not ENABLE_RATE_LIMITING:
            return True
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        with self._lock:
            lst = self.requests.get(user_id, [])
            lst = [t for t in lst if t > minute_ago]
            if len(lst) >= self.max_requests:
                rate_limit_exceeded_total.labels(user_id=user_id).inc()
                self.requests[user_id] = lst
                return False
            lst.append(now)
            self.requests[user_id] = lst
            return True


# ---------------------------------------------------------------------------
# Resource Manager (user/accounting, unchanged semantics)
# ---------------------------------------------------------------------------

class ResourceManager:
    def __init__(self):
        self.active_users = set()
        self.user_streams: Dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def can_accept_user(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self.active_users:
                return True
            if len(self.active_users) >= MAX_CONCURRENT_USERS:
                return False
            self.active_users.add(user_id)
            self.user_streams[user_id] = set()
            fastapi_concurrent_users.set(len(self.active_users))
            return True

    def register_stream(self, user_id: str, stream_id: str):
        with self._lock:
            if user_id in self.user_streams:
                self.user_streams[user_id].add(stream_id)

    def unregister_stream(self, user_id: str, stream_id: str):
        with self._lock:
            if user_id in self.user_streams:
                self.user_streams[user_id].discard(stream_id)
                if not self.user_streams[user_id]:
                    self.active_users.discard(user_id)
                    del self.user_streams[user_id]
                    fastapi_concurrent_users.set(len(self.active_users))

    def get_user_from_stream(self, stream_id: str) -> Optional[str]:
        with self._lock:
            for uid, streams in self.user_streams.items():
                if stream_id in streams:
                    return uid
        return None


# ---------------------------------------------------------------------------
# OptimizedStreamManager (1-pass streaming, cancel-aware)
# ---------------------------------------------------------------------------

class OptimizedStreamManager:
    """Gère les streams sans re-prompting O(n²)."""

    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=STREAM_EXECUTOR_WORKERS)
        self._lock = threading.Lock()
        self.resource_manager = ResourceManager()
        self.model_semaphore = asyncio.Semaphore(MAX_PARALLEL_INFER)

    def register_stream(self, request_id: str, user_id: str) -> threading.Event:
        cancel_event = threading.Event()
        with self._lock:
            self.active_streams[request_id] = {
                "cancel_event": cancel_event,
                "start_time": time.time(),
                "tokens_generated": 0,
                "user_id": user_id,
            }
        self.resource_manager.register_stream(user_id, request_id)
        return cancel_event

    def cancel_stream(self, request_id: str) -> bool:
        with self._lock:
            if request_id in self.active_streams:
                info = self.active_streams[request_id]
                info["cancel_event"].set()
                duration = time.time() - info["start_time"]
                stream_cancellation_total.inc()
                stream_cancellation_latency_seconds.observe(duration)
                logger.info(f"[STREAM] Cancelled {request_id} after {info['tokens_generated']} tokens")
                return True
        return False

    def unregister_stream(self, request_id: str):
        user_id = None
        with self._lock:
            info = self.active_streams.pop(request_id, None)
            if info:
                user_id = info["user_id"]
        if user_id:
            self.resource_manager.unregister_stream(user_id, request_id)

    async def generate_async(
        self,
        llm_model: Llama,
        prompt: str,
        request_id: str,
        user_id: str,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream token events from a *single* llama_cpp call.

        Yields raw output dicts identical à llama_cpp(stream=True) pour compat aval.
        """
        cancel_event = self.register_stream(request_id, user_id)
        loop = asyncio.get_event_loop()
        tokens_generated = 0

        # clamp tokens
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", 200), MAX_TOKENS_PER_REQUEST)

        # Acquire model semaphore (préviens surcharge GPU / reentrance)
        queue_size = MAX_PARALLEL_INFER - self.model_semaphore._value  # approximation
        fastapi_inference_queue_size.set(max(queue_size, 0))
        await self.model_semaphore.acquire()
        fastapi_inference_queue_size.set(max(queue_size - 1, 0))

        start_t = time.time()

        try:
            # L'appel bloquant (stream=True) renvoie un itérateur Python -> exécuter côté thread
            def llama_call():
                return llm_model(prompt, stream=True, **kwargs)

            stream_iter = await loop.run_in_executor(self.executor, llama_call)

            for output in stream_iter:
                if cancel_event.is_set():
                    break
                token = output["choices"][0]["text"]
                tokens_generated += 1
                with self._lock:
                    if request_id in self.active_streams:
                        self.active_streams[request_id]["tokens_generated"] = tokens_generated
                yield output
                await asyncio.sleep(0)  # cooperative

        except Exception as e:  # pragma: no cover
            logger.error(f"[STREAM] Error in {request_id}: {e}", exc_info=True)
            raise
        finally:
            duration = time.time() - start_t
            if duration > 0 and tokens_generated > 0:
                fastapi_inference_tokens_per_second.set(tokens_generated / duration)
            self.unregister_stream(request_id)
            self.model_semaphore.release()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

llm: Optional[Llama] = None
stream_manager = OptimizedStreamManager()
rate_limiter = RateLimiter()
download_in_progress = False
_download_lock = threading.Lock()

download_complete = False


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

async def get_user_id(request: Request, token: str = Depends(verify_token)) -> str:
    # header X-User-ID > body.user > hash(token+ip)
    hdr = request.headers.get("X-User-ID")
    if hdr:
        return hdr
    try:
        body = await request.body()
        if body:
            try:
                js = json.loads(body)
                if isinstance(js, dict) and js.get("user"):
                    return str(js["user"])
            except Exception:  # ignore parse errors
                pass
    except Exception:  # pragma: no cover
        pass
    h = hashlib.md5(f"{token}:{request.client.host}".encode()).hexdigest()[:8]
    return h


# ---------------------------------------------------------------------------
# System metrics collection
# ---------------------------------------------------------------------------

def update_gpu_metrics():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization_percent.set(util.gpu)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used_bytes.set(mem.used)
        gpu_memory_total_bytes.set(mem.total)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_temperature_celsius.set(temp)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            gpu_power_watts.set(power)
        except Exception:  # pragma: no cover
            pass
    except Exception as e:  # pragma: no cover
        logger.debug(f"GPU metrics unavailable: {e}")

def update_system_metrics():
    try:
        cpu_usage_percent.set(psutil.cpu_percent(interval=0.05))
        mem = psutil.virtual_memory()
        memory_used_bytes.set(mem.used)
        memory_total_bytes.set(mem.total)
        disk = psutil.disk_usage('/')
        disk_usage_percent.set(disk.percent)
    except Exception as e:  # pragma: no cover
        logger.debug(f"System metrics unavailable: {e}")

async def metrics_update_task():  # background
    while True:
        update_gpu_metrics()
        update_system_metrics()
        await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Prompt formatter Qwen ChatML-ish
# ---------------------------------------------------------------------------

def format_messages_qwen(messages: List[Message]) -> str:
    prompt = ""
    for m in messages:
        if m.role == "system":
            prompt += f"<|im_start|>system\n{m.content}<|im_end|>\n"
        elif m.role == "user":
            prompt += f"<|im_start|>user\n{m.content}<|im_end|>\n"
        elif m.role == "assistant":
            prompt += f"<|im_start|>assistant\n{m.content}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# ---------------------------------------------------------------------------
# JSON parsing helpers (copiés / légers ajustements)
# ---------------------------------------------------------------------------

def clean_escaped_json(text: str) -> str:
    text = text.replace(r"\_", "_")
    text = text.replace("\\\\", "\\")
    text = re.sub(r"\n\s*\n", "\n", text)
    return text

def extract_json_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = text.strip()
    text = clean_escaped_json(text)
    if text.startswith("{") and text.endswith("}"):
        return text, None
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        js = m.group(1).strip()
        rem = text[:m.start()] + text[m.end():]
        return js, rem.strip() or None
    m = re.search(r"(?:JSON|json|Json):\s*({.*?})", text, re.DOTALL)
    if m:
        js = m.group(1)
        rem = text[:m.start()] + text[m.end():]
        return js, rem.strip() or None
    start = text.find("{")
    if start != -1:
        count = 0
        end = -1
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                count += 1
            elif ch == "}":
                count -= 1
                if count == 0:
                    end = i
                    break
        if end != -1:
            js = text[start:end+1]
            rem = text[:start] + text[end+1:]
            return js, rem.strip() or None
    return None, text

def smart_json_parse(text: str) -> Dict[str, Any]:
    orig = text
    js, rem = extract_json_from_text(text)
    if not js:
        logger.warning("No JSON found in response")
        json_parse_failure_total.inc()
        return {"error": "No JSON found in response", "original_response": orig}
    # try 1
    try:
        data = json.loads(js)
        json_parse_success_total.inc()
        if rem:
            data["_metadata"] = {"additional_text": rem, "json_extracted": True}
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"First parse attempt failed: {e}")
    # try 2 cleanup
    js2 = re.sub(r"//.*?$", "", js, flags=re.MULTILINE)
    js2 = re.sub(r",\s*}", "}", js2)
    js2 = re.sub(r",\s*]", "]", js2)
    try:
        data = json.loads(js2)
        json_parse_success_total.inc()
        if rem:
            data["_metadata"] = {"additional_text": rem, "json_extracted": True, "required_cleanup": True}
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"Second parse attempt failed: {e}")
    # try 3 heavy
    js3 = re.sub(r"(\w+):", r'"\1":', js2)
    try:
        data = json.loads(js3)
        json_parse_success_total.inc()
        if rem:
            data["_metadata"] = {"additional_text": rem, "json_extracted": True, "heavy_cleanup": True}
        return data
    except json.JSONDecodeError as e:
        logger.error(f"All parse attempts failed: {e}")
        json_parse_failure_total.inc()
        return {
            "error": "JSON parsing failed after all attempts",
            "original_response": orig,
            "attempted_json": js3,
            "parse_error": str(e),
        }

def ensure_json_response(text: str, request_format: Optional[Dict] = None) -> str:
    if request_format and request_format.get("type") == "json_object":
        parsed = smart_json_parse(text)
        if "_metadata" in parsed and not DEBUG_MODE:
            del parsed["_metadata"]
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return text


# ---------------------------------------------------------------------------
# Model download helpers (copiés / simplifiés)
# ---------------------------------------------------------------------------

def cleanup_old_models():
    models_dir = os.path.dirname(MODEL_PATH) or "."
    print("\n============================================================")
    print("🧹 CLEANING OLD MODELS")
    print("============================================================")
    try:
        import glob
        old = glob.glob(os.path.join(models_dir, "*.gguf"))
        if not old:
            print("✅ No old models found")
        else:
            total_size = sum(os.path.getsize(f) for f in old)
            print(f"📊 Space used by old models: {total_size/(1024**3):.1f} GB")
            target = os.path.basename(MODEL_PATH)
            for f in old:
                if os.path.basename(f) != target:
                    print(f"🗑️  Deleting: {os.path.basename(f)}")
                    try:
                        os.remove(f)
                        print("   ✅ Deleted")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
        import shutil
        stat = shutil.disk_usage(models_dir)
        free_gb = stat.free / (1024**3)
        print(f"\n💾 Free space after cleanup: {free_gb:.1f} GB")
        if free_gb < 30:
            print("⚠️  WARNING: low free space vs ~25GB model")
        else:
            print("✅ Sufficient space for new model")
    except Exception as e:  # pragma: no cover
        print(f"❌ Cleanup error: {e}")
    print("============================================================\n")

def download_with_retry(url: str, dest: str, max_retries: int = 3, delay: int = 5) -> bool:
    import urllib.error
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("⚠️  WARNING: No HuggingFace token found! You may hit 429 errors.")
    # try huggingface-cli
    if hf_token and subprocess.run(["which", "huggingface-cli"], capture_output=True).returncode == 0:
        print("🔑 HuggingFace token detected, using huggingface-cli...")
        try:
            subprocess.run(["huggingface-cli", "login", "--token", hf_token], capture_output=True, check=True)
            hf_cmd = [
                "huggingface-cli",
                "download",
                "bartowski/Qwen2.5-32B-Instruct-GGUF",
                "Qwen2.5-32B-Instruct-Q6_K.gguf",
                "--local-dir",
                os.path.dirname(dest),
                "--local-dir-use-symlinks",
                "False",
            ]
            print("📥 Downloading with huggingface-cli...")
            result = subprocess.run(hf_cmd, text=True)
            if result.returncode == 0:
                print("✅ Download successful!")
                return True
        except Exception as e:  # pragma: no cover
            print(f"⚠️ huggingface-cli failed: {e}")
    # fallback direct
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = delay * (2 ** attempt)
                print(f"\n⏳ Waiting {wait}s before retry...")
                time.sleep(wait)
            print(f"\n📥 Attempt {attempt + 1}/{max_retries}")
            headers = {"User-Agent": "Mozilla/5.0"}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            request = urllib.request.Request(url, headers=headers)
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                mb_d = downloaded / 1024 / 1024
                mb_t = total_size / 1024 / 1024
                model_download_progress.set(percent)
                sys.stdout.write(f"\rDownload: {percent:.1f}% ({mb_d:.1f}/{mb_t:.1f} MB) ")
                sys.stdout.flush()
            urllib.request.urlretrieve(request.full_url, dest, reporthook=progress)
            print("\n✅ Download complete!")
            return True
        except urllib.error.HTTPError as e:  # pragma: no cover
            if e.code == 429:
                print("\n⚠️ Error 429: Rate limited")
                if not hf_token:
                    print("💡 Tip: set HF_TOKEN")
                continue
            else:
                print(f"\n❌ HTTP Error {e.code}: {e.reason}")
    return False

def download_model_if_needed():
    global download_in_progress, download_complete
    with _download_lock:
        if download_complete:
            return
        cleanup_old_models()
        if os.path.exists(MODEL_PATH):
            fsz = os.path.getsize(MODEL_PATH)
            expected = 25_000_000_000  # ~25GB
            if fsz > expected * 0.95:
                print(f"✅ Model found: {fsz/(1024**3):.1f} GB")
                download_complete = True
                return
            else:
                print(f"⚠️ Incomplete model ({fsz/(1024**3):.1f} GB), redownload...")
        if download_in_progress:
            print("⏳ Download already in progress...")
            return
        download_in_progress = True
    # hors verrous lourds
    print("\n============================================================")
    print("📥 DOWNLOADING QWEN2.5-32B Q6_K")
    print("============================================================")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        st = time.time()
        ok = download_with_retry(MODEL_URL, MODEL_PATH)
        if not ok:
            raise RuntimeError("Download failed after retries")
        dl = time.time() - st
        print(f"\n✅ Download completed in {dl/60:.1f} minutes")
        fsz = os.path.getsize(MODEL_PATH)
        print(f"📦 File size: {fsz/(1024**3):.1f} GB")
        if fsz < 24_000_000_000:
            raise RuntimeError("File too small, corrupt download")
        model_download_progress.set(100)
        download_complete = True
    finally:
        download_in_progress = False


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def load_model() -> None:
    global llm
    if UPSTREAM_URL:
        logger.warning("UPSTREAM_URL set -> skipping local model load; running in proxy-pass mode.")
        model_loaded.set(0)
        return

    download_model_if_needed()

    print("\n============================================================")
    print("🚀 LOADING QWEN2.5-32B Q6_K")
    print("============================================================")

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        vram_free_gb = mem_info.free / (1024**3)
        print("📊 GPU Detection:")
        print(f"   Total VRAM: {vram_gb:.1f} GB")
        print(f"   Free VRAM: {vram_free_gb:.1f} GB")
        n_gpu_layers = -1  # all layers
        print("✅ Configuration: ALL layers on GPU (forced)")
    except Exception as e:  # pragma: no cover
        print(f"⚠️ GPU detection failed: {e}")
        print("✅ Forcing GPU usage anyway...")
        n_gpu_layers = -1

    print("\n📋 Model Configuration:")
    print("   Model: Qwen2.5-32B Q6_K")
    print(f"   Context: {LLM_N_CTX} tokens")
    print("   GPU Layers: ALL")
    print(f"   Batch Size: {LLM_N_BATCH}")
    print(f"   Max Users: {MAX_CONCURRENT_USERS}")

    st = time.time()
    llm_local = Llama(
        model_path=MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=n_gpu_layers,
        n_batch=LLM_N_BATCH,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
        seed=42,
        rope_freq_scale=1.0,
        f16_kv=True,
        logits_all=False,
        vocab_only=False,
        embedding=False,
        low_vram=False,
        n_threads_batch=2,
        tensor_split=None,
        main_gpu=0,
    )
    load_time = time.time() - st
    print(f"\n✅ Model loaded in {load_time:.1f} seconds")
    model_loaded.set(1)
    model_loading_duration_seconds.set(load_time)
    gpu_layers_offloaded.set(-1)

    # mini perf test
    print("\n🧪 Performance test...")
    t_st = time.time()
    test_prompt = "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"
    res = llm_local(test_prompt, max_tokens=10, temperature=0.1)
    t_dur = time.time() - t_st
    tps = 10 / t_dur if t_dur > 0 else 0
    print(f"⏱️ Time for 10 tokens: {t_dur:.2f}s")
    print(f"📊 Speed: {tps:.1f} tokens/second")
    if t_dur > 1:
        print("⚠️ Performance seems limited, check GPU usage...")
    else:
        print("✅ Excellent GPU performance!")
    print("\n============================================================")
    print("✅ QWEN2.5-32B Q6_K READY FOR TEST")
    print("============================================================\n")

    llm = llm_local


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
def lifespan(app: FastAPI):
    print("=== Starting FastAPI Proxy Application ===")
    # background metrics
    metrics_task = asyncio.create_task(metrics_update_task())
    try:
        load_model()
        print("=== Model (or proxy) ready ===")
    except Exception as e:  # pragma: no cover
        print(f"Fatal error loading model: {e}")
        model_loaded.set(0)
    yield
    print("=== Shutting down ===")
    model_loaded.set(0)
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    # close executor
    stream_manager.executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Qwen2.5-32B Q6_K Proxy/Test API",
    version="proxy-optim-1.0.0",
    description="API de test optimisée streaming (no O(n²) chunking)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Proxy helper (only used if UPSTREAM_URL set)
# ---------------------------------------------------------------------------

if UPSTREAM_URL:
    import httpx
    _httpx_client = httpx.AsyncClient(timeout=None)

    async def proxy_upstream(method: str, path: str, **kwargs):
        url = f"{UPSTREAM_URL}{path}"
        r = await _httpx_client.request(method, url, **kwargs)
        return r


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def metrics():
    update_gpu_metrics()
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    return {
        "message": "Qwen2.5-32B Q6_K Proxy/Test API",
        "status": "running" if (llm is not None or UPSTREAM_URL) else "loading",
        "model": {
            "name": os.path.basename(MODEL_PATH),
            "loaded": llm is not None,
            "size": "~25GB",
            "context": f"{LLM_N_CTX} tokens",
            "gpu_layers": "all",
        },
        "system": {
            "max_concurrent_users": MAX_CONCURRENT_USERS,
            "active_users": len(stream_manager.resource_manager.active_users),
            "active_streams": len(stream_manager.active_streams),
            "rate_limiting": ENABLE_RATE_LIMITING,
            "upstream_mode": bool(UPSTREAM_URL),
        },
        "endpoints": {
            "/v1/chat/completions": "OpenAI-compatible chat endpoint",
            "/ws": "WebSocket streaming with interruption",
            "/v1/summary": "Extract structured summary",
            "/v1/models": "List available models",
            "/health": "Health check with detailed status",
            "/metrics": "Prometheus metrics",
        },
    }

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy" if (llm is not None or UPSTREAM_URL) else "loading",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": llm is not None,
            "path": MODEL_PATH,
            "exists": os.path.exists(MODEL_PATH),
            "size": f"{os.path.getsize(MODEL_PATH)/(1024**3):.1f} GB" if os.path.exists(MODEL_PATH) else None,
            "upstream_mode": bool(UPSTREAM_URL),
        },
        "system": {
            "active_users": len(stream_manager.resource_manager.active_users),
            "max_users": MAX_CONCURRENT_USERS,
            "active_streams": len(stream_manager.active_streams),
            "active_stream_ids": list(stream_manager.active_streams.keys())[:5],
        },
        "performance": {
            "json_parse_success": json_parse_success_total._value.get() if hasattr(json_parse_success_total, "_value") else None,
            "json_parse_failure": json_parse_failure_total._value.get() if hasattr(json_parse_failure_total, "_value") else None,
            "stream_cancellations": stream_cancellation_total._value.get() if hasattr(stream_cancellation_total, "_value") else None,
        },
    }
    if len(stream_manager.resource_manager.active_users) >= MAX_CONCURRENT_USERS:
        health_status["status"] = "overloaded"
        health_status["message"] = "Maximum concurrent users reached"
    return health_status

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    start_time = time.time()
    status = "success"
    try:
        if UPSTREAM_URL:  # proxy mode
            import httpx
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            r = await proxy_upstream("GET", "/v1/models", headers=headers)
            fastapi_requests_total.labels(method="GET", endpoint="/v1/models", status="proxy").inc()
            return r.json()
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
                    "context_length": LLM_N_CTX,
                    "quantization": "Q6_K",
                }
            ],
        }
        return result
    except Exception:  # pragma: no cover
        status = "error"
        raise
    finally:
        fastapi_requests_total.labels(method="GET", endpoint="/v1/models", status=status).inc()
        fastapi_request_duration_seconds.labels(method="GET", endpoint="/v1/models").observe(time.time() - start_time)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

STOP_SEQS_DEFAULT = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]

async def _run_llm_nonstream(prompt: str, **gen_kwargs) -> Dict[str, Any]:
    if llm is None and not UPSTREAM_URL:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if UPSTREAM_URL:  # proxy
        import httpx
        headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": gen_kwargs.pop("model_name", "qwen2.5-32b-q6k"),
            "messages": gen_kwargs.pop("messages_payload", []),
            "temperature": gen_kwargs.get("temperature", 0.1),
            "max_tokens": gen_kwargs.get("max_tokens", 200),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "top_k": gen_kwargs.get("top_k", 40),
            "stream": False,
        }
        r = await proxy_upstream("POST", "/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()

    # local path
    # Acquire semaphore (same que streaming)
    queue_size = MAX_PARALLEL_INFER - stream_manager.model_semaphore._value
    fastapi_inference_queue_size.set(max(queue_size, 0))
    await stream_manager.model_semaphore.acquire()
    fastapi_inference_queue_size.set(max(queue_size - 1, 0))

    try:
        out = llm(prompt, **gen_kwargs)
        return out
    finally:
        stream_manager.model_semaphore.release()


# ---------------------------------------------------------------------------
# /v1/chat/completions (POST)
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=Union[ChatCompletionResponse, None])
async def chat_completions(
    request: ChatCompletionRequest = Body(...),
    user_id: str = Depends(get_user_id),
):
    start_time = time.time()
    status = "success"

    # rate limit
    if not rate_limiter.check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # ressources
    if not stream_manager.resource_manager.can_accept_user(user_id):
        raise HTTPException(status_code=503, detail=f"Server at capacity ({MAX_CONCURRENT_USERS} concurrent users)")

    # Format prompt
    prompt = format_messages_qwen(request.messages)

    logger.info(f"[CHAT] User {user_id}: {len(request.messages)} messages | stream={request.stream}")

    # streaming path
    if request.stream:
        resp = await handle_streaming_response(prompt, request, user_id, start_time)
        fastapi_requests_total.labels(method="POST", endpoint="/v1/chat/completions", status=status).inc()
        fastapi_request_duration_seconds.labels(method="POST", endpoint="/v1/chat/completions").observe(time.time() - start_time)
        return resp

    # non-stream path
    try:
        gen_kwargs = dict(
            max_tokens=min(request.max_tokens or 200, MAX_TOKENS_PER_REQUEST),
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 40,
            stop=request.stop or STOP_SEQS_DEFAULT,
            echo=False,
            repeat_penalty=1.1,
            model_name=request.model,
            messages_payload=[m.dict() for m in request.messages],
        )
        inf_st = time.time()
        response = await _run_llm_nonstream(prompt, **gen_kwargs)
        inf_dur = time.time() - inf_st
        fastapi_inference_duration_seconds.labels(model="qwen2.5-32b-q6k").observe(inf_dur)

        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        fastapi_inference_tokens_total.labels(type="prompt").inc(prompt_tokens)
        fastapi_inference_tokens_total.labels(type="completion").inc(completion_tokens)
        if inf_dur > 0:
            fastapi_inference_tokens_per_second.set(completion_tokens / inf_dur)
            logger.info(f"[PERF] User {user_id}: {completion_tokens/inf_dur:.1f} tps")

        generated_text = response['choices'][0]['text'].strip()
        if request.response_format and request.response_format.get("type") == "json_object":
            generated_text = ensure_json_response(generated_text, request.response_format)

        chat_resp = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=generated_text),
                    finish_reason=response['choices'][0]['finish_reason'],
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=response['usage']['total_tokens'],
            ),
        )

        fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="success").inc()
        return chat_resp
    except HTTPException:
        status = "error"
        fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="error").inc()
        raise
    except Exception as e:  # pragma: no cover
        status = "error"
        fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="error").inc()
        logger.error(f"Generation error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        fastapi_requests_total.labels(method="POST", endpoint="/v1/chat/completions", status=status).inc()
        fastapi_request_duration_seconds.labels(method="POST", endpoint="/v1/chat/completions").observe(time.time() - start_time)


# ---------------------------------------------------------------------------
# Streaming SSE helper
# ---------------------------------------------------------------------------

async def handle_streaming_response(prompt: str, request: ChatCompletionRequest, user_id: str, start_time: float):
    request_id = f"stream_{uuid.uuid4().hex[:8]}"

    async def generate():
        tokens_count = 0
        try:
            # event start (optionnel)
            yield b"event: start\n\n"
            async for output in stream_manager.generate_async(
                llm if not UPSTREAM_URL else None,  # ignored in proxy mode (see below)
                prompt,
                request_id,
                user_id,
                max_tokens=min(request.max_tokens or 200, MAX_TOKENS_PER_REQUEST),
                temperature=request.temperature or 0.1,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 40,
                stop=STOP_SEQS_DEFAULT,
            ):
                # Si en mode proxy, on ne devrait jamais arriver ici (voir plus bas)
                token = output['choices'][0]['text']
                tokens_count += 1
                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[StreamChoice(index=0, delta=StreamDelta(content=token))],
                )
                yield f"data: {chunk.json()}\n\n".encode("utf-8")

        except Exception as e:  # pragma: no cover
            logger.error(f"Streaming error for {user_id}: {e}")
            fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="error").inc()
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode("utf-8")
        finally:
            # Final chunk stop
            final_chunk = ChatCompletionChunk(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")],
            )
            yield f"data: {final_chunk.json()}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
            duration = time.time() - start_time
            if duration > 0 and tokens_count > 0:
                fastapi_inference_tokens_per_second.set(tokens_count / duration)
            fastapi_inference_requests_total.labels(model="qwen2.5-32b-q6k", status="success").inc()

    # Mode proxy streaming (si UPSTREAM_URL) -> on forward brut.
    if UPSTREAM_URL:
        import httpx
        headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": request.model,
            "messages": [m.dict() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stream": True,
        }
        async def proxy_stream():
            tokens_count = 0
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{UPSTREAM_URL}/v1/chat/completions", headers=headers, json=payload) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            # Pass-through
                            yield (line + "\n\n").encode("utf-8")
                            if line.strip() == "data: [DONE]":
                                break
                            tokens_count += 1
            duration = time.time() - start_time
            if duration > 0 and tokens_count > 0:
                fastapi_inference_tokens_per_second.set(tokens_count / duration)
        return StreamingResponse(proxy_stream(), media_type="text/event-stream")

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /v1/summary (POST)
# ---------------------------------------------------------------------------

@app.post("/v1/summary", dependencies=[Depends(verify_token)])
async def create_summary(request: dict, user_id: str = Depends(get_user_id)):
    if llm is None and not UPSTREAM_URL:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not rate_limiter.check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    try:
        messages = request.get("messages", [])
        validated_messages = [Message(**msg) for msg in messages]
        prompt = format_messages_qwen(validated_messages)
        logger.info(f"[SUMMARY] User {user_id}: {len(validated_messages)} messages")
        if UPSTREAM_URL:
            import httpx
            headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": "qwen2.5-32b",
                "messages": [m.dict() for m in validated_messages],
                "temperature": 0.1,
                "max_tokens": 300,
                "stream": False,
            }
            r = await proxy_upstream("POST", "/v1/chat/completions", headers=headers, json=payload)
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)
            response = r.json()
        else:
            response = await _run_llm_nonstream(
                prompt,
                max_tokens=300,
                temperature=0.1,
                top_p=0.9,
                stop=STOP_SEQS_DEFAULT,
                model_name="qwen2.5-32b",
                messages_payload=[m.dict() for m in validated_messages],
            )
        result_text = response['choices'][0]['text'].strip()
        parsed_result = smart_json_parse(result_text)
        return {"status": "success", "extraction": parsed_result, "usage": response['usage']}
    except Exception as e:  # pragma: no cover
        logger.error(f"Summary error for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /v1/warmup (POST)
# ---------------------------------------------------------------------------

@app.post("/v1/warmup", dependencies=[Depends(verify_token)])
async def warmup():
    if llm is None and not UPSTREAM_URL:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        warmup_prompt = "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"
        if UPSTREAM_URL:
            import httpx
            headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": "qwen2.5-32b",
                "messages": [{"role": "user", "content": "Bonjour"}],
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": False,
            }
            st = time.time()
            r = await proxy_upstream("POST", "/v1/chat/completions", headers=headers, json=payload)
            dur = time.time() - st
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)
            response = r.json()
            txt = response['choices'][0]['text'].strip()
            return {"status": "success", "warmup_time": f"{dur:.2f}s", "model_ready": True, "response": txt}
        st = time.time()
        response = await _run_llm_nonstream(
            warmup_prompt,
            max_tokens=10,
            temperature=0.1,
            model_name="qwen2.5-32b",
            messages_payload=[{"role": "user", "content": "Bonjour"}],
        )
        dur = time.time() - st
        return {
            "status": "success",
            "warmup_time": f"{dur:.2f}s",
            "model_ready": True,
            "response": response['choices'][0]['text'].strip(),
        }
    except Exception as e:  # pragma: no cover
        logger.error(f"Warmup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# WebSocket /ws
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    user_id = websocket.headers.get('X-User-ID', hashlib.md5(f"{token}:{websocket.client.host}".encode()).hexdigest()[:8])
    if not stream_manager.resource_manager.can_accept_user(user_id):
        await websocket.close(code=1008, reason="Server at capacity")
        return
    await websocket.accept()
    fastapi_websocket_connections.inc()

    welcome_msg = {
        "type": "connection",
        "status": "connected",
        "model": "qwen2.5-32b-q6k",
        "model_loaded": bool(llm or UPSTREAM_URL),
        "user_id": user_id,
        "capabilities": [
            "French medical conversations",
            "Streaming responses",
            "Async stream cancellation",
            "Multi-user support",
        ],
    }
    await websocket.send_json(welcome_msg)

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "cancel_stream":
                rid = data.get("request_id")
                if rid:
                    cancelled = stream_manager.cancel_stream(rid)
                    await websocket.send_json({"type": "stream_cancelled", "request_id": rid, "success": cancelled})
                    logger.info(f"[WS] User {user_id} cancelled stream {rid}")
                continue
            if llm is None and not UPSTREAM_URL:
                await websocket.send_json({"type": "error", "error": "Model not loaded"})
                continue
            if not rate_limiter.check_rate_limit(user_id):
                await websocket.send_json({"type": "error", "error": "Rate limit exceeded"})
                continue
            try:
                messages = [Message(**msg) for msg in data.get("messages", [])]
                rid = data.get("request_id") or f"ws_{uuid.uuid4().hex[:8]}"
                logger.info(f"[WS] User {user_id} stream {rid}: {len(messages)} messages")
                prompt = format_messages_qwen(messages)
                await websocket.send_json({"type": "stream_start", "request_id": rid})

                # streaming
                full_response = ""
                tokens_count = 0
                start_ts = time.time()
                cancelled = False

                if UPSTREAM_URL:
                    # Proxy mode: forward
                    import httpx
                    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
                    payload = {
                        "model": data.get("model", "qwen2.5-32b"),
                        "messages": [m.dict() for m in messages],
                        "temperature": data.get("temperature", 0.1),
                        "max_tokens": min(data.get("max_tokens", 200), MAX_TOKENS_PER_REQUEST),
                        "top_p": data.get("top_p", 0.9),
                        "top_k": data.get("top_k", 40),
                        "stream": True,
                    }
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream("POST", f"{UPSTREAM_URL}/v1/chat/completions", headers=headers, json=payload) as r:
                            async for line in r.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith("data: "):
                                    if line.strip() == "data: [DONE]":
                                        break
                                    # naive parse to get token? we'll just forward
                                    await websocket.send_json({"type": "stream_token", "token": line, "request_id": rid})
                                    tokens_count += 1
                                    full_response += line
                else:
                    async for output in stream_manager.generate_async(
                        llm,
                        prompt,
                        rid,
                        user_id,
                        max_tokens=min(data.get("max_tokens", 200), MAX_TOKENS_PER_REQUEST),
                        temperature=data.get("temperature", 0.1),
                        top_p=data.get("top_p", 0.9),
                        top_k=data.get("top_k", 40),
                        stop=STOP_SEQS_DEFAULT,
                    ):
                        token = output['choices'][0]['text']
                        full_response += token
                        tokens_count += 1
                        if websocket.client_state.value != 1:  # disconnected
                            cancelled = True
                            break
                        try:
                            await websocket.send_json({"type": "stream_token", "token": token, "request_id": rid})
                        except Exception as e:  # pragma: no cover
                            logger.info(f"[WS] Send error for {user_id}: {e}")
                            cancelled = True
                            break

                duration = time.time() - start_ts
                tps = tokens_count / duration if duration > 0 else 0
                if websocket.client_state.value == 1:
                    try:
                        await websocket.send_json(
                            {
                                "type": "stream_end",
                                "full_response": full_response,
                                "tokens": tokens_count,
                                "request_id": rid,
                                "cancelled": cancelled,
                                "duration": duration,
                                "tokens_per_second": tps,
                            }
                        )
                    except Exception:  # pragma: no cover
                        pass
                logger.info(
                    f"[WS] User {user_id} stream {rid}: {tokens_count} tokens in {duration:.2f}s ({tps:.1f} tps, cancelled={cancelled})"
                )
                fastapi_inference_requests_total.labels(
                    model="qwen2.5-32b-q6k", status="cancelled" if cancelled else "success"
                ).inc()
            except Exception as e:  # pragma: no cover
                logger.error(f"[WS] Error for {user_id}: {e}", exc_info=True)
                if websocket.client_state.value == 1:
                    try:
                        await websocket.send_json({"type": "error", "error": str(e), "request_id": data.get("request_id")})
                    except Exception:
                        pass
    except WebSocketDisconnect:
        logger.info(f"[WS] User {user_id} disconnected")
    finally:
        fastapi_websocket_connections.dec()
        stream_manager.resource_manager.active_users.discard(user_id)
        fastapi_concurrent_users.set(len(stream_manager.resource_manager.active_users))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Qwen Proxy/Test API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    import uvicorn
    uvicorn.run(
        "qwen_proxy_fastapi:app",  # ce fichier
        host=args.host,
        port=args.port,
        workers=1,  # 1 process -> partage unique modèle
        log_level=args.log_level,
        reload=False,
    )
