#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Qwen2.5-14B GGUF avec llama-cpp-python
Avec authentification Bearer et format de réponse Claude-like
+ ENDPOINT WEBSOCKET
"""
import os
import time
import uuid
import json
import re
import logging
from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from llama_cpp import Llama

# Configuration - CHANGÉ ICI
MODEL_PATH = "/app/models/qwen2.5-14b-instruct-q4_k_m.gguf"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf"
API_TOKEN = os.getenv("API_TOKEN", "supersecret")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sécurité
security = HTTPBearer()

# Modèles Pydantic (identiques)
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-14b-gguf"  # CHANGÉ ICI
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.95
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

# Initialisation de l'application
app = FastAPI(
    title="Qwen2.5-14B GGUF API",  # CHANGÉ ICI
    version="1.0.0",
    description="API FastAPI pour Qwen2.5-14B avec llama-cpp-python et authentification"
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
    """Télécharger le modèle s'il n'existe pas"""
    # Note: Dans le Dockerfile, on télécharge déjà le modèle
    # Cette fonction est gardée pour compatibilité
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle non trouvé à {MODEL_PATH}")
        raise Exception("Le modèle doit être pré-téléchargé dans l'image Docker")

def load_model():
    """Charger le modèle GGUF"""
    global llm
    
    download_model_if_needed()
    
    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    
    # Configuration adaptée pour Qwen2.5-14B
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,  # Qwen supporte un contexte plus long
        n_threads=8,
        n_gpu_layers=-1,  # Toutes les couches sur GPU
        verbose=True
    )
    
    print("Modèle chargé avec succès!")

def format_messages_qwen(messages: List[Message]) -> str:
    """Formater les messages pour Qwen2.5 (format ChatML)"""
    formatted = ""
    
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

# Alias pour compatibilité
format_messages_mistral = format_messages_qwen

def clean_and_parse_json(text: str) -> Optional[Dict]:
    """Nettoyer et parser du JSON potentiellement mal formaté"""
    text = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \[.*?\] => ', '', text.strip())
    text = re.sub(r'^.*?Extracted content:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?:\s*(?=\{)', '', text)
    
    json_start = text.find('{')
    json_end = text.rfind('}')
    
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_text = text[json_start:json_end+1]
    else:
        return None
    
    json_text = json_text.replace(r'\_', '_')
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        json_text = json_text.replace("'", '"')
        try:
            return json.loads(json_text)
        except:
            return None

def ensure_json_response(text: str, request_format: Optional[Dict] = None) -> str:
    """S'assurer que la réponse est du JSON valide si demandé"""
    if request_format and request_format.get("type") == "json_object":
        parsed = clean_and_parse_json(text)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False)
        else:
            return json.dumps({
                "response": text,
                "error": "Could not parse as valid JSON"
            }, ensure_ascii=False)
    return text

@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    load_model()

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Qwen2.5-14B GGUF API",
        "status": "running",
        "model": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/ws": "WebSocket - Chat endpoint (requires token in query)",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check"
        }
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
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5-14b-gguf",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "Qwen",
                "permission": [],
                "root": "qwen2.5-14b-gguf",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_token)])
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint compatible OpenAI pour les complétions de chat"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = format_messages_qwen(request.messages)
        
        if request.response_format and request.response_format.get("type") == "json_object":
            # Qwen comprend mieux les instructions JSON
            prompt += "Please respond with valid JSON only. No explanations, just the JSON object.\n"
        
        response = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop or ["<|im_end|>", "<|endoftext|>"],  # CHANGÉ ICI: stop tokens pour Qwen
            echo=False
        )
        
        generated_text = response['choices'][0]['text'].strip()
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
                prompt_tokens=response['usage']['prompt_tokens'],
                completion_tokens=response['usage']['completion_tokens'],
                total_tokens=response['usage']['total_tokens']
            )
        )
        
        return chat_response
        
    except Exception as e:
        print(f"Erreur lors de la génération: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== ENDPOINT WEBSOCKET ======

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """Endpoint WebSocket pour les complétions de chat"""
    
    # Vérifier le token
    if token != API_TOKEN:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()
    
    # Envoyer un message de bienvenue
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "model": "qwen2.5-14b-gguf"
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
                continue
            
            # Traiter la requête
            try:
                # Convertir les messages en objets Message
                messages = [Message(**msg) for msg in data.get("messages", [])]
                prompt = format_messages_qwen(messages)
                
                # Ajouter instruction JSON si demandé
                if data.get("response_format", {}).get("type") == "json_object":
                    prompt += "Please respond with valid JSON only. No explanations, just the JSON object.\n"
                
                # Générer
                start_time = time.time()
                
                response = llm(
                    prompt,
                    max_tokens=data.get("max_tokens", 512),
                    temperature=data.get("temperature", 0.7),
                    top_p=data.get("top_p", 0.95),
                    stop=data.get("stop", ["<|im_end|>", "<|endoftext|>"]),  # CHANGÉ ICI
                    echo=False
                )
                
                elapsed = (time.time() - start_time) * 1000
                
                # Extraire et nettoyer
                generated_text = response['choices'][0]['text'].strip()
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
                    "time_ms": round(elapsed)
                }
                
                # Ajouter request_id s'il existe
                if "request_id" in data:
                    response_json["request_id"] = data["request_id"]
                
                await websocket.send_json(response_json)
                
                print(f"[WS] Réponse envoyée en {elapsed:.0f}ms")
                
            except Exception as e:
                print(f"[WS] Erreur: {str(e)}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)