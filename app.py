#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Gemma-2-9B GGUF avec llama-cpp-python
Optimisé pour catégorisation et outputs JSON
Avec authentification Bearer et endpoint WebSocket
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

# Configuration
MODEL_PATH = "/app/models/gemma-2-9b-it-Q5_K_M.gguf"
MODEL_URL = "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q5_K_M.gguf"
API_TOKEN = os.getenv("API_TOKEN", "supersecret")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sécurité
security = HTTPBearer()

# Modèles Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-2-9b"
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
    title="Gemma-2-9B GGUF API",
    version="1.0.0",
    description="API FastAPI pour Gemma-2-9B optimisée pour catégorisation et JSON"
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
    """Charger le modèle GGUF avec configuration optimale pour Gemma"""
    global llm
    
    download_model_if_needed()
    
    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    
    # Configuration optimisée pour Gemma-2-9B
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,  # Gemma supporte 8K de contexte
        n_threads=8,
        n_gpu_layers=-1,  # Toutes les couches sur GPU
        n_batch=512,  # Batch plus grand pour Gemma
        use_mmap=True,
        verbose=True,
        seed=42,  # Pour reproductibilité des outputs JSON
        repeat_penalty=1.1  # Évite les répétitions dans JSON
    )
    
    print("Modèle chargé avec succès!")

def format_messages_gemma(messages: List[Message]) -> str:
    """Formater les messages pour Gemma-2 avec focus sur JSON et catégorisation"""
    formatted = ""
    
    # System prompt implicite pour améliorer les outputs structurés
    system_added = False
    
    for message in messages:
        if message.role == "system":
            # Gemma n'a pas de format system explicite, on l'intègre au premier user
            system_added = True
            continue
        elif message.role == "user":
            if system_added and formatted == "":
                # Intégrer le system prompt dans le premier message user
                system_msg = next((m.content for m in messages if m.role == "system"), "")
                formatted += f"<start_of_turn>user\n{system_msg}\n\n{message.content}<end_of_turn>\n"
                system_added = False
            else:
                formatted += f"<start_of_turn>user\n{message.content}<end_of_turn>\n"
        elif message.role == "assistant":
            formatted += f"<start_of_turn>model\n{message.content}<end_of_turn>\n"
    
    # Ajouter le début de la réponse du modèle
    formatted += "<start_of_turn>model\n"
    
    return formatted

# Alias pour compatibilité avec l'ancienne API
format_messages_mistral = format_messages_gemma

def extract_json_from_text(text: str) -> str:
    """Extraire JSON même si Gemma ajoute du texte autour"""
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
            # Fallback : retourner un objet JSON d'erreur
            return json.dumps({
                "response": text,
                "error": "Could not parse as valid JSON",
                "raw_output": text[:200] + "..." if len(text) > 200 else text
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
        "message": "Gemma-2-9B GGUF API",
        "status": "running",
        "model": "gemma-2-9b-it-Q5_K_M.gguf",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/ws": "WebSocket - Chat endpoint (requires token in query)",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check"
        },
        "optimized_for": ["categorization", "JSON outputs", "date extraction", "query reformulation"]
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
                "id": "gemma-2-9b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "Google",
                "permission": [],
                "root": "gemma-2-9b",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_token)])
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint compatible OpenAI optimisé pour catégorisation et JSON"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = format_messages_gemma(request.messages)
        
        # Instructions spécifiques pour JSON avec Gemma
        if request.response_format and request.response_format.get("type") == "json_object":
            # Gemma comprend très bien ces instructions directes
            prompt += "Output only valid JSON. Begin with { and end with }. No explanations or additional text.\n"
        
        # Paramètres optimisés pour génération structurée
        response = llm(
            prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.3,  # Plus bas pour JSON précis
            top_p=request.top_p or 0.9,
            top_k=40,  # Limiter pour outputs plus déterministes
            stop=request.stop or ["<end_of_turn>", "<start_of_turn>", "<eos>"],
            echo=False
        )
        
        generated_text = response['choices'][0]['text'].strip()
        
        # Post-processing spécifique pour JSON
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
        "model": "gemma-2-9b",
        "capabilities": ["categorization", "JSON", "date_extraction", "reformulation"]
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
                prompt = format_messages_gemma(messages)
                
                # Ajouter instruction JSON si demandé
                if data.get("response_format", {}).get("type") == "json_object":
                    prompt += "Output only valid JSON. Begin with { and end with }. No explanations or additional text.\n"
                
                # Générer
                start_time = time.time()
                
                response = llm(
                    prompt,
                    max_tokens=data.get("max_tokens", 512),
                    temperature=data.get("temperature", 0.3),  # Bas par défaut pour précision
                    top_p=data.get("top_p", 0.9),
                    top_k=data.get("top_k", 40),
                    stop=data.get("stop", ["<end_of_turn>", "<start_of_turn>", "<eos>"]),
                    echo=False
                )
                
                elapsed = (time.time() - start_time) * 1000
                
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
                
                print(f"[WS] Réponse envoyée en {elapsed:.0f}ms ({response_json['tokens_per_second']} t/s)")
                
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