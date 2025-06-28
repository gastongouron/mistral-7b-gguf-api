#!/usr/bin/env python3
"""
API FastAPI pour servir le modèle Mistral 7B GGUF avec llama-cpp-python
Avec authentification Bearer et format de réponse Claude-like
"""
import os
import time
import uuid
import json
import re
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from llama_cpp import Llama

# Configuration
MODEL_PATH = "/app/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
API_TOKEN = os.getenv("API_TOKEN", "supersecret")  # Peut être défini via variable d'environnement

# Sécurité
security = HTTPBearer()

# Modèles Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mistral-7b-gguf"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[str, str]] = None  # Pour forcer JSON

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
    title="Mistral 7B GGUF API",
    version="1.0.0",
    description="API FastAPI pour Mistral 7B avec llama-cpp-python et authentification"
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
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle non trouvé. Téléchargement depuis {MODEL_URL}...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        import httpx
        with httpx.stream("GET", MODEL_URL) as response:
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        print("Téléchargement terminé!")

def load_model():
    """Charger le modèle GGUF"""
    global llm
    
    # Vérifier/télécharger le modèle
    download_model_if_needed()
    
    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    
    # Paramètres pour llama-cpp-python
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,  # Taille du contexte
        n_threads=8,  # Nombre de threads CPU
        n_gpu_layers=35,  # Nombre de couches sur GPU
        verbose=True
    )
    
    print("Modèle chargé avec succès!")

def format_messages_mistral(messages: List[Message]) -> str:
    """Formater les messages pour Mistral Instruct"""
    formatted = ""
    
    for i, message in enumerate(messages):
        if message.role == "system":
            formatted += f"[INST] {message.content}\n"
        elif message.role == "user":
            if i == 0 or messages[i-1].role != "system":
                formatted += f"[INST] {message.content} [/INST]"
            else:
                formatted += f"{message.content} [/INST]"
        elif message.role == "assistant":
            formatted += f" {message.content}</s> "
    
    return formatted.strip()

def clean_and_parse_json(text: str) -> Optional[Dict]:
    """Nettoyer et parser du JSON potentiellement mal formaté"""
    # Enlever les timestamps et préfixes
    text = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \[.*?\] => ', '', text.strip())
    text = re.sub(r'^.*?Extracted content:\s*', '', text, flags=re.IGNORECASE)
    
    # Trouver le JSON dans le texte
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not json_match:
        return None
    
    json_text = json_match.group(0)
    
    # Corriger les problèmes courants
    # Remplacer les underscores échappés
    json_text = json_text.replace(r'\_', '_')
    
    # Essayer de parser
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Essayer de corriger les guillemets simples
        json_text = json_text.replace("'", '"')
        try:
            return json.loads(json_text)
        except:
            return None

def ensure_json_response(text: str, request_format: Optional[Dict] = None) -> str:
    """S'assurer que la réponse est du JSON valide si demandé"""
    if request_format and request_format.get("type") == "json_object":
        # Essayer de parser et nettoyer le JSON
        parsed = clean_and_parse_json(text)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False)
        else:
            # Si on ne peut pas parser, créer une structure JSON basique
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
    """Point d'entrée de l'API - pas d'auth nécessaire"""
    return {
        "message": "Mistral 7B GGUF API",
        "status": "running",
        "model": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions endpoint (requires Bearer token)",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifier l'état de l'API - pas d'auth nécessaire"""
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
                "id": "mistral-7b-gguf",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "TheBloke",
                "permission": [],
                "root": "mistral-7b-gguf",
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
        # Formater les messages
        prompt = format_messages_mistral(request.messages)
        
        # Si on demande du JSON, ajouter des instructions
        if request.response_format and request.response_format.get("type") == "json_object":
            prompt += "\n[INST] Réponds uniquement avec un objet JSON valide, sans texte supplémentaire. [/INST]"
        
        # Générer la réponse
        response = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop or ["</s>", "[INST]"],
            echo=False
        )
        
        # Extraire le texte généré
        generated_text = response['choices'][0]['text'].strip()
        
        # S'assurer que c'est du JSON valide si demandé
        generated_text = ensure_json_response(generated_text, request.response_format)
        
        # Créer la réponse au format OpenAI/Claude
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

@app.get("/v1/chat/completions")
async def chat_completions_info():
    """Information sur l'endpoint de chat"""
    return {
        "error": "This endpoint only supports POST requests",
        "hint": "Send a POST request with a JSON body containing 'messages' array and Bearer token in header"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)