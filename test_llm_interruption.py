#!/usr/bin/env python3
"""
Test d'interruption pour le serveur LLM
Ce script simule VoxImplant qui :
1. Se connecte au LLM WebSocket
2. Lance un streaming
3. Interrompt après quelques tokens
4. Vérifie que le streaming s'arrête vraiment
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

# Configuration
POD_ID = "0g1yim1q032z3x"
LLM_WS_URL = f"wss://{POD_ID}-8000.proxy.runpod.net/ws"
LLM_API_KEY = "supersecret"

# État global
tokens_received = 0
last_token_time = None
streaming_start_time = None
interruption_time = None
tokens_after_interruption = 0
current_request_id = None
full_response = ""

# Couleurs pour l'output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'

def log(msg):
    print(f"{Colors.BLUE}[TEST]{Colors.END} {msg}")

def success(msg):
    print(f"{Colors.GREEN}[✓]{Colors.END} {msg}")

def error(msg):
    print(f"{Colors.RED}[✗]{Colors.END} {msg}")

# Prompt système pour générer une longue réponse
SYSTEM_PROMPT = """Tu es un assistant qui doit donner des explications TRÈS détaillées et longues.
Peu importe la question, tu dois répondre avec AU MOINS 200 mots, en donnant beaucoup de détails."""

USER_MESSAGE = "Explique-moi ce qu'est une pomme."

async def test_llm_interruption():
    global tokens_received, last_token_time, streaming_start_time
    global interruption_time, tokens_after_interruption, current_request_id, full_response
    
    try:
        log("=== Début du test d'interruption LLM ===")
        
        # 1. Connexion au WebSocket
        uri = f"{LLM_WS_URL}?token={LLM_API_KEY}"
        log(f"Connexion LLM: {uri}")
        
        async with websockets.connect(uri) as websocket:
            success("LLM WebSocket connecté")
            
            # Attendre le message de connexion
            connection_msg = await websocket.recv()
            msg = json.loads(connection_msg)
            if msg.get('type') == 'connection':
                log("Message de connexion reçu")
            
            # 2. Démarrer le streaming
            current_request_id = f"test_req_{int(time.time() * 1000)}"
            
            # Essayons un format plus simple
            payload = {
                "request_id": current_request_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Raconte-moi une longue histoire sur les pommes. Au moins 200 mots s'il te plaît."
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            log("Envoi de la requête de streaming...")
            await websocket.send(json.dumps(payload))
            success("Requête envoyée")
            
            # 3. Recevoir les tokens pendant 0.5 secondes avant d'interrompre
            interrupt_task = asyncio.create_task(interrupt_after_delay(websocket, 0.5))
            
            # Recevoir les messages
            try:
                while True:
                    try:
                        # Augmenter le timeout à 5 secondes
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        msg = json.loads(message)
                        
                        # DEBUG: Log tous les messages
                        log(f"Message reçu: {json.dumps(msg, indent=2)}")
                        
                        if msg.get('type') == 'stream_start' and msg.get('request_id') == current_request_id:
                            log("Début du streaming")
                            streaming_start_time = time.time()
                        
                        elif msg.get('type') == 'stream_token' and msg.get('request_id') == current_request_id:
                            tokens_received += 1
                            last_token_time = time.time()
                            token = msg.get('token', '')
                            full_response += token
                            
                            # Compter les tokens après interruption
                            if interruption_time and time.time() > interruption_time:
                                tokens_after_interruption += 1
                                if tokens_after_interruption == 1:
                                    log(f"[INTERRUPTION] Premier token APRÈS interruption: #{tokens_received}")
                            
                            # Log chaque token pour debug
                            if tokens_received <= 5 or tokens_received % 10 == 0:
                                log(f"Token #{tokens_received}: '{token}'")
                        
                        elif msg.get('type') == 'stream_end' and msg.get('request_id') == current_request_id:
                            log(f"Fin du streaming - Réponse complète: {len(full_response)} caractères")
                            break
                        
                        elif msg.get('type') == 'error':
                            error(f"Erreur LLM: {msg.get('error')}")
                            error(f"Message complet: {json.dumps(msg, indent=2)}")
                            break
                            
                    except asyncio.TimeoutError:
                        # Ne pas break immédiatement, continuer à attendre
                        if streaming_start_time and (time.time() - streaming_start_time) > 10:
                            log("Timeout après 10 secondes de streaming")
                            break
                        continue
                        
            except Exception as e:
                error(f"Erreur dans la boucle de réception: {str(e)}")
            
            # Attendre que l'interruption soit faite
            await interrupt_task
            
            # 5. Attendre encore 2 secondes pour voir si des tokens arrivent
            log("Attente de 2 secondes pour vérifier l'arrêt...")
            tokens_before_wait = tokens_received
            tokens_after_cancel = 0
            
            try:
                for _ in range(20):  # 20 x 100ms = 2 secondes
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    msg = json.loads(message)
                    
                    if msg.get('type') == 'stream_token' and msg.get('request_id') == current_request_id:
                        tokens_received += 1
                        tokens_after_interruption += 1
                        tokens_after_cancel += 1
                        last_token_time = time.time()
                        
                        if tokens_after_cancel <= 5:
                            log(f"[APRÈS CANCEL] Token #{tokens_received} reçu")
                            
                    elif msg.get('type') == 'stream_cancelled' and msg.get('request_id') == current_request_id:
                        log(f"[CANCEL] Confirmation d'annulation reçue!")
                        break
                        
            except asyncio.TimeoutError:
                pass
            
            # 6. Analyser les résultats
            log("=== RÉSULTATS DU TEST ===")
            log(f"Tokens totaux reçus: {tokens_received}")
            log(f"Tokens reçus APRÈS interruption: {tokens_after_interruption}")
            
            if last_token_time:
                time_since_last_token = int((time.time() - last_token_time) * 1000)
                log(f"Temps depuis dernier token: {time_since_last_token}ms")
                
                if time_since_last_token > 1000:
                    success("Streaming bien arrêté (pas de tokens depuis > 1s)")
                else:
                    error("Streaming semble continuer!")
            
            # Vérifications
            if tokens_after_interruption == 0:
                success("PARFAIT! Aucun token reçu après interruption")
            elif tokens_after_interruption < 5:
                log(f"ACCEPTABLE: {tokens_after_interruption} tokens reçus après interruption (buffer résiduel)")
            else:
                error(f"PROBLÈME! {tokens_after_interruption} tokens reçus après interruption")
            
            # 7. Test bonus : nouveau streaming après interruption
            log("\n=== Test de reprise après interruption ===")
            await asyncio.sleep(1)
            
            tokens_received = 0
            full_response = ""
            current_request_id = f"test_req_2_{int(time.time() * 1000)}"
            
            new_payload = {
                "request_id": current_request_id,
                "messages": [
                    {"role": "user", "content": "Dis juste 'Bonjour' et rien d'autre."}
                ],
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": True
            }
            
            await websocket.send(json.dumps(new_payload))
            
            # Recevoir la réponse courte
            short_response = ""
            short_tokens = 0
            
            try:
                for _ in range(30):  # Max 3 secondes
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    msg = json.loads(message)
                    
                    if msg.get('type') == 'stream_token' and msg.get('request_id') == current_request_id:
                        short_tokens += 1
                        short_response += msg.get('token', '')
                    elif msg.get('type') == 'stream_end':
                        break
            except asyncio.TimeoutError:
                pass
            
            success(f"Nouveau streaming: \"{short_response}\" ({short_tokens} tokens)")
            
    except Exception as e:
        error(f"Erreur durant le test: {str(e)}")
        raise

async def interrupt_after_delay(websocket, delay):
    """Interrompt le streaming après un délai"""
    global interruption_time
    
    await asyncio.sleep(delay)
    
    log("INTERRUPTION DU STREAMING LLM!")
    interruption_time = time.time()
    
    # DEBUG: Vérifier que current_request_id est bien défini
    log(f"Request ID à annuler: {current_request_id}")
    
    cancel_payload = {
        "type": "cancel_stream",
        "request_id": current_request_id
    }
    
    await websocket.send(json.dumps(cancel_payload))
    success("Commande d'annulation envoyée")

if __name__ == "__main__":
    try:
        asyncio.run(test_llm_interruption())
    except KeyboardInterrupt:
        log("Test interrompu par l'utilisateur")
    except Exception as e:
        error(f"Erreur fatale: {str(e)}")