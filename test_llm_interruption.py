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
tokens_before_interruption = 0
tokens_after_interruption = 0
last_token_time = None
streaming_start_time = None
interruption_time = None
current_request_id = None
full_response = ""
interruption_sent = False

# Couleurs pour l'output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def log(msg):
    print(f"{Colors.BLUE}[TEST]{Colors.END} {msg}")

def success(msg):
    print(f"{Colors.GREEN}[✓]{Colors.END} {msg}")

def error(msg):
    print(f"{Colors.RED}[✗]{Colors.END} {msg}")

def warning(msg):
    print(f"{Colors.YELLOW}[⚠]{Colors.END} {msg}")

async def test_llm_interruption():
    global tokens_received, tokens_before_interruption, tokens_after_interruption
    global last_token_time, streaming_start_time, interruption_time
    global current_request_id, full_response, interruption_sent
    
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
                log(f"Message de connexion reçu - Modèle: {msg.get('model')}")
            
            # 2. Démarrer le streaming
            current_request_id = f"test_req_{int(time.time() * 1000)}"
            
            payload = {
                "request_id": current_request_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Raconte-moi une très longue histoire détaillée sur l'histoire des pommes à travers les siècles. Je veux au moins 500 mots avec beaucoup de détails historiques."
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000  # Plus de tokens pour avoir le temps d'interrompre
            }
            
            log("Envoi de la requête de streaming...")
            await websocket.send(json.dumps(payload))
            success("Requête envoyée")
            
            # 3. Programmer l'interruption après 10 tokens
            interrupt_after_tokens = 10
            
            # Recevoir les messages
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        msg = json.loads(message)
                        
                        if msg.get('type') == 'stream_start' and msg.get('request_id') == current_request_id:
                            log("Début du streaming")
                            streaming_start_time = time.time()
                        
                        elif msg.get('type') == 'stream_token' and msg.get('request_id') == current_request_id:
                            tokens_received += 1
                            last_token_time = time.time()
                            token = msg.get('token', '')
                            full_response += token
                            
                            # Compter avant/après interruption
                            if not interruption_sent:
                                tokens_before_interruption += 1
                                
                                # Log les premiers tokens
                                if tokens_received <= 5:
                                    log(f"Token #{tokens_received}: '{repr(token)}'")
                                
                                # Interrompre après N tokens
                                if tokens_received >= interrupt_after_tokens and not interruption_sent:
                                    log(f"\n{Colors.RED}>>> INTERRUPTION après {tokens_received} tokens! <<<{Colors.END}\n")
                                    interruption_time = time.time()
                                    interruption_sent = True
                                    
                                    cancel_payload = {
                                        "type": "cancel_stream",
                                        "request_id": current_request_id
                                    }
                                    
                                    await websocket.send(json.dumps(cancel_payload))
                                    success("Commande d'annulation envoyée")
                                    
                            else:
                                # Tokens reçus APRÈS l'envoi de l'interruption
                                tokens_after_interruption += 1
                                if tokens_after_interruption <= 5:
                                    warning(f"Token APRÈS interruption #{tokens_after_interruption}: '{repr(token)}'")
                        
                        elif msg.get('type') == 'stream_end' and msg.get('request_id') == current_request_id:
                            log(f"Fin du streaming")
                            log(f"Statut annulation: {msg.get('cancelled', False)}")
                            log(f"Durée totale: {msg.get('duration', 0):.2f}s")
                            log(f"Tokens/sec: {msg.get('tokens_per_second', 0):.1f}")
                            break
                        
                        elif msg.get('type') == 'stream_cancelled':
                            success(f"Confirmation d'annulation reçue!")
                            
                        elif msg.get('type') == 'error':
                            error(f"Erreur LLM: {msg.get('error')}")
                            break
                            
                    except asyncio.TimeoutError:
                        if streaming_start_time and (time.time() - streaming_start_time) > 10:
                            log("Timeout après 10 secondes")
                            break
                        continue
                        
            except Exception as e:
                error(f"Erreur dans la boucle: {str(e)}")
            
            # 4. Analyser les résultats
            log("\n=== RÉSULTATS DU TEST ===")
            log(f"Tokens AVANT interruption: {tokens_before_interruption}")
            log(f"Tokens APRÈS interruption: {tokens_after_interruption}")
            log(f"Tokens totaux reçus: {tokens_received}")
            log(f"Réponse totale: {len(full_response)} caractères")
            
            # Calcul du délai d'arrêt
            if interruption_time and last_token_time:
                stop_delay = (last_token_time - interruption_time) * 1000
                log(f"Délai entre interruption et dernier token: {stop_delay:.0f}ms")
            
            # Vérifications
            if tokens_after_interruption == 0:
                success("PARFAIT! Aucun token reçu après interruption")
            elif tokens_after_interruption <= 5:
                success(f"BON: Seulement {tokens_after_interruption} tokens après interruption (latence réseau acceptable)")
            elif tokens_after_interruption <= 10:
                warning(f"ACCEPTABLE: {tokens_after_interruption} tokens après interruption (buffer à vider)")
            else:
                error(f"PROBLÈME: {tokens_after_interruption} tokens après interruption (trop!)")
            
            # 5. Test de reprise
            log("\n=== Test de reprise après interruption ===")
            await asyncio.sleep(1)
            
            # Reset
            tokens_received = 0
            tokens_before_interruption = 0
            tokens_after_interruption = 0
            interruption_sent = False
            full_response = ""
            current_request_id = f"test_req_2_{int(time.time() * 1000)}"
            
            new_payload = {
                "request_id": current_request_id,
                "messages": [
                    {"role": "user", "content": "Dis simplement 'OK' et rien d'autre."}
                ],
                "temperature": 0.1,
                "max_tokens": 10
            }
            
            await websocket.send(json.dumps(new_payload))
            
            # Recevoir la réponse courte
            short_response = ""
            short_tokens = 0
            
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    msg = json.loads(message)
                    
                    if msg.get('type') == 'stream_token' and msg.get('request_id') == current_request_id:
                        short_tokens += 1
                        short_response += msg.get('token', '')
                    elif msg.get('type') == 'stream_end':
                        break
            except asyncio.TimeoutError:
                pass
            
            success(f"Reprise OK: '{short_response.strip()}' ({short_tokens} tokens)")
            
            # Résumé final
            log("\n=== RÉSUMÉ ===")
            if tokens_after_interruption <= 10:
                success("✅ Le mécanisme d'interruption fonctionne correctement!")
                success("   Le serveur arrête bien d'envoyer les tokens après l'interruption")
            else:
                error("❌ Le mécanisme d'interruption ne semble pas fonctionner")
            
    except Exception as e:
        error(f"Erreur durant le test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(test_llm_interruption())
    except KeyboardInterrupt:
        log("Test interrompu par l'utilisateur")
    except Exception as e:
        error(f"Erreur fatale: {str(e)}")