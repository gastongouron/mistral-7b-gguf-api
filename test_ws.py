import asyncio
import websockets
import json
import time

# Configuration RunPod
RUNPOD_ID = "wsp137k5y3cf0p"
WS_URL = f"wss://{RUNPOD_ID}-8000.proxy.runpod.net/ws"
TOKEN = "supersecret"  # Remplacez par votre token si différent

async def test_simple():
    """Test simple de connexion et requête"""
    print(f"🔗 Connexion à {WS_URL}")
    
    try:
        # Connexion avec token
        uri = f"{WS_URL}?token={TOKEN}"
        
        async with websockets.connect(uri) as ws:
            # Recevoir le message de connexion
            msg = await ws.recv()
            print("✅ Connecté:", json.loads(msg))
            
            # Test 1: Message simple
            print("\n📤 Envoi d'une requête simple...")
            request = {
                "messages": [
                    {"role": "user", "content": "Dis-moi bonjour en français"}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            start = time.time()
            await ws.send(json.dumps(request))
            
            # Recevoir la réponse
            response = await ws.recv()
            data = json.loads(response)
            elapsed = (time.time() - start) * 1000
            
            if data["type"] == "completion":
                print(f"\n✅ Réponse reçue en {elapsed:.0f}ms")
                print(f"🤖 Mistral: {data['choices'][0]['message']['content']}")
                print(f"⏱️  Temps serveur: {data['time_ms']}ms")
                print(f"📊 Tokens utilisés: {data['usage']['total_tokens']}")
            
            # Test 2: Réponse JSON
            print("\n📤 Test avec format JSON...")
            request2 = {
                "messages": [
                    {"role": "user", "content": "Crée un JSON avec name='Pierre' et age=25"}
                ],
                "max_tokens": 100,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            await ws.send(json.dumps(request2))
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "completion":
                content = data['choices'][0]['message']['content']
                print(f"✅ Réponse JSON: {content}")
                
                # Vérifier que c'est du JSON valide
                try:
                    parsed = json.loads(content)
                    print(f"✅ JSON valide parsé: {parsed}")
                except:
                    print("❌ Le contenu n'est pas du JSON valide")
            
            print("\n✅ Tests terminés avec succès!")
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ Erreur de connexion: {e}")
        print("Vérifiez le token d'authentification")
    except Exception as e:
        print(f"❌ Erreur: {type(e).__name__}: {e}")

async def test_performance():
    """Test de performance avec plusieurs requêtes"""
    print(f"🚀 Test de performance sur {WS_URL}")
    
    uri = f"{WS_URL}?token={TOKEN}"
    
    async with websockets.connect(uri) as ws:
        # Ignorer le message de connexion
        await ws.recv()
        
        times = []
        
        print("📊 Envoi de 5 requêtes successives...")
        
        for i in range(5):
            request = {
                "messages": [
                    {"role": "user", "content": f"Réponds juste 'OK {i+1}'"}
                ],
                "max_tokens": 20,
                "temperature": 0.1
            }
            
            start = time.time()
            await ws.send(json.dumps(request))
            
            response = await ws.recv()
            data = json.loads(response)
            elapsed = (time.time() - start) * 1000
            
            if data["type"] == "completion":
                times.append(elapsed)
                print(f"  Requête {i+1}: {elapsed:.0f}ms - Réponse: {data['choices'][0]['message']['content']}")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"\n📈 Statistiques:")
            print(f"  - Temps moyen: {avg_time:.0f}ms")
            print(f"  - Min: {min(times):.0f}ms")
            print(f"  - Max: {max(times):.0f}ms")

async def main():
    print("🧪 TEST WEBSOCKET MISTRAL SUR RUNPOD")
    print("=" * 50)
    
    # Test simple
    await test_simple()
    
    print("\n" + "=" * 50 + "\n")
    
    # Test de performance
    await test_performance()

if __name__ == "__main__":
    # Installer d'abord: pip install websockets
    asyncio.run(main())