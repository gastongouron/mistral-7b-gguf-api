#!/usr/bin/env python3
"""
Test du flux conversationnel complet avec Qwen2.5-32B
Simule une vraie conversation comme VoxImplant
Version corrigée sans bugs
"""

import asyncio
import websockets
import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Optional

# Configuration
POD_ID = "6owx2sl5ef6dcy"
WS_URL = f"wss://{POD_ID}-8000.proxy.runpod.net/ws"
API_TOKEN = "supersecret"

# Le prompt système optimisé pour Qwen
SYSTEM_PROMPT = """Tu es UNIQUEMENT une messagerie vocale automatique. NE JAMAIS donner de conseils médicaux.

COLLECTE STRICTE dans cet ordre:
1. Nom de famille
2. Prénom (APRÈS avoir confirmé le nom)
3. Date de naissance complète
4. Si déjà patient (oui/non)
5. Praticien habituel (SEULEMENT si déjà patient = oui)

RÈGLES ABSOLUES:
- Une seule question courte par message
- Pour l'orthographe des noms: "J'ai noté [épellation], c'est correct ?"
- Après confirmation du nom, demander le prénom dans un NOUVEAU message
- Ne JAMAIS répéter une information déjà collectée
- TOUJOURS utiliser des formulations neutres (pas de Monsieur/Madame)

GESTION DES INTERRUPTIONS:
- Si l'utilisateur dit "urgent" ou exprime de la douleur, répondre: "J'ai bien noté votre urgence. Puis-je avoir votre nom de famille ?"
- Ne pas se laisser détourner du script de collecte

EXEMPLES STRICTS:
User: "Bonjour j'ai mal aux dents"
Toi: "Bonjour, j'ai bien noté votre appel. Puis-je avoir votre nom de famille ?"

User: "Dupont"
Toi: "J'ai noté D-U-P-O-N-T, c'est correct ?"

User: "Oui"
Toi: "Merci. Quel est votre prénom ?"

User: "Marie"
Toi: "J'ai bien noté Marie. Quelle est votre date de naissance complète ?"

FIN OBLIGATOIRE:
Quand TOUTES les informations sont collectées, terminer EXACTEMENT par:
"Parfait, j'ai toutes les informations nécessaires. ##FIN_COLLECTE##"

IMPORTANT: Le marqueur ##FIN_COLLECTE## est OBLIGATOIRE à la fin."""

# Scénarios de test
SCENARIOS = {
    "ideal": {
        "name": "Parcours idéal",
        "intro": "Bonjour, j'ai mal aux dents depuis 3 jours",
        "responses": {
            "nom de famille": "Dubois",
            "c'est correct": "Oui c'est ça",
            "prénom": "Marie",
            "date de naissance": "15 mars 1985",
            "déjà patient": "Oui, je viens régulièrement",
            "praticien habituel": "Docteur Martin",
            "praticien": "Docteur Martin"
        },
        "expected_category": "appointment_create"
    },
    
    "interruption": {
        "name": "Avec interruptions",
        "intro": "Bonjour, j'ai une rage de dents terrible, c'est urgent !",
        "responses": {
            "nom de famille": "Durand",
            "bien noté": "Mon nom c'est Durand",
            "j'ai bien noté votre urgence": "Durand",
            "c'est correct": "Oui",
            "prénom": "Pierre",
            "date de naissance": "12 juillet 1990",
            "déjà patient": "Non c'est la première fois",
            "description": "C'est insupportable, ma joue est gonflée, je pense que c'est un abcès"
        },
        "expected_category": "emergency"
    },
    
    "confus": {
        "name": "Patient confus",
        "intro": "Euh bonjour, je sais pas trop, j'ai un problème avec mon rendez-vous",
        "responses": {
            "nom de famille": "Lefebvre",
            "besoin de votre nom": "Ah oui, Lefebvre",
            "pour pouvoir vous aider": "Lefebvre", 
            "l-e-f-e-b-v-r-e": "Non attendez, c'est Lefèvre avec un accent",
            "l-e-f-è-v-r-e": "Oui c'est ça",  # Accepter quand corrigé
            "prénom": "Sophie",
            "date de naissance": "28 février 1978",
            "déjà patient": "Je pense que oui, mais ça fait longtemps",
            "praticien": "Je ne me souviens plus de son nom",
            "rendez-vous": "Je voulais changer mon rendez-vous de la semaine prochaine"
        },
        "expected_category": "appointment_update"
    },
    
    "rapide": {
        "name": "Patient qui donne tout d'un coup",
        "intro": "Bonjour, Pierre Martin, j'ai rendez-vous demain mais je dois l'annuler",
        "responses": {
            "nom de famille": "Martin",
            "c'est correct": "Oui M-A-R-T-I-N",
            "prénom": "Pierre",
            "date de naissance": "10 octobre 1980",
            "déjà patient": "Bien sûr, j'ai rendez-vous demain avec Dr Moreau",
            "praticien": "Dr Moreau",
            "annuler": "Oui c'est ça, un imprévu professionnel"
        },
        "expected_category": "appointment_delete"
    },
    
    "certificat": {
        "name": "Demande de certificat",
        "intro": "Bonjour, j'ai besoin d'un certificat médical pour mon travail",
        "responses": {
            "nom de famille": "Bernard",
            "c'est correct": "Oui",
            "prénom": "Jean-Claude",
            "date de naissance": "15 juin 1965",
            "déjà patient": "Oui, je suis patient du Dr Dubois",
            "praticien": "Dr Dubois",
            "certificat": "C'est pour une reprise du sport au travail"
        },
        "expected_category": "medical_certificate"
    }
}

class ConversationTester:
    def __init__(self, scenario_name: str):
        self.scenario = SCENARIOS[scenario_name]
        self.scenario_name = scenario_name
        self.websocket = None
        self.messages = []
        self.current_request_id = None
        self.response_buffer = ""
        self.conversation_complete = False
        self.test_results = {
            "scenario": scenario_name,
            "turns": 0,
            "errors": [],
            "fin_collecte_detected": False,
            "conversation": [],
            "expected_category": self.scenario.get("expected_category", "unknown")
        }
        # Tracker pour éviter les boucles
        self.last_questions = []
    
    async def connect(self):
        """Se connecter au WebSocket"""
        uri = f"{WS_URL}?token={API_TOKEN}"
        print(f"🔌 Connexion à {uri}")
        self.websocket = await websockets.connect(uri)
        
        # Attendre le message de connexion
        conn_msg = await self.websocket.recv()
        msg = json.loads(conn_msg)
        if msg.get('type') == 'connection':
            print(f"✅ Connecté - Modèle: {msg.get('model')}")
            return True
        return False
    
    async def send_message(self, user_input: str):
        """Envoyer un message utilisateur avec config optimisée"""
        print(f"\n👤 USER: {user_input}")
        self.test_results["conversation"].append({"role": "user", "content": user_input})
        
        # Ajouter le message à l'historique
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Construire la requête
        self.current_request_id = f"test_{int(time.time() * 1000)}"
        
        # Pour Qwen, toujours inclure le système prompt
        messages_to_send = []
        
        # Toujours commencer par le système
        messages_to_send.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })
        
        # Ajouter l'historique de conversation
        messages_to_send.extend(self.messages)
        
        # Configuration optimisée pour suivre les instructions
        payload = {
            "request_id": self.current_request_id,
            "messages": messages_to_send,
            "temperature": 0.01,  # Très bas pour suivre le script
            "max_tokens": 60,     # Limité pour éviter les digressions
            "top_p": 0.1,         # Très focalisé
            "top_k": 10,          # Limiter les choix
            "repetition_penalty": 1.15  # Éviter les répétitions
        }
        
        await self.websocket.send(json.dumps(payload))
        
        # Attendre et collecter la réponse
        self.response_buffer = ""
        response_complete = False
        
        while not response_complete:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                msg = json.loads(message)
                
                if msg.get('type') == 'stream_start':
                    continue
                    
                elif msg.get('type') == 'stream_token' and msg.get('request_id') == self.current_request_id:
                    self.response_buffer += msg.get('token', '')
                    
                elif msg.get('type') == 'stream_end' and msg.get('request_id') == self.current_request_id:
                    response_complete = True
                    
            except asyncio.TimeoutError:
                self.test_results["errors"].append("Timeout en attendant la réponse")
                break
        
        # Nettoyer la réponse
        assistant_response = self.response_buffer.strip()
        
        # Vérifier si FIN_COLLECTE
        if "##FIN_COLLECTE##" in assistant_response:
            self.test_results["fin_collecte_detected"] = True
            self.conversation_complete = True
            # Ne pas enlever le marqueur pour l'analyse
        
        # Ajouter à l'historique
        self.messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        print(f"🤖 ASSISTANT: {assistant_response}")
        self.test_results["conversation"].append({"role": "assistant", "content": assistant_response})
        self.test_results["turns"] += 1
        
        # Tracker les questions pour détecter les boucles
        self.last_questions.append(assistant_response.lower())
        if len(self.last_questions) > 3:
            self.last_questions.pop(0)
        
        return assistant_response
    
    def find_best_response(self, assistant_message: str) -> Optional[str]:
        """Trouve la meilleure réponse du scénario basée sur le message de l'assistant"""
        assistant_lower = assistant_message.lower()
        
        # Détection spécifique par type de question
        if "nom de famille" in assistant_lower:
            return self.scenario["responses"].get("nom de famille", "Dubois")
        
        elif "c'est correct" in assistant_lower:
            # Vérifier le contexte exact
            if "l-e-f-è-v-r-e" in assistant_lower:
                # C'est la version corrigée avec accent
                return self.scenario["responses"].get("l-e-f-è-v-r-e", "Oui")
            elif "l-e-f-e-b-v-r-e" in assistant_lower:
                # C'est la version sans accent
                return self.scenario["responses"].get("l-e-f-e-b-v-r-e", "Non, avec un accent sur le è")
            elif "prénom" in assistant_lower:
                # Le modèle redemande car il a détecté une erreur
                return self.scenario["responses"].get("prénom", "Marie")
            else:
                return self.scenario["responses"].get("c'est correct", "Oui")
        
        elif "prénom" in assistant_lower:
            # Si le modèle insiste pour le prénom, donner un prénom différent
            if len(self.last_questions) >= 2 and all("prénom" in q for q in self.last_questions[-2:]):
                # On est dans une boucle, forcer une réponse différente
                return "Marie" if self.scenario_name != "ideal" else "Sophie"
            return self.scenario["responses"].get("prénom", "Marie")
        
        elif "date de naissance" in assistant_lower or "date naissance" in assistant_lower:
            return self.scenario["responses"].get("date de naissance", "15 mars 1985")
        
        elif "déjà patient" in assistant_lower or "patient ici" in assistant_lower or "êtes-vous déjà" in assistant_lower:
            return self.scenario["responses"].get("déjà patient", "Oui")
        
        elif "praticien" in assistant_lower:
            return self.scenario["responses"].get("praticien", 
                   self.scenario["responses"].get("praticien habituel", "Dr Martin"))
        
        elif "j'ai bien noté votre urgence" in assistant_lower or "bien noté votre urgence" in assistant_lower:
            # Pour le scénario interruption
            return self.scenario["responses"].get("j'ai bien noté votre urgence",
                   self.scenario["responses"].get("bien noté", "Durand"))
        
        elif "pour pouvoir vous aider" in assistant_lower:
            # Pour le scénario confus
            return self.scenario["responses"].get("pour pouvoir vous aider", "Lefebvre")
        
        # Fallback: chercher par mots-clés
        best_match = None
        best_score = 0
        
        for trigger, response in self.scenario["responses"].items():
            trigger_words = trigger.lower().split()
            matches = sum(1 for word in trigger_words if word in assistant_lower)
            
            if matches > best_score:
                best_score = matches
                best_match = response
        
        return best_match
    
    async def run_scenario(self):
        """Exécuter le scénario complet"""
        print(f"\n{'='*60}")
        print(f"SCÉNARIO: {self.scenario['name']}")
        print(f"Catégorie attendue: {self.scenario.get('expected_category', 'unknown')}")
        print(f"{'='*60}")
        
        try:
            # Se connecter
            if not await self.connect():
                self.test_results["errors"].append("Échec de connexion")
                return
            
            # Envoyer le message d'introduction
            response = await self.send_message(self.scenario["intro"])
            
            # Boucle de conversation
            max_turns = 20  # Un peu plus pour laisser le modèle finir
            turn = 0
            
            while not self.conversation_complete and turn < max_turns:
                turn += 1
                
                # Trouver la meilleure réponse
                user_response = self.find_best_response(response)
                
                if not user_response:
                    # Réponse de fallback intelligente
                    print(f"⚠️  Pas de réponse trouvée pour: {response[:50]}...")
                    user_response = "Je ne comprends pas la question"
                
                # Petite pause pour simuler un humain
                await asyncio.sleep(0.5)
                
                # Envoyer la réponse
                response = await self.send_message(user_response)
            
            if turn >= max_turns and not self.conversation_complete:
                self.test_results["errors"].append(f"Conversation trop longue ({max_turns} tours)")
            
        except Exception as e:
            self.test_results["errors"].append(f"Erreur: {str(e)}")
        
        finally:
            if self.websocket:
                await self.websocket.close()
    
    def print_results(self):
        """Afficher les résultats du test"""
        print(f"\n{'='*60}")
        print("RÉSULTATS DU TEST")
        print(f"{'='*60}")
        
        print(f"Scénario: {self.test_results['scenario']}")
        print(f"Tours de conversation: {self.test_results['turns']}")
        print(f"FIN_COLLECTE détecté: {'✅ OUI' if self.test_results['fin_collecte_detected'] else '❌ NON'}")
        print(f"Catégorie attendue: {self.test_results['expected_category']}")
        
        if self.test_results["errors"]:
            print(f"\nERREURS ({len(self.test_results['errors'])}):")
            for error in self.test_results["errors"]:
                print(f"  - {error}")
        
        # Analyser la conversation
        print("\nANALYSE:")
        collected_info = {
            "nom": False,
            "prénom": False,
            "date_naissance": False,
            "déjà_patient": False,
            "praticien": False,
            "motif": False
        }
        
        questions_asked = {
            "nom": False,
            "prénom": False,
            "date_naissance": False,
            "déjà_patient": False,
            "praticien": False
        }
        
        # Analyser les questions posées
        for msg in self.test_results["conversation"]:
            if msg["role"] == "assistant":
                content_lower = msg["content"].lower()
                if "nom de famille" in content_lower:
                    questions_asked["nom"] = True
                if "prénom" in content_lower:
                    questions_asked["prénom"] = True
                if "date de naissance" in content_lower or "date naissance" in content_lower:
                    questions_asked["date_naissance"] = True
                if "patient" in content_lower and ("déjà" in content_lower or "êtes-vous" in content_lower):
                    questions_asked["déjà_patient"] = True
                if "praticien" in content_lower:
                    questions_asked["praticien"] = True
        
        # Analyser les réponses données
        for i, msg in enumerate(self.test_results["conversation"]):
            if msg["role"] == "user":
                content_lower = msg["content"].lower()
                # Vérifier le contexte de la question précédente
                prev_assistant = None
                if i > 0:
                    prev_assistant = self.test_results["conversation"][i-1]["content"].lower()
                
                # Nom
                if any(nom in content_lower for nom in ["dubois", "durand", "lefebvre", "martin", "bernard"]):
                    if not prev_assistant or "prénom" not in prev_assistant:
                        collected_info["nom"] = True
                
                # Prénom
                if any(prenom in content_lower for prenom in ["marie", "pierre", "sophie", "jean-claude", "sophia"]):
                    if prev_assistant and "prénom" in prev_assistant:
                        collected_info["prénom"] = True
                
                # Date naissance
                if any(year in content_lower for year in ["1985", "1990", "1978", "1980", "1965"]):
                    collected_info["date_naissance"] = True
                
                # Déjà patient
                if prev_assistant and "patient" in prev_assistant:
                    if "oui" in content_lower or "non" in content_lower:
                        collected_info["déjà_patient"] = True
                
                # Praticien
                if ("docteur" in content_lower or "dr" in content_lower) and prev_assistant and "praticien" in prev_assistant:
                    collected_info["praticien"] = True
                
                # Motif
                if i < 2:  # Généralement dans les premiers messages
                    if any(word in content_lower for word in ["mal", "douleur", "urgent", "rdv", "rendez-vous", "certificat", "abcès"]):
                        collected_info["motif"] = True
        
        print("\nQuestions posées:")
        for info, asked in questions_asked.items():
            status = "✅" if asked else "❌"
            print(f"  {status} {info}")
        
        print("\nInformations collectées:")
        for info, collected in collected_info.items():
            status = "✅" if collected else "❌"
            print(f"  {status} {info}")
        
        # Score
        questions_score = sum(1 for asked in questions_asked.values() if asked)
        info_score = sum(1 for collected in collected_info.values() if collected)
        print(f"\nScores:")
        print(f"  Questions: {questions_score}/5")
        print(f"  Informations: {info_score}/6")
        
        # Évaluation
        if self.test_results["fin_collecte_detected"] and info_score >= 5:
            print("\n✅ TEST RÉUSSI - Le système a collecté les informations et terminé correctement")
        elif info_score >= 4:
            print("\n⚠️  TEST PARTIEL - Bonnes informations mais pas de fin détectée")
        else:
            print("\n❌ TEST ÉCHOUÉ - Informations manquantes ou comportement incorrect")

async def main():
    """Exécuter tous les tests"""
    print(f"Test du flux conversationnel Qwen2.5-32B")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WebSocket: {WS_URL}")
    
    # Choix du scénario
    if len(sys.argv) > 1 and sys.argv[1] in SCENARIOS:
        scenarios_to_test = [sys.argv[1]]
    else:
        print("\nScénarios disponibles:")
        for name, scenario in SCENARIOS.items():
            print(f"  - {name}: {scenario['name']} → {scenario.get('expected_category', '?')}")
        print(f"\nUtilisation: python {sys.argv[0]} [scenario_name]")
        print("Exécution de tous les scénarios...\n")
        scenarios_to_test = list(SCENARIOS.keys())
    
    # Exécuter les tests
    all_results = []
    
    for scenario_name in scenarios_to_test:
        tester = ConversationTester(scenario_name)
        await tester.run_scenario()
        tester.print_results()
        
        all_results.append({
            "scenario": scenario_name,
            "success": tester.test_results["fin_collecte_detected"],
            "turns": tester.test_results["turns"],
            "errors": len(tester.test_results["errors"]),
            "expected_category": tester.test_results["expected_category"]
        })
        
        # Pause entre les scénarios
        if scenario_name != scenarios_to_test[-1]:
            print("\nPause de 3 secondes avant le prochain scénario...")
            await asyncio.sleep(3)
    
    # Résumé global
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("RÉSUMÉ GLOBAL")
        print(f"{'='*60}")
        
        for result in all_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['scenario']}: {result['turns']} tours, {result['errors']} erreurs → {result['expected_category']}")
        
        success_rate = sum(1 for r in all_results if r["success"]) / len(all_results) * 100
        print(f"\nTaux de réussite: {success_rate:.0f}%")

if __name__ == "__main__":
    asyncio.run(main())