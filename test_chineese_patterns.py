#!/usr/bin/env python3
"""
Test pour détecter les caractères non-latins (chinois, japonais, coréens) dans les réponses
Teste plusieurs scénarios et variations de prompts pour s'assurer de la stabilité
"""

import asyncio
import websockets
import json
import time
import re
import sys
from datetime import datetime
from typing import List, Dict, Optional, Set
import unicodedata

# Configuration
POD_ID = "0g1yim1q032z3x"
WS_URL = f"wss://{POD_ID}-8000.proxy.runpod.net/ws"
API_TOKEN = "supersecret"

# Le prompt système optimisé
SYSTEM_PROMPT = """Tu es une messagerie vocale automatique. Tu ne donnes jamais de conseils médicaux.

APPROCHE CONVERSATIONNELLE:
- Montre de l'empathie et de la compréhension face aux problèmes de santé
- Pose des questions de clarification naturelles sur le motif
- Utilise des transitions douces entre les questions
- Adapte ton ton selon l'urgence exprimée
- Explore toujours le motif en détail avant de demander l'identité
- Adapter le nombre de questions sur le motif selon la complexité, pose les une par une.
- Si la demande de l'utilisateur est totallement hors sujet explique que tu es la pour des questions relatives au cabinet dentaire et redemander avant de continuer

COLLECTE STRICTE dans cet ordre:
1. **Motif détaillé**: 
   - Commence par comprendre vraiment le besoin, toujours avec des questions
   - Pose 2-3 questions de clarification pour bien cerner la demande
   - Exemples de questions selon le contexte:
     * "Je comprends que vous avez mal aux dents. Depuis combien de temps ressentez-vous cette douleur ?"
     * "Est-ce une douleur constante ou qui va et vient ?"
     * "Y a-t-il quelque chose qui déclenche ou aggrave la douleur ?"
     * "Pour quel motif souhaitez-vous prendre ce rendez-vous ?"

2. **Transition naturelle vers l'identification**:
   - "Pour que je puisse transmettre votre demande avec tous les détails, j'aurais besoin de quelques informations. Puis-je avoir votre nom de famille ?"
3. **Nom de famille** 
4. **Prénom** 
5. **Date de naissance**
6. **Statut patient**:
   - "Êtes-vous déjà patient(e) du cabinet ?"
7. **Praticien habituel** (SEULEMENT si déjà patient = oui):
   - "Avec quel praticien avez-vous l'habitude de consulter ?"

RÈGLES ABSOLUES:
- Réponds UNIQUEMENT en français. N'utilise JAMAIS de caractères chinois, japonais ou coréens.
- Sois conversationnel et empathique
- Une seule question courte par message
- Terminer chaque phrase par une question durant la collecte
- Après confirmation du nom, demander le prénom dans un NOUVEAU message
- Ne JAMAIS répéter une information déjà collectée
- TOUJOURS utiliser des formulations neutres et le vouvoiement (pas de Monsieur/Madame)

GESTION DES INTERRUPTIONS:
- Si l'utilisateur dit "urgent" ou exprime de la douleur, répondre: "J'ai bien noté votre urgence. Puis-je avoir votre nom de famille ?"
- Ne pas se laisser détourner du script de collecte

FIN OBLIGATOIRE:
- Une fois les informations collectées tu termines par "J'ai toutes les informations! §FIN§"
- Ne JAMAIS ajouter §FIN§ si tu viens de poser une question
- Vérifier que toutes les informations obligatoires ont été collectées ET confirmées"""

def detect_non_latin_characters(text: str) -> Dict[str, any]:
    """
    Détecte les caractères non-latins dans un texte
    Retourne un dictionnaire avec les statistiques et les caractères trouvés
    """
    non_latin_chars = set()
    char_positions = []
    
    # Catégories Unicode pour CJK (Chinese, Japanese, Korean)
    cjk_ranges = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Extension A
        (0x20000, 0x2A6DF), # CJK Extension B
        (0x2A700, 0x2B73F), # CJK Extension C
        (0x2B740, 0x2B81F), # CJK Extension D
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
        (0x1100, 0x11FF),   # Hangul Jamo
        (0xAC00, 0xD7AF),   # Hangul Syllables
        (0x3130, 0x318F),   # Hangul Compatibility Jamo
        (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
        (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
    ]
    
    for i, char in enumerate(text):
        ord_val = ord(char)
        
        # Vérifier si le caractère est dans une plage CJK
        for start, end in cjk_ranges:
            if start <= ord_val <= end:
                non_latin_chars.add(char)
                char_positions.append({
                    'char': char,
                    'position': i,
                    'unicode': f'U+{ord_val:04X}',
                    'name': unicodedata.name(char, 'UNKNOWN')
                })
                break
        
        # Vérifier aussi par catégorie Unicode
        category = unicodedata.category(char)
        if category.startswith('Lo'):  # "Letter, other" - souvent non-latin
            script = unicodedata.name(char, '').split()[0] if unicodedata.name(char, '') else ''
            if any(s in script for s in ['CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL']):
                non_latin_chars.add(char)
                if not any(p['char'] == char for p in char_positions):
                    char_positions.append({
                        'char': char,
                        'position': i,
                        'unicode': f'U+{ord_val:04X}',
                        'name': unicodedata.name(char, 'UNKNOWN')
                    })
    
    return {
        'has_non_latin': len(non_latin_chars) > 0,
        'count': len(non_latin_chars),
        'unique_chars': list(non_latin_chars),
        'positions': char_positions,
        'text_preview': text[:100] + '...' if len(text) > 100 else text
    }

# Scénarios de test variés pour déclencher potentiellement des caractères non-latins
TEST_SCENARIOS = {
    "normal": {
        "name": "Conversation normale",
        "messages": [
            "Bonjour, j'ai mal aux dents",
            "Depuis hier soir",
            "C'est une douleur qui lance",
            "Dupont",
            "Marie",
            "15 mars 1985",
            "Oui",
            "Dr Martin"
        ]
    },
    
    "urgence": {
        "name": "Cas d'urgence",
        "messages": [
            "Au secours j'ai très mal ! C'est urgent !",
            "Ma dent est cassée",
            "Elle saigne beaucoup",
            "Martin",
            "Pierre",
            "12 juillet 1990",
            "Non c'est la première fois"
        ]
    },
    
    "confusion": {
        "name": "Patient confus",
        "messages": [
            "Euh... je sais pas trop...",
            "Ben... j'ai un truc bizarre dans la bouche",
            "C'est difficile à expliquer",
            "Comment ça mon nom ?",
            "Ah oui, Lefebvre",
            "Sophie",
            "28 février 1978",
            "Je crois que oui"
        ]
    },
    
    "technique": {
        "name": "Termes techniques",
        "messages": [
            "J'ai une péricoronarite sur la 38",
            "Oui avec œdème et trismus",
            "Depuis 3 jours, avec fièvre",
            "Bernard",
            "Jean-Claude",
            "15 juin 1965",
            "Oui",
            "Dr Dubois"
        ]
    },
    
    "emotionnel": {
        "name": "Patient émotionnel",
        "messages": [
            "J'ai tellement peur du dentiste... 😰",
            "J'ai une phobie depuis tout petit",
            "Mais là j'ai vraiment mal",
            "Moreau",
            "Sylvie",
            "20 août 1992",
            "Non jamais"
        ]
    },
    
    "multilingue": {
        "name": "Tentative multilingue",
        "messages": [
            "Hello, I have toothache",
            "Sorry, j'ai mal aux dents",
            "Depuis ce matin",
            "Johnson",
            "Non pardon, c'est Jeanson",
            "Philippe",
            "3 mai 1988",
            "Non"
        ]
    },
    
    "hors_sujet": {
        "name": "Demandes hors sujet",
        "messages": [
            "Bonjour, vous vendez des brosses à dents ?",
            "J'ai besoin d'un dentifrice spécial",
            "Ah ok, alors j'ai une carie",
            "Depuis 2 semaines",
            "Petit",
            "Lucie",
            "7 novembre 1995",
            "Oui",
            "Je sais plus son nom"
        ]
    },
    
    "repetition": {
        "name": "Patient qui répète",
        "messages": [
            "J'ai mal",
            "J'ai mal aux dents",
            "J'ai très mal aux dents",
            "Depuis longtemps",
            "Martin Martin",
            "Martin",
            "Jean Martin",
            "Jean",
            "1er janvier 2000",
            "Non"
        ]
    }
}

class NonLatinCharacterTester:
    def __init__(self):
        self.websocket = None
        self.test_results = {
            "total_tests": 0,
            "tests_with_non_latin": 0,
            "all_non_latin_chars": set(),
            "scenario_results": {},
            "messages_with_issues": []
        }
    
    async def connect(self):
        """Se connecter au WebSocket"""
        uri = f"{WS_URL}?token={API_TOKEN}"
        self.websocket = await websockets.connect(uri)
        
        # Attendre le message de connexion
        conn_msg = await self.websocket.recv()
        msg = json.loads(conn_msg)
        return msg.get('type') == 'connection'
    
    async def send_and_receive(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Envoie des messages et reçoit la réponse complète"""
        request_id = f"test_{int(time.time() * 1000)}"
        
        # Toujours inclure le system prompt
        all_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        all_messages.extend(messages)
        
        payload = {
            "request_id": request_id,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": 100,
            "top_p": 0.1,
            "top_k": 10
        }
        
        await self.websocket.send(json.dumps(payload))
        
        # Collecter la réponse
        response_buffer = ""
        response_complete = False
        
        while not response_complete:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                msg = json.loads(message)
                
                if msg.get('type') == 'stream_token':
                    response_buffer += msg.get('token', '')
                elif msg.get('type') == 'stream_end':
                    response_complete = True
                    
            except asyncio.TimeoutError:
                break
        
        return response_buffer.strip()
    
    async def test_scenario(self, scenario_name: str, scenario: Dict, num_iterations: int = 5):
        """Teste un scénario plusieurs fois avec différentes températures"""
        print(f"\n📋 Test du scénario: {scenario['name']}")
        
        scenario_stats = {
            "name": scenario['name'],
            "iterations": num_iterations,
            "non_latin_found": 0,
            "all_chars_found": set(),
            "messages_with_issues": []
        }
        
        # Tester avec différentes températures pour voir si ça affecte
        temperatures = [0.01, 0.1, 0.3, 0.5, 0.7]
        
        for iteration in range(num_iterations):
            temp = temperatures[iteration % len(temperatures)]
            print(f"  Itération {iteration + 1}/{num_iterations} (temp={temp})...", end='', flush=True)
            
            conversation = []
            issues_found = False
            
            try:
                # Simuler la conversation
                for user_msg in scenario['messages']:
                    conversation.append({"role": "user", "content": user_msg})
                    
                    # Obtenir la réponse
                    response = await self.send_and_receive(conversation, temperature=temp)
                    
                    # Analyser la réponse
                    analysis = detect_non_latin_characters(response)
                    
                    if analysis['has_non_latin']:
                        issues_found = True
                        scenario_stats['non_latin_found'] += 1
                        scenario_stats['all_chars_found'].update(analysis['unique_chars'])
                        
                        issue_detail = {
                            "scenario": scenario_name,
                            "iteration": iteration + 1,
                            "temperature": temp,
                            "user_message": user_msg,
                            "assistant_response": response,
                            "non_latin_chars": analysis['unique_chars'],
                            "positions": analysis['positions']
                        }
                        
                        scenario_stats['messages_with_issues'].append(issue_detail)
                        self.test_results['messages_with_issues'].append(issue_detail)
                        
                        print(f" ❌ Caractères non-latins trouvés: {analysis['unique_chars']}")
                        break
                    
                    conversation.append({"role": "assistant", "content": response})
                    
                    # Arrêter si fin de collecte
                    if "§FIN§" in response:
                        break
                
                if not issues_found:
                    print(" ✅ OK")
                    
            except Exception as e:
                print(f" ⚠️ Erreur: {str(e)}")
        
        # Résumé du scénario
        if scenario_stats['non_latin_found'] > 0:
            print(f"  ⚠️ Problèmes détectés: {scenario_stats['non_latin_found']}/{num_iterations} itérations")
            print(f"  Caractères trouvés: {scenario_stats['all_chars_found']}")
        
        self.test_results['scenario_results'][scenario_name] = scenario_stats
        self.test_results['all_non_latin_chars'].update(scenario_stats['all_chars_found'])
    
    async def run_all_tests(self, iterations_per_scenario: int = 5):
        """Exécute tous les tests"""
        print(f"\n{'='*70}")
        print("TEST DE DÉTECTION DES CARACTÈRES NON-LATINS")
        print(f"{'='*70}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Itérations par scénario: {iterations_per_scenario}")
        print(f"Nombre de scénarios: {len(TEST_SCENARIOS)}")
        
        # Se connecter
        if not await self.connect():
            print("❌ Échec de connexion")
            return
        
        print("✅ Connecté au serveur")
        
        try:
            # Tester chaque scénario
            for scenario_name, scenario in TEST_SCENARIOS.items():
                await self.test_scenario(scenario_name, scenario, iterations_per_scenario)
                
                # Pause entre scénarios
                await asyncio.sleep(1)
            
            # Afficher le rapport final
            self.print_report()
            
        finally:
            if self.websocket:
                await self.websocket.close()
    
    def print_report(self):
        """Affiche le rapport détaillé des tests"""
        print(f"\n{'='*70}")
        print("RAPPORT FINAL")
        print(f"{'='*70}")
        
        # Statistiques globales
        total_iterations = sum(s['iterations'] for s in self.test_results['scenario_results'].values())
        total_issues = sum(s['non_latin_found'] for s in self.test_results['scenario_results'].values())
        
        print(f"\n📊 STATISTIQUES GLOBALES:")
        print(f"  Total d'itérations: {total_iterations}")
        print(f"  Réponses avec caractères non-latins: {total_issues}")
        print(f"  Taux de problème: {(total_issues/total_iterations*100):.2f}%")
        
        if self.test_results['all_non_latin_chars']:
            print(f"\n⚠️ CARACTÈRES NON-LATINS TROUVÉS:")
            for char in sorted(self.test_results['all_non_latin_chars']):
                try:
                    name = unicodedata.name(char, 'UNKNOWN')
                    print(f"  '{char}' - U+{ord(char):04X} - {name}")
                except:
                    print(f"  '{char}' - U+{ord(char):04X}")
        
        print(f"\n📋 RÉSULTATS PAR SCÉNARIO:")
        for scenario_name, stats in self.test_results['scenario_results'].items():
            status = "✅" if stats['non_latin_found'] == 0 else "❌"
            print(f"\n  {status} {stats['name']}:")
            print(f"     Problèmes: {stats['non_latin_found']}/{stats['iterations']}")
            if stats['all_chars_found']:
                print(f"     Caractères: {stats['all_chars_found']}")
        
        # Exemples détaillés
        if self.test_results['messages_with_issues']:
            print(f"\n🔍 EXEMPLES DÉTAILLÉS (premiers 3):")
            for i, issue in enumerate(self.test_results['messages_with_issues'][:3]):
                print(f"\n  Exemple {i+1}:")
                print(f"    Scénario: {issue['scenario']}")
                print(f"    Temperature: {issue['temperature']}")
                print(f"    User: {issue['user_message']}")
                print(f"    Assistant: {issue['assistant_response'][:100]}...")
                print(f"    Caractères problématiques: {issue['non_latin_chars']}")
                if issue['positions']:
                    print(f"    Positions: {[p['position'] for p in issue['positions'][:5]]}")
        
        # Conclusion
        print(f"\n{'='*70}")
        if total_issues == 0:
            print("✅ SUCCÈS: Aucun caractère non-latin détecté!")
        else:
            print(f"❌ ÉCHEC: {total_issues} réponses contiennent des caractères non-latins")
            print("\nRECOMMANDATIONS:")
            print("1. Ajouter une règle explicite dans le prompt système")
            print("2. Utiliser un post-processing pour filtrer ces caractères")
            print("3. Ajuster les paramètres du modèle (temperature, top_p)")
            print("4. Vérifier l'encodage des tokenizers")

async def quick_test():
    """Test rapide pour débugger"""
    tester = NonLatinCharacterTester()
    
    print("Test rapide de détection...")
    
    # Tester quelques phrases directement
    test_texts = [
        "Bonjour, comment allez-vous ?",
        "J'ai bien noté votre demande.",
        "您好，我是语音留言系统。",  # Chinois intentionnel pour test
        "こんにちは",  # Japonais intentionnel pour test
        "안녕하세요",  # Coréen intentionnel pour test
        "Très bien, continuons en français."
    ]
    
    for text in test_texts:
        analysis = detect_non_latin_characters(text)
        status = "❌" if analysis['has_non_latin'] else "✅"
        print(f"{status} '{text[:30]}...' - Non-latin: {analysis['has_non_latin']}")
        if analysis['has_non_latin']:
            print(f"   Caractères: {analysis['unique_chars']}")

async def main():
    """Point d'entrée principal"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        await quick_test()
    else:
        iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        tester = NonLatinCharacterTester()
        await tester.run_all_tests(iterations)

if __name__ == "__main__":
    asyncio.run(main())