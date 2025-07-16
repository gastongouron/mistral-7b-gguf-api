#!/usr/bin/env python3
"""
Test de l'endpoint d'extraction /v1/summary avec une conversation simulée
Utilise le même prompt que VoxImplant pour tester Qwen2.5-32B
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
POD_ID = "6owx2sl5ef6dcy"
API_URL = f"https://{POD_ID}-8000.proxy.runpod.net"
API_TOKEN = "supersecret"

# Le prompt système utilisé par VoxImplant

# Nouveau prompt d'extraction plus strict
EXTRACTION_PROMPT = """Tu es un système d'extraction d'informations. Analyse cette conversation médicale et retourne UNIQUEMENT un objet JSON valide.

RÈGLES CRITIQUES:
- Extrais UNIQUEMENT les informations explicitement mentionnées
- Utilise null (sans guillemets) pour les informations absentes
- Ne jamais utiliser "null" en string
- AUCUN texte avant ou après le JSON
- Le JSON doit être valide et parsable

Structure EXACTE attendue:
{
  "nom": "valeur trouvée" ou null,
  "prenom": "valeur trouvée" ou null,
  "dateNaissance": "JJ/MM/AAAA" ou null,
  "dejaPatient": "oui" ou "non" ou null,
  "praticien": "Dr Nom" ou null,
  "motif": "motif de l'appel" ou null,
  "resume": "résumé court obligatoire",
  "categorie": "appointment_create" ou "emergency" ou "appointment_update" ou "medical_certificate" ou "general_question"
}

CATÉGORIES (utilise UNIQUEMENT une de ces valeurs):
- emergency
- urgent_consultation
- appointment_confirmation
- appointment_create
- appointment_update
- appointment_delete
- appointment_read
- retard
- prescription_renewal
- medical_certificate
- test_results
- general_question
- administrative
- follow_up
- non_categorise
- complex_question

Choisis la catégorie la plus appropriée selon le contexte.

IMPORTANT: Commence directement par { et termine par }"""

# Différentes conversations de test
CONVERSATIONS = {
    "simple": [
        ("user", "Bonjour, j'ai mal aux dents depuis 3 jours"),
        ("assistant", "Bonjour, j'ai bien noté votre appel. Puis-je avoir votre nom de famille ?"),
        ("user", "Dubois"),
        ("assistant", "J'ai noté D-U-B-O-I-S, c'est correct ?"),
        ("user", "Oui c'est ça"),
        ("assistant", "Merci. Quel est votre prénom ?"),
        ("user", "Marie"),
        ("assistant", "Très bien Marie. Quelle est votre date de naissance complète ?"),
        ("user", "15 mars 1985"),
        ("assistant", "J'ai noté le 15/03/1985. Êtes-vous déjà patient du cabinet ?"),
        ("user", "Oui, je viens régulièrement"),
        ("assistant", "D'accord. Quel est votre praticien habituel ?"),
        ("user", "Docteur Martin"),
        ("assistant", "Parfait, j'ai toutes les informations nécessaires. ##FIN_COLLECTE##")
    ],
    
    "urgence": [
        ("user", "Bonjour, j'ai une rage de dents terrible, c'est urgent !"),
        ("assistant", "Bonjour, j'ai bien noté votre appel urgent. Puis-je avoir votre nom de famille ?"),
        ("user", "Durand, Pierre Durand"),
        ("assistant", "J'ai noté D-U-R-A-N-D, c'est correct ?"),
        ("user", "Oui"),
        ("assistant", "Merci. J'ai bien noté Pierre comme prénom. Quelle est votre date de naissance ?"),
        ("user", "12 juillet 1990"),
        ("assistant", "J'ai noté le 12/07/1990. Êtes-vous déjà patient du cabinet ?"),
        ("user", "Non c'est la première fois"),
        ("assistant", "D'accord. Pouvez-vous me décrire votre douleur ?"),
        ("user", "C'est insupportable, j'ai un abcès je pense, la joue est gonflée"),
        ("assistant", "Parfait, j'ai toutes les informations nécessaires. ##FIN_COLLECTE##")
    ],
    
    "rdv_existant": [
        ("user", "Bonjour, je voudrais modifier mon rendez-vous de la semaine prochaine"),
        ("assistant", "Bonjour, j'ai bien noté votre demande de modification. Puis-je avoir votre nom de famille ?"),
        ("user", "Lefebvre"),
        ("assistant", "J'ai noté L-E-F-E-B-V-R-E, c'est correct ?"),
        ("user", "C'est ça"),
        ("assistant", "Merci. Quel est votre prénom ?"),
        ("user", "Sophie"),
        ("assistant", "Très bien Sophie. Quelle est votre date de naissance ?"),
        ("user", "28 février 1978"),
        ("assistant", "J'ai noté le 28/02/1978. Êtes-vous déjà patient du cabinet ?"),
        ("user", "Oui bien sûr, je viens depuis des années"),
        ("assistant", "D'accord. Quel est votre praticien habituel ?"),
        ("user", "Dr Moreau"),
        ("assistant", "Parfait, j'ai toutes les informations nécessaires. ##FIN_COLLECTE##")
    ],
    
    "incomplete": [
        ("user", "Allo, j'ai besoin d'un certificat médical"),
        ("assistant", "Bonjour, j'ai bien noté votre demande de certificat. Puis-je avoir votre nom de famille ?"),
        ("user", "Bernard"),
        ("assistant", "J'ai noté B-E-R-N-A-R-D, c'est correct ?"),
        ("user", "Oui"),
        ("assistant", "Merci. Quel est votre prénom ?"),
        ("user", "Jean-Claude"),
        ("assistant", "Très bien Jean-Claude. Quelle est votre date de naissance ?"),
        ("user", "Euh... je suis né en 1965 mais je ne me souviens plus du jour exact"),
        ("assistant", "Pas de problème. Êtes-vous déjà patient du cabinet ?"),
        ("user", "Je ne sais plus, ça fait longtemps"),
        ("assistant", "Je comprends. Pour quel motif avez-vous besoin de ce certificat ?"),
        ("user", "C'est pour mon travail, ils me le demandent"),
        ("assistant", "Parfait, j'ai toutes les informations nécessaires. ##FIN_COLLECTE##")
    ]
}

def test_extraction(conversation_name, messages):
    """Teste l'extraction sur une conversation"""
    print(f"\n{'='*60}")
    print(f"TEST: {conversation_name.upper()}")
    print(f"{'='*60}")
    
    # Afficher la conversation
    print("\nCONVERSATION:")
    for role, content in messages:
        print(f"  {role.upper()}: {content}")
    
    # Construire le payload comme VoxImplant
    conversation_text = "\n".join([f"{role}: {content}" for role, content in messages])
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": EXTRACTION_PROMPT
            },
            {
                "role": "user",
                "content": f"Conversation à analyser:\n{conversation_text}"
            }
        ]
    }
    
    # Appeler l'API
    try:
        print("\nAPPEL API...")
        response = requests.post(
            f"{API_URL}/v1/summary",
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            extraction = result.get("extraction", {})
            
            print("\nEXTRACTION RÉUSSIE:")
            print(json.dumps(extraction, indent=2, ensure_ascii=False))
            
            # Vérifications
            print("\nVÉRIFICATIONS:")
            checks = {
                "Nom extrait": extraction.get("nom") is not None,
                "Prénom extrait": extraction.get("prenom") is not None,
                "Date naissance": extraction.get("dateNaissance") is not None,
                "Déjà patient": extraction.get("dejaPatient") is not None,
                "Praticien": extraction.get("praticien") is not None or extraction.get("dejaPatient") == "non",
                "Motif": extraction.get("motif") is not None,
                "Résumé": extraction.get("resume") is not None,
                "Catégorie": extraction.get("categorie") in [
                    "appointment_create", "appointment_update", "emergency",
                    "urgent_consultation", "medical_certificate", "general_question"
                ]
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check}")
            
            # Score
            score = sum(1 for passed in checks.values() if passed)
            print(f"\nSCORE: {score}/{len(checks)}")
            
            return score == len(checks)
            
        else:
            print(f"\n❌ ERREUR API: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        return False

def main():
    """Teste toutes les conversations"""
    print(f"Test d'extraction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API: {API_URL}")
    
    # Vérifier que l'API est accessible
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            health_data = health.json()
            print(f"✅ API en ligne - Modèle chargé: {health_data.get('model_loaded', False)}")
        else:
            print("❌ API non accessible")
            return
    except Exception as e:
        print(f"❌ Impossible de contacter l'API: {e}")
        return
    
    # Tester chaque conversation
    results = {}
    for name, conversation in CONVERSATIONS.items():
        success = test_extraction(name, conversation)
        results[name] = success
        
        # Pause entre les tests
        if name != list(CONVERSATIONS.keys())[-1]:
            print("\nPause de 2 secondes...")
            import time
            time.sleep(2)
    
    # Résumé final
    print(f"\n{'='*60}")
    print("RÉSUMÉ DES TESTS")
    print(f"{'='*60}")
    
    for name, success in results.items():
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"  {name}: {status}")
    
    total_success = sum(1 for success in results.values() if success)
    print(f"\nTOTAL: {total_success}/{len(results)} tests réussis")
    
    # Exemples de catégories attendues
    print("\nCATÉGORIES ATTENDUES:")
    print("  - simple → appointment_create")
    print("  - urgence → emergency ou urgent_consultation")
    print("  - rdv_existant → appointment_update")
    print("  - incomplete → medical_certificate ou general_question")

if __name__ == "__main__":
    main()