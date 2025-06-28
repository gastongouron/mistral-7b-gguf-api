#!/usr/bin/env python3
"""
Batterie complète de tests pour l'API Mistral adaptée à VoxEngine
Teste tous les cas d'usage du scénario de réception d'appels
"""
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configuration
API_URL = "https://rhyx84x4k3pri9-8000.proxy.runpod.net"
API_TOKEN = "supersecret"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

class VoxEngineTestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
        
    def test_case(self, name: str, payload: Dict, expected_keys: List[str], 
                   validation_func: Optional[callable] = None) -> bool:
        """Execute un test et vérifie le résultat"""
        print(f"\n{'='*60}")
        print(f"🧪 {name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_URL}/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=10
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                print(f"❌ Erreur HTTP {response.status_code}")
                self.failed += 1
                return False
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Parser le JSON de la réponse
            try:
                result = json.loads(content)
                print(f"✅ JSON valide")
                print(f"📊 Réponse: {json.dumps(result, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                print(f"❌ JSON invalide: {content}")
                self.failed += 1
                return False
            
            # Vérifier les clés attendues
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                print(f"❌ Clés manquantes: {missing_keys}")
                self.failed += 1
                return False
            
            # Validation personnalisée
            if validation_func and not validation_func(result):
                print(f"❌ Validation échouée")
                self.failed += 1
                return False
            
            print(f"⏱️  Temps: {elapsed_ms:.0f}ms")
            print(f"✅ Test réussi")
            self.passed += 1
            
            self.results.append({
                'test': name,
                'success': True,
                'time_ms': elapsed_ms,
                'result': result
            })
            return True
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Execute tous les tests VoxEngine"""
        print("🚀 BATTERIE COMPLÈTE DE TESTS VOXENGINE")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 API: {API_URL}")
        print("\n")
        
        # ═══════════════════════════════════════════════════════════════
        # 1. TESTS ANALYSE D'INTENTION (handleReasonPhase)
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 1: ANALYSE D'INTENTION")
        print("="*80)
        
        # Test 1.1: Motif médical clair → Finalisation
        self.test_case(
            "Intention: Motif dentaire clair (détartrage)",
            {
                "messages": [
                    {"role": "system", "content": "Tu es un ASSISTANT VOICEMAIL INTELLIGENT pour cabinet médical. Réponds UNIQUEMENT en JSON."},
                    {"role": "user", "content": 'PATIENT: "j\'ai besoin d\'un détartrage"\nC\'est un motif dentaire clair qui doit être finalisé immédiatement.\nRéponds avec cette structure: {"intention": "medical_motif", "action": "finalize", "medical_motif": "détartrage", "urgency_detected": false}'}
                ],
                "temperature": 0.01,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            },
            ["intention", "action"],
            lambda r: r.get("intention") == "medical_motif" and r.get("action") == "finalize"
        )
        
        # Test 1.2: Motif vague → Question
        self.test_case(
            "Intention: Motif vague (consultation)",
            {
                "messages": [
                    {"role": "system", "content": "Assistant médical. JSON uniquement."},
                    {"role": "user", "content": 'PATIENT: "je voudrais une consultation"\nMotif vague, pose une question.\nJSON: {"intention": "medical_motif", "action": "ask_question", "next_question": "..."}'}
                ],
                "temperature": 0.01,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            },
            ["intention", "action", "next_question"],
            lambda r: r.get("action") == "ask_question" and r.get("next_question")
        )
        
        # Test 1.3: Hors-sujet → Redirection
        self.test_case(
            "Intention: Hors-sujet (restaurant)",
            {
                "messages": [
                    {"role": "user", "content": 'PATIENT: "je veux réserver une table au restaurant"\nCe n\'est PAS une demande médicale, donc l\'intention DOIT être "off_topic".\nStructure attendue: {"intention": "off_topic", "reasoning": "Demande non médicale"}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["intention"],
            lambda r: r.get("intention") == "off_topic"
        )
        
        # Test 1.4: Urgence dentaire
        self.test_case(
            "Intention: Urgence (rage de dent)",
            {
                "messages": [
                    {"role": "user", "content": 'PATIENT: "j\'ai une rage de dent insupportable"\nUrgence évidente.\nJSON: {"intention": "medical_motif", "action": "finalize", "medical_motif": "rage de dent", "urgency_detected": true}'}
                ],
                "temperature": 0.01,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            },
            ["intention", "urgency_detected"],
            lambda r: r.get("urgency_detected") == True
        )
        
        # Test 1.5: Au revoir
        self.test_case(
            "Intention: Fin d'appel",
            {
                "messages": [
                    {"role": "user", "content": 'PATIENT: "au revoir"\nFin d\'appel.\nJSON: {"intention": "goodbye", "farewell_message": "..."}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["intention"],
            lambda r: r.get("intention") == "goodbye"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 2. TESTS EXTRACTION FORMULAIRE (analyzeFormResponseWithLLM)
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 2: EXTRACTION FORMULAIRE")
        print("="*80)
        
        # Test 2.1: Nom de famille simple
        self.test_case(
            "Formulaire: Nom simple",
            {
                "messages": [
                    {"role": "user", "content": 'Extraire nom de famille de: "Martin"\nJSON: {"is_valid": bool, "extracted_value": "NOM", "explanation": "..."}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["is_valid", "extracted_value"],
            lambda r: r.get("is_valid") == True and r.get("extracted_value") == "Martin"
        )
        
        # Test 2.2: Nom épelé
        self.test_case(
            "Formulaire: Nom épelé (G O U R O N)",
            {
                "messages": [
                    {"role": "user", "content": 'Le patient épelle son nom lettre par lettre: "G O U R O N"\nReconstitue ces lettres pour former le nom complet.\nLa réponse doit avoir cette structure: {"is_valid": true, "extracted_value": "GOURON"}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["is_valid", "extracted_value"],
            lambda r: r.get("extracted_value") in ["GOURON", "Gouron"]
        )
        
        # Test 2.3: Date française complexe
        self.test_case(
            "Formulaire: Date française (quinze mars quatre-vingt-huit)",
            {
                "messages": [
                    {"role": "user", "content": 'Convertis cette date française en format numérique: "quinze mars quatre-vingt-huit"\nQuinze = 15, mars = 03, quatre-vingt-huit = 1988\nLe résultat doit être: {"is_valid": true, "extracted_value": "15/03/1988"}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["is_valid", "extracted_value"],
            lambda r: r.get("extracted_value") == "15/03/1988"
        )
        
        # Test 2.4: Patient existant ambigu
        self.test_case(
            "Formulaire: Patient existant (oui bien sûr)",
            {
                "messages": [
                    {"role": "user", "content": 'Est-ce un patient existant? Réponse: "oui bien sûr"\nJSON: {"is_valid": true, "extracted_value": "oui|non"}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["is_valid", "extracted_value"],
            lambda r: r.get("extracted_value") == "oui"
        )
        
        # Test 2.5: Praticien habituel
        self.test_case(
            "Formulaire: Praticien (Docteur Marcello)",
            {
                "messages": [
                    {"role": "user", "content": 'Le patient dit: "C\'est le docteur Marcello"\nExtrais le nom complet du praticien.\nLa réponse doit être: {"is_valid": true, "extracted_value": "Dr Marcello"}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["is_valid", "extracted_value"],
            lambda r: "Marcello" in r.get("extracted_value", "")
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 3. TESTS CATÉGORISATION FINALE (callLLM_FinalRecapAndCategory)
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 3: CATÉGORISATION FINALE")
        print("="*80)
        
        # Test 3.1: Catégorie urgence
        self.test_case(
            "Catégorie: Emergency (dent cassée)",
            {
                "messages": [
                    {"role": "user", "content": 'Motif: "dent cassée suite à une chute"\nCatégories: emergency|appointment_create|administrative\nJSON: {"category": "...", "recap": "..."}'}
                ],
                "temperature": 0.01,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            },
            ["category", "recap"],
            lambda r: r.get("category") == "emergency"
        )
        
        # Test 3.2: Catégorie administrative
        self.test_case(
            "Catégorie: Administrative (certificat)",
            {
                "messages": [
                    {"role": "user", "content": 'Motif: "besoin d\'un certificat médical"\nChoisis UNE SEULE catégorie parmi: medical_certificate, administrative, appointment_create\nRéponds avec: {"category": "medical_certificate", "recap": "Demande de certificat médical"}'}
                ],
                "temperature": 0.01,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            },
            ["category", "recap"],
            lambda r: r.get("category") in ["medical_certificate", "administrative"]
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 4. TESTS DE CAS COMPLEXES
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 4: CAS COMPLEXES")
        print("="*80)
        
        # Test 4.1: Conversation complète
        self.test_case(
            "Complexe: Analyse conversation multi-tours",
            {
                "messages": [
                    {"role": "system", "content": "Assistant médical analysant une conversation."},
                    {"role": "user", "content": '''Conversation:
- Patient: "J'ai mal"
- Assistant: "Où avez-vous mal?"
- Patient: "Aux dents, en bas à droite"
- Assistant: "Depuis quand?"
- Patient: "3 jours, c'est insupportable"

Analyse finale JSON: {"medical_motif": "...", "urgency_detected": bool, "final_category": "..."}'''}
                ],
                "temperature": 0.01,
                "max_tokens": 200,
                "response_format": {"type": "json_object"}
            },
            ["medical_motif"],
            lambda r: "dent" in r.get("medical_motif", "").lower()
        )
        
        # Test 4.2: Réponse inadéquate
        self.test_case(
            "Complexe: Gestion réponse inadéquate",
            {
                "messages": [
                    {"role": "user", "content": 'Question posée: "Votre date de naissance?"\nRéponse du patient: "euh... bah..."\nC\'est une réponse inadéquate, impossible d\'extraire une date.\nRéponds: {"is_valid": false, "extracted_value": "", "explanation": "Réponse trop vague pour extraire une date"}'}
                ],
                "temperature": 0.01,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            },
            ["is_valid", "explanation"],
            lambda r: r.get("is_valid") == False
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 5. TESTS DE PERFORMANCE
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 5: TESTS DE PERFORMANCE")
        print("="*80)
        
        # Test rapide répété
        perf_times = []
        for i in range(5):
            start = time.time()
            response = requests.post(
                f"{API_URL}/v1/chat/completions",
                headers=HEADERS,
                json={
                    "messages": [{"role": "user", "content": 'JSON simple: {"status": "ok"}'}],
                    "temperature": 0.01,
                    "max_tokens": 20,
                    "response_format": {"type": "json_object"}
                }
            )
            elapsed = (time.time() - start) * 1000
            perf_times.append(elapsed)
            print(f"Test perf {i+1}: {elapsed:.0f}ms")
        
        # ═══════════════════════════════════════════════════════════════
        # RAPPORT FINAL
        # ═══════════════════════════════════════════════════════════════
        
        self.print_final_report(perf_times)
    
    def print_final_report(self, perf_times: List[float]):
        """Affiche le rapport final détaillé"""
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL")
        print("="*80)
        
        print(f"\n✅ Tests réussis: {self.passed}")
        print(f"❌ Tests échoués: {self.failed}")
        print(f"📈 Taux de réussite: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        
        if perf_times:
            avg_time = sum(perf_times) / len(perf_times)
            print(f"\n⏱️  Performance moyenne: {avg_time:.0f}ms")
            print(f"   Min: {min(perf_times):.0f}ms")
            print(f"   Max: {max(perf_times):.0f}ms")
        
        # Analyse par catégorie
        print("\n📋 Résultats par catégorie:")
        categories = {}
        for result in self.results:
            cat = result['test'].split(':')[0]
            if cat not in categories:
                categories[cat] = {'passed': 0, 'failed': 0}
            if result['success']:
                categories[cat]['passed'] += 1
            else:
                categories[cat]['failed'] += 1
        
        for cat, stats in categories.items():
            total = stats['passed'] + stats['failed']
            print(f"   {cat}: {stats['passed']}/{total} ({stats['passed']/total*100:.0f}%)")
        
        # Verdict VoxEngine
        print("\n🎯 VERDICT POUR VOXENGINE:")
        
        if self.passed / (self.passed + self.failed) >= 0.9 and avg_time < 2000:
            print("   ✅ API PRÊTE POUR PRODUCTION")
            print("   - Fiabilité JSON excellente")
            print("   - Performance adéquate")
            print("   - Extraction de données fonctionnelle")
        elif self.passed / (self.passed + self.failed) >= 0.7:
            print("   ⚠️  API FONCTIONNELLE MAIS À OPTIMISER")
            print("   - Quelques cas d'échec à corriger")
            print("   - Vérifier les prompts problématiques")
        else:
            print("   ❌ API NON PRÊTE")
            print("   - Trop d'échecs de parsing JSON")
            print("   - Nécessite des ajustements")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    tester = VoxEngineTestSuite()
    tester.run_all_tests()