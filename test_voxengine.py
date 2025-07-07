#!/usr/bin/env python3
"""
Batterie complète de tests pour l'API Mistral adaptée à VoxEngine
Version 2.0 avec schémas JSON forcés pour garantir la structure
Tests en mode boîte noire avec validation stricte
"""
import requests
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Configuration
API_URL = "https://75dflbgks878y9-8000.proxy.runpod.net"
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
        
    def test_case(self, name: str, messages: List[Dict], json_schema: Dict[str, Any],
                   validation_func: Optional[callable] = None, 
                   temperature: float = 0.01,
                   max_tokens: int = 200) -> bool:
        """Execute un test avec schéma JSON forcé"""
        print(f"\n{'='*60}")
        print(f"🧪 {name}")
        print(f"{'='*60}")
        
        # Payload avec schéma JSON
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "json_schema": json_schema
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_URL}/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=30  # Timeout augmenté pour les requêtes complexes
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
                self.failed += 1
                return False
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Parser le JSON de la réponse
            try:
                result = json.loads(content)
                print(f"✅ JSON valide")
                print(f"📊 Réponse: {json.dumps(result, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON invalide: {content}")
                print(f"   Erreur: {e}")
                self.failed += 1
                return False
            
            # Vérifier la conformité au schéma
            required_keys = json_schema.get("required", [])
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                print(f"❌ Clés manquantes: {missing_keys}")
                self.failed += 1
                return False
            
            # Validation personnalisée
            if validation_func and not validation_func(result):
                print(f"❌ Validation métier échouée")
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
            
        except requests.exceptions.Timeout:
            print(f"❌ Timeout après 30 secondes")
            self.failed += 1
            return False
        except Exception as e:
            print(f"❌ Exception: {e}")
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Execute tous les tests VoxEngine avec schémas forcés"""
        print("🚀 BATTERIE COMPLÈTE DE TESTS VOXENGINE v2.0")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 API: {API_URL}")
        print(f"🔒 Mode: Schémas JSON forcés")
        print("\n")
        
        # ═══════════════════════════════════════════════════════════════
        # 1. TESTS ANALYSE D'INTENTION (handleReasonPhase)
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 1: ANALYSE D'INTENTION")
        print("="*80)
        
        # Schéma pour l'analyse d'intention médicale
        intention_medical_schema = {
            "type": "object",
            "properties": {
                "intention": {
                    "type": "string",
                    "enum": ["medical_motif", "off_topic", "goodbye"]
                },
                "action": {
                    "type": "string",
                    "enum": ["finalize", "ask_question", "redirect"]
                },
                "medical_motif": {"type": "string"},
                "urgency_detected": {"type": "boolean"},
                "next_question": {"type": "string"}
            },
            "required": ["intention", "action"]
        }
        
        # Test 1.1: Motif médical clair
        self.test_case(
            "Intention: Motif dentaire clair (détartrage)",
            [
                {"role": "system", "content": "Assistant médical pour cabinet dentaire. Analyse l'intention du patient."},
                {"role": "user", "content": "J'ai besoin d'un détartrage"}
            ],
            intention_medical_schema,
            lambda r: r.get("intention") == "medical_motif" and 
                     r.get("action") == "finalize" and
                     "détartrage" in r.get("medical_motif", "").lower()
        )
        
        # Test 1.2: Motif vague
        self.test_case(
            "Intention: Motif vague (consultation)",
            [
                {"role": "system", "content": "Assistant médical. Si le motif est vague, demande des précisions."},
                {"role": "user", "content": "Je voudrais une consultation"}
            ],
            intention_medical_schema,
            lambda r: r.get("action") == "ask_question" and 
                     r.get("next_question") is not None
        )
        
        # Test 1.3: Hors-sujet
        hors_sujet_schema = {
            "type": "object",
            "properties": {
                "intention": {
                    "type": "string",
                    "enum": ["off_topic"]
                },
                "reasoning": {"type": "string"}
            },
            "required": ["intention", "reasoning"]
        }
        
        self.test_case(
            "Intention: Hors-sujet (restaurant)",
            [
                {"role": "system", "content": "Assistant médical. Détecte les demandes non médicales."},
                {"role": "user", "content": "Je veux réserver une table au restaurant"}
            ],
            hors_sujet_schema,
            lambda r: r.get("intention") == "off_topic"
        )
        
        # Test 1.4: Urgence dentaire
        self.test_case(
            "Intention: Urgence (rage de dent)",
            [
                {"role": "system", "content": "Assistant médical. Détecte les urgences dentaires."},
                {"role": "user", "content": "J'ai une rage de dent insupportable depuis cette nuit"}
            ],
            intention_medical_schema,
            lambda r: r.get("urgency_detected") == True and
                     "dent" in r.get("medical_motif", "").lower()
        )
        
        # Test 1.5: Au revoir
        goodbye_schema = {
            "type": "object",
            "properties": {
                "intention": {
                    "type": "string",
                    "enum": ["goodbye"]
                },
                "farewell_message": {"type": "string"}
            },
            "required": ["intention", "farewell_message"]
        }
        
        self.test_case(
            "Intention: Fin d'appel",
            [
                {"role": "system", "content": "Assistant médical. Gère les fins d'appel poliment."},
                {"role": "user", "content": "Au revoir, merci"}
            ],
            goodbye_schema,
            lambda r: r.get("intention") == "goodbye"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 2. TESTS EXTRACTION FORMULAIRE (analyzeFormResponseWithLLM)
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 2: EXTRACTION FORMULAIRE")
        print("="*80)
        
        # Schéma standard pour l'extraction
        extraction_schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "extracted_value": {"type": "string"},
                "explanation": {"type": "string"}
            },
            "required": ["is_valid", "extracted_value"]
        }
        
        # Test 2.1: Nom simple
        self.test_case(
            "Formulaire: Nom simple",
            [
                {"role": "system", "content": "Extrais le nom de famille de la réponse du patient."},
                {"role": "user", "content": "Mon nom c'est Martin"}
            ],
            extraction_schema,
            lambda r: r.get("is_valid") == True and 
                     r.get("extracted_value") == "Martin"
        )
        
        # Test 2.2: Nom épelé
        self.test_case(
            "Formulaire: Nom épelé (G O U R O N)",
            [
                {"role": "system", "content": "Le patient épelle son nom. Reconstitue les lettres pour former le nom complet."},
                {"role": "user", "content": "G O U R O N"}
            ],
            extraction_schema,
            lambda r: r.get("extracted_value").upper() == "GOURON"
        )
        
        # Test 2.3: Prénom composé avec prompt VoxEngine
        prenom_compose_schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "extracted_value": {"type": "string"},
                "explanation": {"type": "string"}
            },
            "required": ["is_valid", "extracted_value"],
            "additionalProperties": False
        }
        
        self.test_case(
            "Formulaire: Prénom composé (Jean Marcello) - Prompt VoxEngine",
            [
                {"role": "system", "content": """Tu es un expert en extraction d'informations. 
RÈGLES pour les prénoms composés:
- Garder TOUS les prénoms mentionnés
- Ajouter des traits d'union entre les prénoms
- "Jean Pierre" → "Jean-Pierre"
- "Marie Claire" → "Marie-Claire"
- Ne JAMAIS tronquer les prénoms"""},
                {"role": "user", "content": "RÉPONSE DU PATIENT: Jean Marcello"}
            ],
            prenom_compose_schema,
            lambda r: r.get("extracted_value") == "Jean-Marcello",
            temperature=0.01,
            max_tokens=200
        )
        
        # ═══════════════════════════════════════════════════════════════
        # TESTS DE DATES FRANÇAISES COMPLEXES
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 TESTS SPÉCIAUX: DATES FRANÇAISES COMPLEXES")
        print("="*80)
        
        date_schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "extracted_value": {
                    "type": "string",
                    "pattern": "^\\d{2}/\\d{2}/\\d{4}$"
                }
            },
            "required": ["is_valid", "extracted_value"]
        }
        
        # Test dates années 70
        self.test_case(
            "Date: 15 mars soixante-quinze (1975)",
            [
                {"role": "system", "content": "Convertis les dates en format DD/MM/YYYY. soixante-quinze = 75 = 1975"},
                {"role": "user", "content": "15 mars soixante-quinze"}
            ],
            date_schema,
            lambda r: r.get("extracted_value") == "15/03/1975"
        )
        
        # Test dates années 80
        self.test_case(
            "Date: 6 mars quatre-vingt-huit (1988)",
            [
                {"role": "system", "content": "Convertis les dates en format DD/MM/YYYY. quatre-vingt-huit = 88 = 1988"},
                {"role": "user", "content": "6 mars quatre-vingt-huit"}
            ],
            date_schema,
            lambda r: r.get("extracted_value") == "06/03/1988"
        )
        
        # Test dates années 90 avec prompt complet VoxEngine
        self.test_case(
            "Date: 6 mars quatre-vingt-quatorze (1994) - Prompt VoxEngine",
            [
                {"role": "system", "content": """Expert en conversion de dates françaises.
RÈGLES CRITIQUES pour 90-99:
- quatre-vingt-dix = 90 → 1990
- quatre-vingt-onze = 91 → 1991
- quatre-vingt-douze = 92 → 1992
- quatre-vingt-treize = 93 → 1993
- quatre-vingt-quatorze = 94 → 1994 (PAS 1984!)
- quatre-vingt-quinze = 95 → 1995
- quatre-vingt-seize = 96 → 1996
- quatre-vingt-dix-sept = 97 → 1997
- quatre-vingt-dix-huit = 98 → 1998
- quatre-vingt-dix-neuf = 99 → 1999

Format de sortie: DD/MM/YYYY (toujours 2 chiffres pour le jour et le mois)"""},
                {"role": "user", "content": "Le 6 mars quatre-vingt-quatorze"}
            ],
            date_schema,
            lambda r: r.get("extracted_value") == "06/03/1994",
            temperature=0.01,
            max_tokens=500
        )
        
        self.test_case(
            "Date: 25 décembre quatre-vingt-dix-sept (1997)",
            [
                {"role": "system", "content": "Convertis en DD/MM/YYYY. quatre-vingt-dix-sept = 97 = 1997"},
                {"role": "user", "content": "25 décembre quatre-vingt-dix-sept"}
            ],
            date_schema,
            lambda r: r.get("extracted_value") == "25/12/1997"
        )
        
        self.test_case(
            "Date: premier janvier quatre-vingt-onze (1991)",
            [
                {"role": "system", "content": "Convertis en DD/MM/YYYY. 'premier' = 01. quatre-vingt-onze = 91 = 1991"},
                {"role": "user", "content": "premier janvier quatre-vingt-onze"}
            ],
            date_schema,
            lambda r: r.get("extracted_value") == "01/01/1991"
        )
        
        # Test 2.4: Patient existant
        oui_non_schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "extracted_value": {
                    "type": "string",
                    "enum": ["oui", "non"]
                }
            },
            "required": ["is_valid", "extracted_value"]
        }
        
        self.test_case(
            "Formulaire: Patient existant (oui bien sûr)",
            [
                {"role": "system", "content": "Extrais 'oui' ou 'non'. 'oui bien sûr' = 'oui'"},
                {"role": "user", "content": "Êtes-vous déjà patient? - Oui bien sûr"}
            ],
            oui_non_schema,
            lambda r: r.get("extracted_value") == "oui"
        )
        
        # Test 2.5: Praticien
        praticien_schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "extracted_value": {"type": "string"},
                "titre_detecte": {"type": "string"}
            },
            "required": ["is_valid", "extracted_value"]
        }
        
        self.test_case(
            "Formulaire: Praticien (Docteur Marcello)",
            [
                {"role": "system", "content": "Extrais le nom du praticien. Garde le titre si mentionné."},
                {"role": "user", "content": "C'est le docteur Marcello"}
            ],
            praticien_schema,
            lambda r: "Marcello" in r.get("extracted_value", "")
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 3. TESTS CATÉGORISATION FINALE
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 3: CATÉGORISATION FINALE")
        print("="*80)
        
        category_schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["emergency", "appointment_create", "administrative", "medical_certificate"]
                },
                "recap": {"type": "string"},
                "priority_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"]
                }
            },
            "required": ["category", "recap"]
        }
        
        # Test 3.1: Urgence
        self.test_case(
            "Catégorie: Emergency (dent cassée)",
            [
                {"role": "system", "content": "Catégorise les demandes médicales. Une dent cassée est une urgence."},
                {"role": "user", "content": "Motif: dent cassée suite à une chute"}
            ],
            category_schema,
            lambda r: r.get("category") == "emergency"
        )
        
        # Test 3.2: Administrative
        self.test_case(
            "Catégorie: Administrative (certificat)",
            [
                {"role": "system", "content": "Catégorise les demandes. Les certificats sont administratifs ou medical_certificate."},
                {"role": "user", "content": "Besoin d'un certificat médical pour le sport"}
            ],
            category_schema,
            lambda r: r.get("category") in ["medical_certificate", "administrative"]
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 4. TESTS DE CAS COMPLEXES
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 4: CAS COMPLEXES")
        print("="*80)
        
        # Test 4.1: Analyse conversation complète
        conversation_schema = {
            "type": "object",
            "properties": {
                "medical_motif": {"type": "string"},
                "urgency_detected": {"type": "boolean"},
                "final_category": {
                    "type": "string",
                    "enum": ["emergency", "appointment_create", "administrative"]
                },
                "localisation": {"type": "string"},
                "duree": {"type": "string"}
            },
            "required": ["medical_motif", "urgency_detected", "final_category"]
        }
        
        self.test_case(
            "Complexe: Analyse conversation multi-tours",
            [
                {"role": "system", "content": "Analyse une conversation médicale complète et extrais les informations clés."},
                {"role": "user", "content": """Conversation:
- Patient: "J'ai mal"
- Assistant: "Où avez-vous mal?"
- Patient: "Aux dents, en bas à droite"
- Assistant: "Depuis quand?"
- Patient: "3 jours, c'est insupportable"

Analyse cette conversation."""}
            ],
            conversation_schema,
            lambda r: "dent" in r.get("medical_motif", "").lower() and
                     r.get("urgency_detected") == True
        )
        
        # Test 4.2: Gestion réponse inadéquate
        invalid_response_schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "extracted_value": {"type": "string"},
                "explanation": {"type": "string"},
                "retry_needed": {"type": "boolean"}
            },
            "required": ["is_valid", "explanation"]
        }
        
        self.test_case(
            "Complexe: Gestion réponse inadéquate",
            [
                {"role": "system", "content": "Gère les réponses vagues ou inadéquates des patients."},
                {"role": "user", "content": "Question: Votre date de naissance? Réponse: euh... bah..."}
            ],
            invalid_response_schema,
            lambda r: r.get("is_valid") == False
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 5. TESTS DE PERFORMANCE
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("📋 PHASE 5: TESTS DE PERFORMANCE")
        print("="*80)
        
        # Test rapide avec schéma minimal
        perf_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"}
            },
            "required": ["status"]
        }
        
        perf_times = []
        for i in range(5):
            start = time.time()
            try:
                response = requests.post(
                    f"{API_URL}/v1/chat/completions",
                    headers=HEADERS,
                    json={
                        "messages": [{"role": "user", "content": "Réponds avec status ok"}],
                        "temperature": 0.01,
                        "max_tokens": 20,
                        "response_format": {"type": "json_object"},
                        "json_schema": perf_schema
                    },
                    timeout=10
                )
                elapsed = (time.time() - start) * 1000
                perf_times.append(elapsed)
                print(f"Test perf {i+1}: {elapsed:.0f}ms - Status: {response.status_code}")
            except Exception as e:
                print(f"Test perf {i+1}: Erreur - {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # RAPPORT FINAL
        # ═══════════════════════════════════════════════════════════════
        
        self.print_final_report(perf_times)
    
    def print_final_report(self, perf_times: List[float]):
        """Affiche le rapport final détaillé"""
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL - TESTS AVEC SCHÉMAS JSON FORCÉS")
        print("="*80)
        
        total_tests = self.passed + self.failed
        success_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n✅ Tests réussis: {self.passed}")
        print(f"❌ Tests échoués: {self.failed}")
        print(f"📈 Taux de réussite: {success_rate:.1f}%")
        
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
                categories[cat] = {'passed': 0, 'failed': 0, 'times': []}
            categories[cat]['passed'] += 1
            categories[cat]['times'].append(result['time_ms'])
        
        for cat, stats in categories.items():
            total = stats['passed']
            avg_cat_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            print(f"   {cat}: {stats['passed']} tests réussis (temps moy: {avg_cat_time:.0f}ms)")
        
        # Verdict VoxEngine
        print("\n🎯 VERDICT POUR VOXENGINE:")
        
        if success_rate >= 90 and avg_time < 3000:
            print("   ✅ API PRÊTE POUR PRODUCTION")
            print("   - Schémas JSON respectés")
            print("   - Performance acceptable (<3s)")
            print("   - Extraction de données fiable")
            print("   - Compatible avec les besoins VoxEngine")
        elif success_rate >= 70:
            print("   ⚠️  API FONCTIONNELLE MAIS À OPTIMISER")
            print("   - Taux de réussite correct mais perfectible")
            print("   - Vérifier les cas d'échec")
            print("   - Optimiser les prompts problématiques")
        else:
            print("   ❌ API NON PRÊTE")
            print(f"   - Taux de réussite insuffisant ({success_rate:.1f}%)")
            print("   - Nécessite des ajustements majeurs")
            print("   - Revoir l'implémentation du parsing JSON")
        
        # Recommandations spécifiques
        print("\n💡 RECOMMANDATIONS:")
        if avg_time > 2000:
            print("   - Performance à optimiser (cible: <2000ms)")
        if self.failed > 0:
            print("   - Analyser les logs des tests échoués")
            print("   - Vérifier la conformité des schémas JSON")
        print("   - Surveiller les métriques Prometheus en production")
        print("   - Implémenter un cache pour les requêtes fréquentes")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    tester = VoxEngineTestSuite()
    tester.run_all_tests()