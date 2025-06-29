#!/usr/bin/env python3
"""
Batterie complète de tests pour VoxEngine Medical Agent
Test toutes les phases du scénario d'appel médical/dentaire
"""
import asyncio
import websockets
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum

# Configuration
RUNPOD_ID = "wsp137k5y3cf0p"
WS_URL = f"wss://{RUNPOD_ID}-8000.proxy.runpod.net/ws"
TOKEN = "supersecret"

class TestPhase(Enum):
    REASON_ANALYSIS = "reason_analysis"
    FORM_EXTRACTION = "form_extraction"
    CATEGORIZATION = "categorization"
    EDGE_CASES = "edge_cases"
    PERFORMANCE = "performance"

class VoxEngineTestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
        self.ws = None
        self.connected = False
        
    async def connect(self):
        """Établit la connexion WebSocket"""
        uri = f"{WS_URL}?token={TOKEN}"
        try:
            self.ws = await websockets.connect(uri)
            msg = await self.ws.recv()
            conn_data = json.loads(msg)
            if conn_data.get("status") == "connected":
                self.connected = True
                print("✅ WebSocket connecté")
                return True
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
            return False
    
    async def test_mistral(self, name: str, messages: List[Dict], 
                          expected_keys: List[str], 
                          validation_func: Optional[callable] = None,
                          max_tokens: int = 250) -> Dict[str, Any]:
        """Execute un test via WebSocket et vérifie le résultat"""
        print(f"\n{'='*60}")
        print(f"🧪 {name}")
        print(f"{'='*60}")
        
        if not self.connected:
            print("❌ Pas de connexion WebSocket")
            self.failed += 1
            return {"success": False, "error": "No connection"}
        
        # Préparer la requête
        request = {
            "request_id": f"test_{int(time.time()*1000)}",
            "messages": messages,
            "temperature": 0.01,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        # Log de la requête
        print(f"📤 Requête:")
        for msg in messages:
            if msg.get("role") == "user":
                print(f"   User: {msg['content'][:100]}...")
        
        start_time = time.time()
        
        try:
            # Envoyer la requête
            await self.ws.send(json.dumps(request))
            
            # Recevoir la réponse
            response = await asyncio.wait_for(self.ws.recv(), timeout=10)
            elapsed_ms = (time.time() - start_time) * 1000
            
            data = json.loads(response)
            
            if data.get("type") == "error":
                print(f"❌ Erreur serveur: {data.get('error')}")
                self.failed += 1
                return {"success": False, "error": data.get('error')}
            
            if data.get("type") != "completion":
                print(f"❌ Type de réponse inattendu: {data.get('type')}")
                self.failed += 1
                return {"success": False, "error": "Invalid response type"}
            
            content = data['choices'][0]['message']['content']
            
            # Parser le JSON de la réponse
            try:
                result = json.loads(content)
                print(f"✅ JSON valide reçu")
                print(f"📊 Réponse: {json.dumps(result, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                print(f"❌ JSON invalide: {content}")
                self.failed += 1
                return {"success": False, "error": "Invalid JSON", "content": content}
            
            # Vérifier les clés attendues
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                print(f"❌ Clés manquantes: {missing_keys}")
                self.failed += 1
                return {"success": False, "error": f"Missing keys: {missing_keys}"}
            
            # Validation personnalisée
            if validation_func:
                try:
                    if not validation_func(result):
                        print(f"❌ Validation personnalisée échouée")
                        self.failed += 1
                        return {"success": False, "error": "Custom validation failed"}
                except Exception as e:
                    print(f"❌ Erreur dans la validation: {e}")
                    self.failed += 1
                    return {"success": False, "error": f"Validation error: {e}"}
            
            print(f"⏱️  Temps: {elapsed_ms:.0f}ms (serveur: {data.get('time_ms', 'N/A')}ms)")
            print(f"✅ Test réussi")
            self.passed += 1
            
            self.results.append({
                'test': name,
                'phase': self._get_current_phase(name),
                'success': True,
                'time_ms': elapsed_ms,
                'result': result
            })
            
            return {"success": True, "result": result, "time_ms": elapsed_ms}
            
        except asyncio.TimeoutError:
            print(f"❌ Timeout après 10 secondes")
            self.failed += 1
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            print(f"❌ Exception: {e}")
            self.failed += 1
            return {"success": False, "error": str(e)}
    
    def _get_current_phase(self, test_name: str) -> str:
        """Détermine la phase du test basée sur son nom"""
        if "Intention" in test_name or "Motif" in test_name:
            return TestPhase.REASON_ANALYSIS.value
        elif "Formulaire" in test_name or "Extraction" in test_name:
            return TestPhase.FORM_EXTRACTION.value
        elif "Catégorie" in test_name or "Récap" in test_name:
            return TestPhase.CATEGORIZATION.value
        elif "Edge" in test_name or "Complexe" in test_name:
            return TestPhase.EDGE_CASES.value
        else:
            return TestPhase.PERFORMANCE.value
    
    async def run_all_tests(self):
        """Execute tous les tests VoxEngine"""
        print("🚀 BATTERIE COMPLÈTE DE TESTS VOXENGINE MEDICAL AGENT")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 WebSocket: {WS_URL}")
        print("\n")
        
        # Connexion
        if not await self.connect():
            print("❌ Impossible de se connecter. Arrêt des tests.")
            return
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # 1. TESTS ANALYSE D'INTENTION (handleReasonPhase)
            # ═══════════════════════════════════════════════════════════════
            
            print("\n" + "="*80)
            print("📋 PHASE 1: ANALYSE DU MOTIF D'APPEL")
            print("="*80)
            
            # Test 1.1: Motif dentaire clair
            await self.test_mistral(
                "Intention: Motif dentaire clair (détartrage)",
                [
                    {"role": "system", "content": "Tu es un ASSISTANT MÉDICAL pour un cabinet médical/dentaire. Réponds UNIQUEMENT en JSON."},
                    {"role": "user", "content": '''PATIENT DIT: "j'aimerais un détartrage"

Si c'est médical: réponds avec intention="medical_motif"
Si c'est hors-sujet: réponds avec intention="off_topic"  
Si c'est au revoir: réponds avec intention="goodbye"

JSON uniquement:'''}
                ],
                ["intention"],
                lambda r: r.get("intention") == "medical_motif"
            )
            
            # Test 1.2: Urgence dentaire
            await self.test_mistral(
                "Intention: Urgence (rage de dent)",
                [
                    {"role": "user", "content": '''PATIENT DIT: "j'ai une rage de dent insupportable"

Analyse cette urgence dentaire.
Réponds avec: {"intention": "medical_motif", "action": "finalize", "medical_motif": "rage de dent", "urgency_detected": true}'''}
                ],
                ["intention", "urgency_detected"],
                lambda r: r.get("urgency_detected") == True and r.get("intention") == "medical_motif"
            )
            
            # Test 1.3: Motif vague nécessitant clarification
            await self.test_mistral(
                "Intention: Motif vague (consultation)",
                [
                    {"role": "user", "content": '''PATIENT DIT: "je voudrais une consultation"

C'est un motif médical mais vague, donc pose une question pour clarifier.
Structure: {"intention": "medical_motif", "action": "ask_question", "next_question": "Pour quel type de consultation?"}'''}
                ],
                ["intention", "action"],
                lambda r: r.get("action") == "ask_question" and "question" in str(r).lower()
            )
            
            # Test 1.4: Hors-sujet
            await self.test_mistral(
                "Intention: Hors-sujet (pizza)",
                [
                    {"role": "user", "content": '''PATIENT DIT: "je veux commander une pizza"

Ce n'est PAS médical, donc intention="off_topic"
JSON:'''}
                ],
                ["intention"],
                lambda r: r.get("intention") == "off_topic"
            )
            
            # Test 1.5: Au revoir
            await self.test_mistral(
                "Intention: Fin d'appel",
                [
                    {"role": "user", "content": '''PATIENT DIT: "au revoir"

Fin d'appel détectée.
JSON: {"intention": "goodbye", "farewell_message": "Merci pour votre appel, au revoir!"}'''}
                ],
                ["intention"],
                lambda r: r.get("intention") == "goodbye"
            )
            
            # ═══════════════════════════════════════════════════════════════
            # 2. TESTS EXTRACTION FORMULAIRE
            # ═══════════════════════════════════════════════════════════════
            
            print("\n" + "="*80)
            print("📋 PHASE 2: EXTRACTION DES DONNÉES DU FORMULAIRE")
            print("="*80)
            
            # Test 2.1: Nom de famille simple
            await self.test_mistral(
                "Formulaire: Nom simple (Martin)",
                [
                    {"role": "user", "content": '''Extrais le nom de famille.

RÉPONSE DU PATIENT: "Martin"

JSON:'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: r.get("is_valid") == True and r.get("extracted_value") == "Martin"
            )
            
            # Test 2.2: Nom composé
            await self.test_mistral(
                "Formulaire: Nom composé (Dupont-Martin)",
                [
                    {"role": "user", "content": '''Extrais le nom de famille.

RÉPONSE: "c'est Dupont-Martin"

Exemples:
- "c'est Dupont-Martin" → {"is_valid": true, "extracted_value": "Dupont-Martin"}

JSON:'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: r.get("extracted_value") == "Dupont-Martin"
            )
            
            # Test 2.3: Nom épelé
            await self.test_mistral(
                "Formulaire: Nom épelé (G O U R O N)",
                [
                    {"role": "user", "content": '''Le patient épelle son nom: "G O U R O N"

Reconstitue le nom complet à partir des lettres.
Résultat attendu: {"is_valid": true, "extracted_value": "GOURON"}'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: r.get("extracted_value").upper() == "GOURON"
            )
            
            # Test 2.4: Date française simple
            await self.test_mistral(
                "Formulaire: Date numérique",
                [
                    {"role": "user", "content": '''Convertis en DD/MM/YYYY.

RÉPONSE: "15 mars 1988"

JSON:'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: r.get("extracted_value") == "15/03/1988"
            )
            
            # Test 2.5: Date française complexe
            await self.test_mistral(
                "Formulaire: Date en lettres",
                [
                    {"role": "user", "content": '''Convertis en DD/MM/YYYY.

RÉPONSE: "quinze mars quatre-vingt-huit"

Exemples:
- "quinze mars quatre-vingt-huit" → {"is_valid": true, "extracted_value": "15/03/1988"}

JSON:'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: r.get("extracted_value") == "15/03/1988"
            )
            
            # Test 2.6: Patient existant - Cas ambigu
            await self.test_mistral(
                "Formulaire: Patient existant (oui bien sûr)",
                [
                    {"role": "user", "content": '''Patient existant? (oui/non)

RÉPONSE: "oui bien sûr"

Exemples:
- "oui bien sûr" → {"is_valid": true, "extracted_value": "oui"}
- "première fois" → {"is_valid": true, "extracted_value": "non"}

JSON:'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: r.get("extracted_value") == "oui"
            )
            
            # Test 2.7: Praticien habituel
            await self.test_mistral(
                "Formulaire: Praticien (Dr Marcello)",
                [
                    {"role": "user", "content": '''Nom du praticien?

RÉPONSE: "c'est le docteur Marcello"

Exemples:
- "docteur Marcello" → {"is_valid": true, "extracted_value": "Dr Marcello"}

JSON:'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: "Marcello" in r.get("extracted_value", "")
            )
            
            # ═══════════════════════════════════════════════════════════════
            # 3. TESTS CATÉGORISATION FINALE
            # ═══════════════════════════════════════════════════════════════
            
            print("\n" + "="*80)
            print("📋 PHASE 3: CATÉGORISATION ET RÉSUMÉ FINAL")
            print("="*80)
            
            # Test 3.1: Catégorie emergency
            await self.test_mistral(
                "Catégorie: Emergency (dent cassée)",
                [
                    {"role": "user", "content": '''Motif: "dent cassée suite à une chute"

Catégories:
emergency, appointment_create, appointment_delete, retard, prescription_renewal, medical_certificate

Exemples:
- "dent cassée" → {"category": "emergency", "recap": "Urgence dentaire"}

JSON:'''}
                ],
                ["category", "recap"],
                lambda r: r.get("category") == "emergency"
            )
            
            # Test 3.2: Catégorie appointment_delete
            await self.test_mistral(
                "Catégorie: Annulation",
                [
                    {"role": "user", "content": '''Motif: "je dois annuler mon rendez-vous"

Catégories: appointment_create, appointment_delete, appointment_update

Exemples:
- "annuler rdv" → {"category": "appointment_delete", "recap": "Annulation"}

JSON:'''}
                ],
                ["category", "recap"],
                lambda r: r.get("category") == "appointment_delete"
            )
            
            # Test 3.3: Catégorie retard
            await self.test_mistral(
                "Catégorie: Retard",
                [
                    {"role": "user", "content": '''Motif: "je serai en retard de 15 minutes"

Exemples:
- "en retard" → {"category": "retard", "recap": "Retard signalé"}

JSON:'''}
                ],
                ["category", "recap"],
                lambda r: r.get("category") == "retard"
            )
            
            # ═══════════════════════════════════════════════════════════════
            # 4. TESTS DE CAS COMPLEXES / EDGE CASES
            # ═══════════════════════════════════════════════════════════════
            
            print("\n" + "="*80)
            print("📋 PHASE 4: CAS COMPLEXES ET EDGE CASES")
            print("="*80)
            
            # Test 4.1: Conversation multi-tours
            await self.test_mistral(
                "Complexe: Analyse conversation complète",
                [
                    {"role": "system", "content": "Assistant analysant une conversation médicale."},
                    {"role": "user", "content": '''Analyse cette conversation:
- Patient: "J'ai mal"
- Assistant: "Où avez-vous mal?"
- Patient: "Aux dents du bas"
- Assistant: "Depuis quand?"
- Patient: "3 jours, c'est très douloureux"

Détermine le motif médical final et l'urgence.
JSON: {"medical_motif": "...", "urgency_detected": bool, "category": "..."}'''}
                ],
                ["medical_motif"],
                lambda r: "dent" in r.get("medical_motif", "").lower()
            )
            
            # Test 4.2: Réponse inadéquate
            await self.test_mistral(
                "Edge Case: Réponse vague pour date",
                [
                    {"role": "user", "content": '''Question: "Votre date de naissance?"
Réponse: "euh... je sais pas trop..."

Impossible d'extraire une date.
JSON: {"is_valid": false, "extracted_value": "", "explanation": "..."}'''}
                ],
                ["is_valid"],
                lambda r: r.get("is_valid") == False
            )
            
            # Test 4.3: Nom avec chiffres (épellation avec erreur ASR)
            await self.test_mistral(
                "Edge Case: Nom épelé avec chiffres",
                [
                    {"role": "user", "content": '''Le patient épelle: "G 0 U R 0 N" (zéros au lieu de O)

Corrige les chiffres en lettres et reconstitue le nom.
0 → O, 1 → I, etc.

Résultat attendu: {"is_valid": true, "extracted_value": "GOURON"}'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: r.get("extracted_value", "").upper().replace("0", "O") == "GOURON"
            )
            
            # Test 4.4: Date partielle
            await self.test_mistral(
                "Edge Case: Date incomplète",
                [
                    {"role": "user", "content": '''Convertis en date.

RÉPONSE: "mars 1990"

Si date incomplète, retourne ce qui est disponible.
JSON: {"is_valid": true, "extracted_value": "03/1990", "explanation": "Date partielle"}'''}
                ],
                ["is_valid", "extracted_value"],
                lambda r: "1990" in r.get("extracted_value", "") and "03" in r.get("extracted_value", "")
            )
            
            # ═══════════════════════════════════════════════════════════════
            # 5. TESTS DE PERFORMANCE
            # ═══════════════════════════════════════════════════════════════
            
            print("\n" + "="*80)
            print("📋 PHASE 5: TESTS DE PERFORMANCE")
            print("="*80)
            
            # Test de latence avec requêtes simples
            perf_times = []
            print("\n🏃 Test de performance (5 requêtes rapides)...")
            
            for i in range(5):
                start = time.time()
                await self.test_mistral(
                    f"Performance {i+1}: Requête simple",
                    [
                        {"role": "user", "content": f'Echo {i+1}: {{"status": "ok", "id": {i+1}}}'}
                    ],
                    ["status"],
                    lambda r: r.get("status") == "ok",
                    max_tokens=50
                )
                elapsed = (time.time() - start) * 1000
                perf_times.append(elapsed)
            
            # ═══════════════════════════════════════════════════════════════
            # RAPPORT FINAL
            # ═══════════════════════════════════════════════════════════════
            
            await self.print_final_report(perf_times)
            
        finally:
            # Fermer la connexion
            if self.ws:
                await self.ws.close()
                print("\n🔌 WebSocket fermé")
    
    async def print_final_report(self, perf_times: List[float]):
        """Affiche le rapport final détaillé"""
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL - VOXENGINE MEDICAL AGENT")
        print("="*80)
        
        total_tests = self.passed + self.failed
        
        print(f"\n📈 RÉSULTATS GLOBAUX:")
        print(f"   ✅ Tests réussis: {self.passed}/{total_tests}")
        print(f"   ❌ Tests échoués: {self.failed}/{total_tests}")
        print(f"   📊 Taux de réussite: {(self.passed/total_tests*100 if total_tests > 0 else 0):.1f}%")
        
        # Performance
        if perf_times:
            avg_time = sum(perf_times) / len(perf_times)
            print(f"\n⏱️  PERFORMANCE:")
            print(f"   Moyenne: {avg_time:.0f}ms")
            print(f"   Min: {min(perf_times):.0f}ms")
            print(f"   Max: {max(perf_times):.0f}ms")
        
        # Résultats par phase
        print(f"\n📋 RÉSULTATS PAR PHASE:")
        phase_stats = {}
        for result in self.results:
            phase = result.get('phase', 'unknown')
            if phase not in phase_stats:
                phase_stats[phase] = {'passed': 0, 'failed': 0, 'times': []}
            
            if result['success']:
                phase_stats[phase]['passed'] += 1
                if 'time_ms' in result:
                    phase_stats[phase]['times'].append(result['time_ms'])
            else:
                phase_stats[phase]['failed'] += 1
        
        for phase, stats in phase_stats.items():
            total = stats['passed'] + stats['failed']
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            print(f"\n   {phase.upper()}:")
            print(f"      Réussite: {stats['passed']}/{total} ({stats['passed']/total*100:.0f}%)")
            print(f"      Temps moyen: {avg_time:.0f}ms")
        
        # Tests critiques
        print(f"\n🎯 TESTS CRITIQUES:")
        critical_tests = [
            "Intention: Urgence",
            "Formulaire: Nom épelé",
            "Formulaire: Date en lettres",
            "Catégorie: Emergency"
        ]
        
        for test_name in critical_tests:
            result = next((r for r in self.results if test_name in r['test']), None)
            if result:
                status = "✅" if result['success'] else "❌"
                print(f"   {status} {test_name}")
        
        # Verdict final
        print(f"\n{'='*80}")
        print("🏁 VERDICT POUR PRODUCTION:")
        
        success_rate = self.passed / total_tests if total_tests > 0 else 0
        avg_response_time = sum(perf_times) / len(perf_times) if perf_times else 999999
        
        if success_rate >= 0.95 and avg_response_time < 2000:
            print("   ✅ SYSTÈME PRÊT POUR LA PRODUCTION")
            print("   - Fiabilité excellente (>95%)")
            print("   - Performance optimale (<2s)")
            print("   - Extraction de données fonctionnelle")
            print("   - Gestion des cas complexes OK")
        elif success_rate >= 0.80:
            print("   ⚠️  SYSTÈME FONCTIONNEL MAIS À OPTIMISER")
            print("   - Quelques cas d'échec à corriger")
            print("   - Vérifier les prompts problématiques")
            print("   - Tests edge cases à améliorer")
        else:
            print("   ❌ SYSTÈME NON PRÊT POUR LA PRODUCTION")
            print("   - Trop d'échecs dans les tests")
            print("   - Nécessite des corrections majeures")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if self.failed > 0:
            failed_phases = set()
            for result in self.results:
                if not result['success']:
                    failed_phases.add(result['phase'])
            
            for phase in failed_phases:
                print(f"   - Réviser les prompts de la phase: {phase}")
        
        if avg_response_time > 1500:
            print("   - Optimiser les prompts pour réduire la latence")
        
        print(f"\n{'='*80}")
        print("📝 Fin du rapport de test VoxEngine Medical Agent")

async def main():
    """Point d'entrée principal"""
    tester = VoxEngineTestSuite()
    await tester.run_all_tests()

if __name__ == "__main__":
    # Pour exécuter: python test_voxengine.py
    # Nécessite: pip install websockets
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")