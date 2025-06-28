#!/bin/bash

# Configuration
API_URL="https://opls0sp8kal7rd-8000.proxy.runpod.net"
API_TOKEN="supertoken"

echo "🧪 TESTS VOXENGINE - MISTRAL 7B API"
echo "=================================="
echo ""

# Fonction pour mesurer le temps
measure_time() {
    local start=$(date +%s%N)
    "$@"
    local end=$(date +%s%N)
    local elapsed=$(( ($end - $start) / 1000000 ))
    echo "⏱️  Temps de réponse: ${elapsed}ms"
}

# Test 1: Analyse d'intention utilisateur (JSON obligatoire)
echo "📋 TEST 1: Analyse d'intention médicale"
echo "---------------------------------------"
measure_time curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "system",
      "content": "Tu es un ASSISTANT VOICEMAIL INTELLIGENT pour un cabinet médical. Réponds UNIQUEMENT en JSON valide."
    }, {
      "role": "user", 
      "content": "PATIENT: \"j'\''ai mal aux dents depuis 3 jours\"\nDétecte intention (medical_motif|off_topic|goodbye) et action (ask_question|finalize)."
    }],
    "temperature": 0.01,
    "max_tokens": 200,
    "response_format": {"type": "json_object"}
  }' | jq .

echo ""

# Test 2: Extraction de formulaire - Nom de famille
echo "📋 TEST 2: Extraction nom de famille"
echo "------------------------------------"
measure_time curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "system",
      "content": "Expert extraction données formulaire médical. JSON strict uniquement."
    }, {
      "role": "user",
      "content": "RÉPONSE: \"Oui bien sûr mon nom de famille c'\''est corniqué\"\nExtraire le nom de famille.\nRéponds: {\"is_valid\": bool, \"extracted_value\": \"NOM\", \"explanation\": \"...\"}"
    }],
    "temperature": 0.01,
    "max_tokens": 150,
    "response_format": {"type": "json_object"}
  }' | jq .

echo ""

# Test 3: Extraction date de naissance française
echo "📋 TEST 3: Extraction date de naissance"
echo "---------------------------------------"
measure_time curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Convertis \"Le 7 septembre mille-neuf-cent-quatre-vingt-douze\" en format DD/MM/YYYY. Réponds uniquement le format date ou AUCUNE_DATE."
    }],
    "temperature": 0.01,
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
  }' | jq .

echo ""

# Test 4: Récapitulatif et catégorisation
echo "📋 TEST 4: Catégorisation finale"
echo "--------------------------------"
measure_time curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "system",
      "content": "Analyse conversation médicale. Catégories: appointment_create|emergency|prescription_renewal|administrative|follow_up"
    }, {
      "role": "user",
      "content": "Motif: \"détartrage urgent\"\nCatégorise et résume.\nJSON: {\"recap\": \"...\", \"category\": \"...\"}"
    }],
    "temperature": 0.01,
    "max_tokens": 150,
    "response_format": {"type": "json_object"}
  }' | jq .

echo ""

# Test 5: Détection hors-sujet
echo "📋 TEST 5: Détection hors-sujet"
echo "-------------------------------"
measure_time curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "PATIENT: \"stage de trampoline\"\nDétection obligatoire hors-sujet si pas médical.\nJSON: {\"intention\": \"medical_motif|off_topic|goodbye\"}"
    }],
    "temperature": 0.01,
    "max_tokens": 100,
    "response_format": {"type": "json_object"}
  }' | jq .

echo ""

# Test 6: Validation patient existant
echo "📋 TEST 6: Validation patient existant"
echo "--------------------------------------"
measure_time curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Réponse: \"oui bien sûr\"\nLa personne est-elle déjà patiente?\nJSON: {\"is_valid\": true, \"extracted_value\": \"oui|non\"}"
    }],
    "temperature": 0.01,
    "max_tokens": 100,
    "response_format": {"type": "json_object"}
  }' | jq .

echo ""
echo "✅ Tests terminés!"