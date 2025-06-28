#!/usr/bin/env python3
"""
Script pour tester l'API Mistral 7B avec authentification
"""
import requests
import json

# Configuration
API_URL = "https://rhyx84x4k3pri9-8000.proxy.runpod.net"
API_TOKEN = "supersecret"  # Ton token Bearer

def test_health():
    """Tester l'endpoint de santé (pas d'auth nécessaire)"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))

def test_chat_completion():
    """Tester l'endpoint de chat completion avec auth"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    data = {
        "model": "mistral-7b-gguf",
        "messages": [
            {"role": "system", "content": "Tu es un assistant IA utile."},
            {"role": "user", "content": "Explique ce qu'est RunPod en une phrase."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    response = requests.post(
        f"{API_URL}/v1/chat/completions",
        json=data,
        headers=headers
    )
    
    print("\nChat Completion:")
    print(json.dumps(response.json(), indent=2))

def test_json_format():
    """Tester la réponse en format JSON"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    data = {
        "model": "mistral-7b-gguf",
        "messages": [
            {"role": "user", "content": "Extrais la date de naissance: 'Je suis né le 06/03/1988'. Réponds en JSON avec is_valid, extracted_value et explanation."}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 150
    }
    
    response = requests.post(
        f"{API_URL}/v1/chat/completions",
        json=data,
        headers=headers
    )
    
    print("\nJSON Format Test:")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    # Vérifier que le contenu est bien du JSON
    if result.get("choices"):
        content = result["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            print("\nParsed JSON content:")
            print(json.dumps(parsed, indent=2))
        except:
            print("\nCould not parse content as JSON")

def test_unauthorized():
    """Tester sans token (devrait échouer)"""
    response = requests.post(
        f"{API_URL}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]}
    )
    
    print("\nUnauthorized test:")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Tester la santé
    test_health()
    
    # Tester sans auth (devrait échouer)
    test_unauthorized()
    
    # Tester le chat avec auth
    test_chat_completion()
    
    # Tester le format JSON
    test_json_format()