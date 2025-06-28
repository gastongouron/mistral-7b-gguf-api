#!/usr/bin/env python3
"""
Script pour tester l'API Mistral 7B
"""
import requests
import json

# URL de ton pod RunPod (remplace par ton URL)
API_URL = "https://YOUR-POD-ID-8000.proxy.runpod.net"

def test_health():
    """Tester l'endpoint de santé"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))

def test_chat_completion():
    """Tester l'endpoint de chat completion"""
    data = {
        "model": "mistral-7b",
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
        headers={"Content-Type": "application/json"}
    )
    
    print("\nChat Completion:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Tester la santé
    test_health()
    
    # Tester le chat
    test_chat_completion()