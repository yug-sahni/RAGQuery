import requests
import json
from typing import Dict, List
import subprocess
import time


class OllamaLLM:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.ensure_ollama_running()
        self.ensure_model_available()
    
    def ensure_ollama_running(self):
        """Check if Ollama is running, start if needed"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is running")
                return
        except:
            pass
        
        print("⚠️ Ollama not running. Please start Ollama:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Run: ollama serve")
        raise Exception("Ollama is not running")
    
    def ensure_model_available(self):
        """Ensure the model is downloaded"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get('models', [])
            
            if not any(model['name'].startswith(self.model_name) for model in models):
                print(f"📥 Downloading {self.model_name}...")
                subprocess.run(['ollama', 'pull', self.model_name], check=True)
                print(f"✅ {self.model_name} downloaded successfully")
        except Exception as e:
            print(f"Error checking/downloading model: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

class HuggingFaceLLM:
    """Backup option using Hugging Face transformers"""
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        from transformers import pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=-1  # Use CPU
        )
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using Hugging Face model"""
        try:
            result = self.generator(
                prompt,
                max_length=max_tokens,
                num_return_sequences=1,
                temperature=0.1,
                pad_token_id=50256
            )
            return result[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
