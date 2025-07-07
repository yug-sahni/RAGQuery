import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_ollama():
    """Setup instructions for Ollama"""
    print("🚀 Setting up Ollama for free LLM inference:")
    print("1. Visit https://ollama.ai and download Ollama")
    print("2. Install Ollama on your system")
    print("3. Run: ollama serve")
    print("4. Run: ollama pull llama3.1:8b")
    print("5. Your system is ready!")

def create_directories():
    """Create necessary directories"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

if __name__ == "__main__":
    print("🔧 Setting up Document Q&A System...")
    
    create_directories()
    print("✅ Created directories")
    
    install_requirements()
    print("✅ Installed requirements")
    
    setup_ollama()
    print("✅ Setup complete!")
    
    print("\n🚀 To run the application:")
    print("streamlit run app.py")
