"""
Setup script to install all dependencies and download required models.
Run: python setup.py
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 MeetingMinutesAI Setup Script")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("\n⚠️  Some packages failed to install. Continuing anyway...")
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy language model"):
        print("\n⚠️  spaCy model download failed. You may need to install it manually.")
    
    # Download NLTK data
    nltk_script = """
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        nltk.download('punkt', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
"""
    if not run_command(f'python -c "{nltk_script}"', "Downloading NLTK data"):
        print("\n⚠️  NLTK data download failed. You may need to download it manually.")
    
    # Verify installations
    print(f"\n{'='*60}")
    print("🔍 Verifying installations...")
    print(f"{'='*60}")
    
    packages_to_check = [
        ("streamlit", "Streamlit"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("nltk", "NLTK"),
        ("spacy", "spaCy"),
        ("sklearn", "scikit-learn"),
    ]
    
    failed = []
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            failed.append(name)
    
    # Optional packages
    print("\n📋 Optional packages:")
    optional_packages = [
        ("whisper", "OpenAI Whisper"),
        ("whisperx", "WhisperX"),
        ("pyannote", "pyannote.audio"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - Not installed (optional)")
    
    print(f"\n{'='*60}")
    if failed:
        print("❌ Some required packages are missing!")
        print("Please install them manually: pip install " + " ".join(failed))
        sys.exit(1)
    else:
        print("✅ All required packages are installed!")
        print(f"{'='*60}")
        print("\n🎉 Setup complete!")
        print("\nTo run the application:")
        print("  streamlit run app.py --server.port 5000")
        print("\nNote: For pyannote.audio speaker diarization, you need to:")
        print("  1. Create Hugging Face account")
        print("  2. Run: huggingface-cli login")
        print("  3. Accept terms for pyannote models")

if __name__ == "__main__":
    main()

