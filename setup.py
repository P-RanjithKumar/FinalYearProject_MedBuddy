# setup.py
import nltk
import os
from pathlib import Path
from config import Config
import ssl

def setup_environment():
    """Set up all necessary resources and directories for the RAG chatbot."""
    print("Setting up RAG chatbot environment...")
    
    # Handle SSL certificate issues that sometimes occur with NLTK downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download required NLTK data
    print("Downloading NLTK resources...")
    required_nltk_resources = ['punkt', 'punkt_tab']
    
    for resource in required_nltk_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"✓ {resource} already downloaded")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)
    
    # Create necessary directories
    config = Config()
    directories = [
        config.UPLOAD_FOLDER,
        config.CHROMA_PATH
    ]
    
    for directory in directories:
        if not directory.exists():
            print(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    print("\nSetup completed!")
    print("\nVerifying NLTK resources...")
    try:
        from nltk.tokenize import sent_tokenize
        test_text = "This is a test sentence. This is another one."
        sent_tokenize(test_text)
        print("✓ NLTK tokenizer is working correctly")
    except Exception as e:
        print(f"! Error testing NLTK tokenizer: {str(e)}")
        print("Please try running these commands manually in a Python console:")
        print("import nltk")
        print("nltk.download('punkt')")
        print("nltk.download('punkt_tab')")

if __name__ == "__main__":
    setup_environment()