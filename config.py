# config.py
import os
from dataclasses import dataclass, field # Import field for default factory
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
# Create a file named .env in the same directory as config.py
# Add the following line to the .env file:
# GEMINI_API_KEY=your_actual_google_ai_api_key
load_dotenv()

@dataclass
class Config:
    # --- Database ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://chatbotuser:pass1234@127.0.0.1:5432/ragchatbot")

    # --- RAG / Vector Store ---
    CHROMA_PATH: Path = field(default_factory=lambda: Path("chroma_db"))
    CHROMA_COLLECTION_NAME: str = os.getenv('CHROMA_COLLECTION_NAME', 'medical_docs') # Added Collection Name
    EMBEDDING_MODEL_NAME: str = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2') # Added Embedding Model
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_CONTEXT_CHUNKS: int = 3

    # --- File Uploads ---
    UPLOAD_TEXT: Path = field(default_factory=lambda: Path(__file__).parent / "uploads/text")
    UPLOAD_IMAGES: Path = field(default_factory=lambda: Path(__file__).parent / "uploads/images")

    # --- LLM APIs ---
    OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
    # --- Google Gemini API Key (Loaded from Environment Variable) ---
    GEMINI_API_KEY: str | None = os.getenv('GEMINI_API_KEY') # Use str | None for type hinting

    # --- STT (Whisper doesn't need a model path here, loaded in app2.py) ---
    STT_TIMEOUT: int = 60  # Default STT timeout in seconds
    # VOSK_MODEL_PATH is removed as Whisper is used now

    # --- Directory Creation (optional: can be done here or in app startup) ---
    # This method can be called after creating a Config instance
    def create_dirs(self):
        """Creates necessary upload directories if they don't exist."""
        print(f"Ensuring directory exists: {self.UPLOAD_TEXT}")
        self.UPLOAD_TEXT.mkdir(parents=True, exist_ok=True)
        print(f"Ensuring directory exists: {self.UPLOAD_IMAGES}")
        self.UPLOAD_IMAGES.mkdir(parents=True, exist_ok=True)
        print(f"Ensuring directory exists: {self.CHROMA_PATH}")
        self.CHROMA_PATH.mkdir(parents=True, exist_ok=True) # Also ensure Chroma path exists

# Example of how to ensure directories are created when the app starts
# You would typically do this once in your main application file (e.g., app2.py)
# after importing and creating the config object:
#
# from config import Config
# config = Config()
# config.create_dirs() # Call this early in your app setup

# --- Check if Gemini Key is set ---
_temp_config = Config() # Create temporary instance to check key
if not _temp_config.GEMINI_API_KEY:
    print("\n" + "*"*60)
    print("WARNING: GEMINI_API_KEY environment variable is not set.")
    print("Prescription image upload functionality will not work.")
    print("Please create a '.env' file in the project root containing:")
    print("GEMINI_API_KEY=your_actual_google_ai_api_key")
    print("*"*60 + "\n")
del _temp_config # Clean up temporary instance