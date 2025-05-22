# rag.py
import chromadb
from chromadb.utils import embedding_functions
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from typing import List, Dict, Any
from config import Config

nltk.download('punkt')

def _init_(self, config: Config):
    self.config = config
    
    # Ensure NLTK resources are available
    required_resources = ['punkt', 'punkt_tab']
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)
    
    self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
class RAGManager:
    def __init__(self, config: Config):
        self.config = config
        
        # Ensure NLTK resources are available
        required_resources = ['punkt', 'punkt_tab']
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
    
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into semantic units using sentence tokenization."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.config.CHUNK_SIZE:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Add a document to the vector store."""
        chunks = self.chunk_text(content)
        
        # Create unique IDs for each chunk
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Add metadata to each chunk
        chunk_metadata = [{
            **metadata,
            "chunk_index": i,
            "doc_id": doc_id
        } for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=chunk_metadata
        )
        
        return len(chunks)

    def get_relevant_chunks(self, query: str, chat_history: List[Dict] = None) -> str:
        """Get relevant chunks for a query, considering chat history."""
        # Combine recent chat history with query for better context
        if chat_history:
            recent_messages = chat_history[-3:]  # Consider last 3 messages
            context_query = " ".join([msg["content"] for msg in recent_messages] + [query])
        else:
            context_query = query
        
        results = self.collection.query(
            query_texts=[context_query],
            n_results=self.config.MAX_CONTEXT_CHUNKS,
            include=["documents", "metadatas", "distances"]
        )
        
        # Sort chunks by relevance (distance)
        chunks_with_scores = list(zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ))
        chunks_with_scores.sort(key=lambda x: x[1])  # Sort by distance
        
        # Format context with source information
        formatted_chunks = []
        for chunk, distance, metadata in chunks_with_scores:
            source_info = f"\nSource: {metadata['filename']}, Relevance: {1 - distance:.2f}"
            formatted_chunks.append(f"{chunk}{source_info}")
        
        return "\n\n".join(formatted_chunks)

