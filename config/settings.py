import os

# === Configuracao do PostgreSQL ===
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "rag_database")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# === Configuracao dos Modelos ===
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

print(f"✅ Configuration loaded - LLM: {LLM_MODEL} | Embedding: {EMBEDDING_MODEL}")

# === Configuracoes de Chunking ===
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "fixed")

# === Configuracoes de Busca Vetorial ===
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # all-MiniLM-L6-v2
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# === Configuracoes de Pastas ===
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "documents")
INDICES_FOLDER = os.getenv("INDICES_FOLDER", "Indices")

# === Configuracoes do Ollama ===
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))

# Validacao basica das configuracoes
def validate_config():
    """Valida configuracoes criticas"""
    required_configs = {
        'DB_HOST': DB_HOST,
        'DB_NAME': DB_NAME, 
        'DB_USER': DB_USER,
        'DB_PASSWORD': DB_PASSWORD
    }
    
    for key, value in required_configs.items():
        if not value:
            raise ValueError(f"❌ Missing required configuration: {key}")
    
    print("✅ All required configurations are set")

# Valida automaticamente ao importar
validate_config()