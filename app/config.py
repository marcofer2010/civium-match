"""
Configurações da aplicação Civium Match Service
"""

import os


class Settings:
    """Configurações simples da aplicação."""
    
    # Servidor
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", "8002"))
    
    # CORS
    ALLOWED_ORIGINS = ["*"]
    
    # FAISS Configuration
    FAISS_DATA_DIR = "collections"
    
    # Parâmetros de match
    DEFAULT_MATCH_THRESHOLD = float(os.getenv("DEFAULT_MATCH_THRESHOLD", "0.4"))
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
    
    # Embedding configuration
    EMBEDDING_DIMENSION = 512
    
    # Debug
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# Instância global das configurações
settings = Settings() 