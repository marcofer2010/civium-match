"""
Configurações da aplicação Civium Match Service
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configurações da aplicação."""
    
    # Configurações básicas
    DEBUG: bool = Field(default=False, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    
    # Servidor
    HOST: str = Field(default="0.0.0.0", description="Host do servidor")
    PORT: int = Field(default=8001, description="Porta do servidor")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"], 
        description="Origens permitidas para CORS"
    )
    
    # Database (se necessário)
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="URL do banco de dados PostgreSQL"
    )
    
    # FAISS Configuration
    FAISS_DATA_DIR: str = Field(
        default="data",
        description="Diretório para armazenar dados FAISS"
    )
    
    FAISS_INDEX_TYPE: str = Field(
        default="IVF1024,Flat",
        description="Tipo de índice FAISS"
    )
    
    # Parâmetros de match
    DEFAULT_MATCH_THRESHOLD: float = Field(
        default=0.4,
        description="Threshold padrão para match"
    )
    
    DEFAULT_TOP_K: int = Field(
        default=10,
        description="Número padrão de resultados retornados"
    )
    
    # Embedding configuration
    EMBEDDING_DIMENSION: int = Field(
        default=512,
        description="Dimensão dos embeddings"
    )
    
    # Performance
    MAX_BATCH_SIZE: int = Field(
        default=100,
        description="Tamanho máximo do batch para processamento"
    )
    
    # Cache settings
    ENABLE_CACHE: bool = Field(
        default=True,
        description="Habilitar cache de resultados"
    )
    
    CACHE_TTL: int = Field(
        default=3600,
        description="TTL do cache em segundos"
    )
    
    # Monitoramento
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Habilitar coleta de métricas"
    )
    
    # Security (se necessário)
    SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Chave secreta para JWT"
    )
    
    API_TOKEN: Optional[str] = Field(
        default=None,
        description="Token de API para autenticação"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Instância global das configurações
settings = Settings() 