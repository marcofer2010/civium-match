"""
Configuração de logging para o Civium Match Service
"""

import logging
import sys
from datetime import datetime
from typing import Optional

from app.config import settings


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura logger com formato simples e níveis apropriados.
    
    Args:
        name: Nome do logger
        log_file: Arquivo de log opcional
        
    Returns:
        Logger configurado
    """
    
    # Criar logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    # Formatter simples e legível
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (se especificado)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Se não conseguir criar arquivo, continua só com console
            pass
    
    # Evitar propagação para root logger
    logger.propagate = False
    
    return logger


def get_request_logger() -> logging.Logger:
    """Logger específico para requisições HTTP."""
    return setup_logger("civium-match.requests")


def get_performance_logger() -> logging.Logger:
    """Logger específico para métricas de performance."""
    return setup_logger("civium-match.performance") 