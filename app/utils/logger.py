"""
Configuração de logging para o Civium Match Service
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from pythonjsonlogger import jsonlogger

from app.config import settings


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura logger com formato JSON e níveis apropriados.
    
    Args:
        name: Nome do logger
        log_file: Arquivo de log opcional
        
    Returns:
        Logger configurado
    """
    
    # Criar logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    # Formatter para logs estruturados
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Evitar propagação para root logger
    logger.propagate = False
    
    return logger


def get_request_logger() -> logging.Logger:
    """Logger específico para requisições HTTP."""
    return setup_logger("civium-match.requests", "logs/requests.log")


def get_performance_logger() -> logging.Logger:
    """Logger específico para métricas de performance."""
    return setup_logger("civium-match.performance", "logs/performance.log") 