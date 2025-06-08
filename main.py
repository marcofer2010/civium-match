#!/usr/bin/env python3
"""
Civium Match Service - Serviço de Match FAISS
Responsável por operações de match/busca facial usando FAISS
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.match_service import MatchService
from app.models.api_models import (
    HealthResponse,
    SmartMatchRequest,
    SmartMatchResponse,
    AddFaceRequest,
    AddFaceResponse
)
from app.utils.logger import setup_logger

# Configurar logging
logger = setup_logger("civium-match")

# Instância global do serviço de match
match_service: Optional[MatchService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciar ciclo de vida da aplicação."""
    global match_service
    
    logger.info("🚀 Inicializando Civium Match Service...")
    
    try:
        # Inicializar serviço de match
        match_service = MatchService()
        await match_service.initialize()
        
        logger.info("✅ Civium Match Service inicializado com sucesso!")
        yield
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar serviço: {e}")
        raise
    finally:
        logger.info("🛑 Finalizando Civium Match Service...")
        if match_service:
            await match_service.cleanup()

# Criar aplicação FastAPI
app = FastAPI(
    title="Civium Match Service",
    description="Serviço de match facial inteligente com busca em cascata",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        service_status = "healthy" if match_service and match_service.is_ready else "initializing"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            service="civium-match",
            version="1.0.0",
            details={
                "match_service": service_status,
                "collections_loaded": len(match_service.collection_manager.collections_cache) if match_service else 0
            }
        )
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/api/smart-match", response_model=SmartMatchResponse)
async def smart_match_faces(request: SmartMatchRequest):
    """
    Realizar match inteligente com busca em cascata.
    
    Etapas:
    1. Buscar em collections 'known' (federada se camera_shared=True)
       - Se camera_shared=True: busca em TODAS as collections 'known' públicas + própria
       - Se camera_shared=False: busca apenas na própria collection 'known'
    2. Se não encontrar e search_unknown=True, buscar na collection 'unknown' da própria empresa
    3. Se não encontrar e auto_register=True, cadastrar na collection 'unknown' da própria empresa
    
    Args:
        request: Dados da requisição com embedding e parâmetros de controle
        
    Returns:
        Resultado inteligente do match com informações detalhadas da busca
    """
    if not match_service or not match_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Match service não está pronto"
        )
    
    try:
        logger.info(f"🧠 Smart match: company={request.company_id}, type={request.company_type}, "
                   f"shared={request.camera_shared}, search_unknown={request.search_unknown}, "
                   f"auto_register={request.auto_register}")
        
        start_time = time.time()
        
        # Realizar smart match
        result = await match_service.smart_match(
            embedding=request.embedding,
            company_id=request.company_id,
            company_type=request.company_type,
            camera_shared=request.camera_shared,
            search_unknown=request.search_unknown,
            auto_register=request.auto_register,
            threshold=request.threshold,
            top_k=request.top_k,
            metadata=request.metadata
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Smart match completed in {elapsed_time:.1f}ms - "
                   f"result_type={result.result_type}, collections_searched={result.total_collections_searched}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Smart match error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no smart match: {str(e)}"
        )

@app.post("/api/faces", response_model=AddFaceResponse)
async def add_face(request: AddFaceRequest):
    """
    Adicionar uma face ao sistema.
    
    A face será automaticamente adicionada à collection apropriada baseada em:
    - company_id: ID da empresa/tenant
    - company_type: 'public_org' ou 'private'
    - collection_type: 'known' ou 'unknown'
    
    Args:
        request: Dados da face a ser adicionada
        
    Returns:
        Informações da face adicionada
    """
    if not match_service or not match_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Match service não está pronto"
        )
    
    try:
        logger.info(f"👤 Adding face: company={request.company_id}, type={request.company_type}, "
                   f"collection={request.collection_type}, person_id={request.person_id}")
        
        start_time = time.time()
        
        face_id = await match_service.add_face_to_collection(
            company_id=request.company_id,
            company_type=request.company_type,
            collection_type=request.collection_type,
            embedding=request.embedding,
            person_id=request.person_id,
            metadata=request.metadata
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Face added in {elapsed_time:.1f}ms: {face_id}")
        
        return AddFaceResponse(
            face_id=face_id,
            company_id=request.company_id,
            collection_type=request.collection_type,
            person_id=request.person_id,
            added_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"❌ Add face error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao adicionar face: {str(e)}"
        )

@app.get("/api/stats")
async def get_service_stats():
    """Obter estatísticas do serviço."""
    if not match_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Match service não está pronto"
        )
    
    return await match_service.get_stats()

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global para exceções não tratadas."""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=settings.DEBUG,
        log_level="info"
    ) 