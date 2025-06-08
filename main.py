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
    # Modelos para paths
    AddFaceByPathRequest,
    AddFaceByPathResponse,
    RemoveFaceByPathRequest,
    RemoveFaceByPathResponse
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
    1. Parse do collection_path para extrair company_type, company_id
    2. Buscar em collections 'known' (federada se camera_shared=True e public)
       - Se camera_shared=True e company_type=public: busca em TODAS as collections 'known' públicas + própria
       - Caso contrário: busca apenas na própria collection 'known'
    3. Se não encontrar e search_unknown=True, buscar na collection 'unknown' da própria empresa
    4. Se não encontrar e auto_register=True, cadastrar na collection 'unknown' da própria empresa
    
    Args:
        request: Dados da requisição com embedding, collection_path e parâmetros de controle
        
    Returns:
        Resultado inteligente do match com informações detalhadas da busca
    """
    if not match_service or not match_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Match service não está pronto"
        )
    
    try:
        # Parse do collection_path
        company_type, company_id, _ = match_service._parse_collection_path(request.collection_path)
        
        logger.info(f"🧠 Smart match: path={request.collection_path}, "
                   f"shared={request.camera_shared}, search_unknown={request.search_unknown}, "
                   f"auto_register={request.auto_register}")
        
        start_time = time.time()
        
        # Realizar smart match
        result = await match_service.smart_match(
            embedding=request.embedding,
            company_id=company_id,
            company_type=company_type,
            camera_shared=request.camera_shared,
            search_unknown=request.search_unknown,
            auto_register=request.auto_register,
            threshold=request.threshold,
            top_k=request.top_k
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

@app.get("/api/stats")
async def get_service_stats():
    """Obter estatísticas do serviço."""
    if not match_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Match service não está pronto"
        )
    
    return await match_service.get_stats()

# ==========================================
# NOVOS ENDPOINTS COM PATHS DE COLLECTIONS
# ==========================================

@app.post("/api/v2/faces", response_model=AddFaceByPathResponse)
async def add_face_by_path(request: AddFaceByPathRequest):
    """
    Adicionar face usando path da collection.
    
    O path deve ter o formato: company_type/company_id/collection_type
    Exemplo: "private/123/known" ou "public/456/unknown"
    
    Args:
        request: Dados da face com path da collection
        
    Returns:
        Informações da face adicionada
    """
    if not match_service or not match_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Match service não está pronto"
        )
    
    try:
        logger.info(f"👤 Adding face by path: {request.collection_path}")
        
        start_time = time.time()
        
        index_position = await match_service.add_face_by_path(
            collection_path=request.collection_path,
            embedding=request.embedding
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Face added by path in {elapsed_time:.1f}ms: {request.collection_path}[{index_position}]")
        
        return AddFaceByPathResponse(
            index_position=index_position,
            collection_path=request.collection_path,
            added_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"❌ Add face by path error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao adicionar face: {str(e)}"
        )

@app.delete("/api/v2/faces", response_model=RemoveFaceByPathResponse)
async def remove_face_by_path(request: RemoveFaceByPathRequest):
    """
    Remover face usando path da collection.
    
    Args:
        request: Path da collection e posição no índice
        
    Returns:
        Confirmação da remoção
    """
    if not match_service or not match_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Match service não está pronto"
        )
    
    try:
        logger.info(f"🗑️ Removing face by path: {request.collection_path}[{request.index_position}]")
        
        start_time = time.time()
        
        success = await match_service.remove_face_by_path(
            collection_path=request.collection_path,
            index_position=request.index_position
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        
        if success:
            logger.info(f"✅ Face removed by path in {elapsed_time:.1f}ms")
            return RemoveFaceByPathResponse(
                success=True,
                collection_path=request.collection_path,
                index_position=request.index_position,
                removed_at=datetime.utcnow()
            )
        else:
            raise ValueError("Falha ao remover face")
            
    except Exception as e:
        logger.error(f"❌ Remove face by path error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao remover face: {str(e)}"
        )



# ==========================================
# EXCEPTION HANDLER
# ==========================================

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