"""
Serviço principal de match FAISS para o Civium Match Service
"""

import asyncio
import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import numpy as np
import faiss
import pickle
import psutil

from app.config import settings
from app.models.api_models import (
    SmartMatchResponse,
    MatchResult,
    ServiceStats
)
from app.utils.logger import setup_logger


class Collection:
    """Representa uma collection de faces."""
    
    def __init__(self, company_id: int, company_type: str, collection_type: str, 
                 metadata: Optional[Dict] = None):
        self.company_id = company_id
        self.company_type = company_type  # 'public' ou 'private'
        self.collection_type = collection_type  # 'known' ou 'unknown'
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.updated_at = None
        self.face_count = 0
        self.was_just_created = False
        
        # Logger
        self.logger = setup_logger(f"collection-{self.collection_key}")
        
        # FAISS index para esta collection
        self.index: Optional[faiss.Index] = None
        self.face_id_mapping: Dict[int, str] = {}  # Mapear index_position para face_id
        self.face_ids: List[str] = []  # Mapear posição no index para face_id
        self.face_metadata: Dict[str, Dict] = {}  # Metadata por face_id
        
        # Soft delete - posições marcadas como removidas
        self.removed_positions: set = set()  # Set de index_positions removidas
        
        self._initialize_index()
    
    @property
    def collection_key(self) -> str:
        """Chave única da collection."""
        tenant_category = "public" if self.company_type == "public" else "private"
        return f"{tenant_category}/{self.company_id}/{self.collection_type}"
    
    @property
    def collection_path(self) -> str:
        """Caminho da collection no filesystem."""
        tenant_category = "public" if self.company_type == "public" else "private"
        return f"collections/{tenant_category}/{str(self.company_id)}/{self.collection_type}"
    
    def _initialize_index(self):
        """Inicializa o índice FAISS para esta collection."""
        # Usar índice flat para começar (pode ser otimizado depois)
        self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
        # Para busca por similaridade cosseno, normalizar embeddings
        
    def add_face(self, embedding: np.ndarray, face_id: str) -> None:
        """
        Adiciona uma face à collection.
        
        Args:
            embedding: Embedding da face
            face_id: ID único da face
        """
        if self.index is None:
            self._initialize_index()
        
        # Adicionar ao índice FAISS
        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(embedding_2d)
        
        # Mapear posição no índice para face_id
        index_position = self.index.ntotal - 1
        self.face_id_mapping[index_position] = face_id
        self.face_ids.append(face_id)
        self.face_metadata[face_id] = {
            'added_at': datetime.utcnow()
        }
        
        self.face_count += 1
        self.updated_at = datetime.utcnow()
        
        self.logger.debug(f"Face {face_id} adicionada à collection {self.collection_key} na posição {index_position}")
        
        # Salvar automaticamente
        self.save_to_disk()
    
    def remove_face(self, index_position: int) -> bool:
        """
        Remove uma face da collection (soft delete).
        
        Args:
            index_position: Posição no índice FAISS
            
        Returns:
            True se removido com sucesso, False se não encontrado
        """
        if index_position < 0 or index_position >= self.face_count:
            self.logger.warning(f"Posição inválida para remoção: {index_position}")
            return False
            
        if index_position in self.removed_positions:
            self.logger.warning(f"Face na posição {index_position} já está removida")
            return False
        
        # Marcar como removida (soft delete)
        self.removed_positions.add(index_position)
        
        # Remover metadata se existir
        if index_position in self.face_id_mapping:
            face_id = self.face_id_mapping[index_position]
            if face_id in self.face_metadata:
                del self.face_metadata[face_id]
        
        self.updated_at = datetime.utcnow()
        
        self.logger.info(f"Face na posição {index_position} marcada como removida (soft delete)")
        
        # Salvar alterações
        self.save_to_disk()
        
        return True
    
    def search(self, embedding: np.ndarray, top_k: int = 10, threshold: float = 0.4) -> List[Dict]:
        """Busca faces similares na collection."""
        if self.face_count == 0:
            return []
        
        # Normalizar query embedding
        query_embedding = embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Buscar mais resultados para compensar faces removidas
        search_k = min(top_k * 3, self.face_count)  # Buscar 3x mais para filtrar removidas
        similarities, indices = self.index.search(query_embedding.reshape(1, -1), search_k)
        
        results = []
        for i, (similarity, index) in enumerate(zip(similarities[0], indices[0])):
            if index == -1:  # Fim dos resultados válidos
                break
                
            if similarity < threshold:  # Abaixo do threshold
                continue
                
            # FILTRAR POSIÇÕES REMOVIDAS (soft delete)
            if index in self.removed_positions:
                continue
            
            results.append({
                'index_position': int(index),  # Retornar posição no índice FAISS
                'similarity': float(similarity),
                'confidence': float(similarity * 100)  # Converter para percentual
            })
            
            # Parar quando atingir top_k resultados válidos
            if len(results) >= top_k:
                break
        
        return results
    
    def save_to_disk(self) -> None:
        """Salva collection em disco."""
        directory = os.path.dirname(self.collection_path)
        os.makedirs(directory, exist_ok=True)
        
        # Salvar índice FAISS
        index_path = f"{self.collection_path}.index"
        faiss.write_index(self.index, index_path)
        
        # Salvar metadata
        metadata_path = f"{self.collection_path}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'company_id': self.company_id,
                'company_type': self.company_type,
                'collection_type': self.collection_type,
                'metadata': self.metadata,
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'face_count': self.face_count,
                'face_ids': self.face_ids,
                'face_id_mapping': self.face_id_mapping,
                'face_metadata': self.face_metadata,
                'removed_positions': self.removed_positions
            }, f)
    
    @classmethod
    def load_from_disk(cls, company_id: int, company_type: str, collection_type: str) -> 'Collection':
        """Carrega collection do disco."""
        tenant_category = "public" if company_type == "public" else "private"
        collection_path = f"collections/{tenant_category}/{str(company_id)}/{collection_type}"
        
        # Carregar metadata
        metadata_path = f"{collection_path}.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        # Criar collection
        collection = cls(
            company_id=data['company_id'],
            company_type=data['company_type'],
            collection_type=data['collection_type'],
            metadata=data['metadata']
        )
        
        # Restaurar dados
        collection.created_at = data['created_at']
        collection.updated_at = data['updated_at']
        collection.face_count = data['face_count']
        collection.face_ids = data['face_ids']
        collection.face_id_mapping = data['face_id_mapping']
        collection.face_metadata = data['face_metadata']
        collection.removed_positions = data['removed_positions']
        
        # Carregar índice FAISS
        index_path = f"{collection_path}.index"
        if os.path.exists(index_path):
            collection.index = faiss.read_index(index_path)
        
        return collection


class CollectionManager:
    """Gerenciador de collections com criação automática."""
    
    def __init__(self):
        self.logger = setup_logger("collection-manager")
        self.collections_cache: Dict[str, Collection] = {}
    
    async def get_or_create_collection(self, company_id: int, company_type: str, 
                                     collection_type: str) -> Collection:
        """Retorna collection existente ou cria nova se não existir."""
        tenant_category = "public" if company_type == "public" else "private"
        collection_key = f"{tenant_category}/{company_id}/{collection_type}"
        
        if collection_key not in self.collections_cache:
            collection_path = f"collections/{tenant_category}/{str(company_id)}/{collection_type}"
            
            if os.path.exists(f"{collection_path}.pkl"):
                # Carregar existente
                self.logger.info(f"📁 Carregando collection existente: {collection_key}")
                collection = Collection.load_from_disk(company_id, company_type, collection_type)
            else:
                # Criar nova
                self.logger.info(f"📁 Criando nova collection: {collection_key}")
                collection = Collection(
                    company_id=company_id,
                    company_type=company_type,
                    collection_type=collection_type
                )
                collection.was_just_created = True
                # Salvar nova collection
                collection.save_to_disk()
                
            self.collections_cache[collection_key] = collection
        
        return self.collections_cache[collection_key]
    
    async def get_all_public_known_collections(self) -> List[Collection]:
        """Retorna todas as collections 'known' de órgãos públicos."""
        public_dir = "collections/public"
        collections = []
        
        if not os.path.exists(public_dir):
            return collections
        
        for company_dir in os.listdir(public_dir):
            company_path = os.path.join(public_dir, company_dir)
            if os.path.isdir(company_path):
                try:
                    collection = await self.get_or_create_collection(
                        company_id=int(company_dir),
                        company_type="public",
                        collection_type="known"
                    )
                    collections.append(collection)
                except Exception as e:
                    self.logger.error(f"❌ Erro ao carregar collection pública {company_dir}/known: {e}")
        
        return collections


class MatchService:
    """Serviço principal de match usando FAISS."""
    
    def __init__(self):
        self.logger = setup_logger("match-service")
        self.collection_manager = CollectionManager()
        self.is_ready = False
        
        # Estatísticas
        self.start_time = datetime.utcnow()
        self.stats = {
            'total_smart_matches': 0,
            'total_match_time_ms': 0,
            'auto_registrations': 0
        }
        
    async def initialize(self) -> None:
        """Inicializa o serviço."""
        self.logger.info("🚀 Inicializando Match Service...")
        
        try:
            # Criar diretório de collections se não existir
            os.makedirs("collections/public", exist_ok=True)
            os.makedirs("collections/private", exist_ok=True)
            
            self.is_ready = True
            self.logger.info("✅ Match Service inicializado com sucesso")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar Match Service: {e}")
            raise
    
    async def smart_match(self, embedding: List[float], company_id: int, company_type: str,
                         camera_shared: bool = False, search_unknown: bool = False,
                         auto_register: bool = False, threshold: float = None, 
                         top_k: int = None) -> SmartMatchResponse:
        """
        Busca inteligente com lógica em cascata:
        1. Buscar em collections 'known' (federada se câmera compartilhada)
        2. Se não encontrar e search_unknown=True, buscar na collection 'unknown' da própria empresa
        3. Se não encontrar e auto_register=True, cadastrar na collection 'unknown' da própria empresa
        """
        start_time = time.time()
        
        # Usar defaults se não fornecidos
        threshold = threshold or settings.DEFAULT_MATCH_THRESHOLD
        top_k = top_k or settings.DEFAULT_TOP_K
        
        # Converter embedding para numpy
        embedding_array = np.array(embedding, dtype=np.float32)
        
        # Gerar hash do embedding para cache/debugging
        embedding_hash = hashlib.md5(embedding_array.tobytes()).hexdigest()[:16]
        
        search_details = {
            "company_id": company_id,
            "company_type": company_type,
            "camera_shared": camera_shared,
            "search_unknown": search_unknown,
            "auto_register": auto_register,
            "threshold": threshold,
            "top_k": top_k
        }
        
        collections_searched = 0
        
        # ETAPA 1: Buscar em collections 'known'
        self.logger.info(f"🔍 Etapa 1: Buscando em collections 'known'...")
        
        if camera_shared:
            # BUSCA FEDERADA: Câmera compartilhada permite acesso a todas as collections 'known' públicas
            # Isso vale tanto para órgãos públicos quanto empresas privadas em espaços compartilhados
            known_collections = await self.collection_manager.get_all_public_known_collections()
            
            # Também incluir a própria collection 'known' da empresa
            own_collection = await self.collection_manager.get_or_create_collection(
                company_id, company_type, "known"
            )
            known_collections.append(own_collection)
            
            self.logger.info(f"🌐 BUSCA FEDERADA: {len(known_collections)} collections (públicas + própria)")
            
        else:
            # BUSCA ISOLADA: Câmera privada acessa apenas própria collection 'known'
            known_collection = await self.collection_manager.get_or_create_collection(
                company_id, company_type, "known"
            )
            known_collections = [known_collection]
            self.logger.info(f"🏢 BUSCA ISOLADA: apenas collection própria")
        
        collections_searched += len(known_collections)
        
        # Buscar em paralelo nas collections 'known'
        known_results, results_by_category = await self._search_multiple_collections(embedding_array, known_collections, threshold, top_k)
        
        if known_results:
            # Encontrou na collection 'known'
            best_match = known_results[0]
            search_time_ms = (time.time() - start_time) * 1000
            
            self.stats['total_smart_matches'] += 1
            self.stats['total_match_time_ms'] += search_time_ms
            
            self.logger.info(f"✅ Encontrado em collection 'known': {best_match['company_id']}/{best_match['collection_type']}")
            
            # Converter results_by_category para MatchResult objects
            matches_formatted = {}
            for category, companies in results_by_category.items():
                if companies:  # Só incluir se tiver companies com matches
                    matches_formatted[category] = {}
                    for company_id, matches in companies.items():
                        if matches:  # Só incluir se tiver matches
                            matches_formatted[category][company_id] = [
                                MatchResult(**match) for match in matches
                            ]
            
            # Log detalhado para câmeras compartilhadas
            if camera_shared and len(matches_formatted) > 0:
                self.logger.info(f"🌐 BUSCA FEDERADA - Resultados por categoria:")
                for category, companies in matches_formatted.items():
                    company_count = len(companies)
                    total_matches = sum(len(matches) for matches in companies.values())
                    self.logger.info(f"   📁 {category}: {company_count} companies, {total_matches} matches")
            
            return SmartMatchResponse(
                query_embedding_hash=embedding_hash,
                search_performed=search_details,
                result_type="found_known",
                matches=matches_formatted if matches_formatted else None,
                auto_registered_index=None,
                total_collections_searched=collections_searched,
                search_time_ms=search_time_ms,
                threshold_used=threshold,
                top_k_used=top_k
            )
        
        # ETAPA 2: Buscar na collection 'unknown' da própria empresa (se habilitado)
        if search_unknown:
            self.logger.info(f"🔍 Etapa 2: Buscando na collection 'unknown' da própria empresa...")
            
            unknown_collection = await self.collection_manager.get_or_create_collection(
                company_id, company_type, "unknown"
            )
            collections_searched += 1
            
            unknown_results = unknown_collection.search(embedding_array, top_k, threshold)
            
            if unknown_results:
                # Encontrou na collection 'unknown'
                best_match = unknown_results[0]
                search_time_ms = (time.time() - start_time) * 1000
                
                self.stats['total_smart_matches'] += 1
                self.stats['total_match_time_ms'] += search_time_ms
                
                self.logger.info(f"✅ Encontrado em collection 'unknown': {company_id}")
                
                # Organizar o match na nova estrutura por categoria
                category = "public" if company_type == "public" else "private"
                matches_formatted = {
                    category: {
                        company_id: [MatchResult(
                            index_position=best_match["index_position"],
                            similarity=best_match["similarity"],
                            confidence=best_match["confidence"]
                        )]
                    }
                }
                
                return SmartMatchResponse(
                    query_embedding_hash=embedding_hash,
                    search_performed=search_details,
                    result_type="found_unknown",
                    matches=matches_formatted if matches_formatted else None,
                    auto_registered_index=None,
                    total_collections_searched=collections_searched,
                    search_time_ms=search_time_ms,
                    threshold_used=threshold,
                    top_k_used=top_k
                )
        
        # ETAPA 3: Auto-registro na collection 'unknown' (se habilitado)
        if auto_register:
            self.logger.info(f"🤖 Etapa 3: Auto-registrando na collection 'unknown'...")
            
            index_position = await self.add_face_to_collection(
                company_id=company_id,
                company_type=company_type,
                collection_type="unknown",
                embedding=embedding
            )
            
            search_time_ms = (time.time() - start_time) * 1000
            
            self.stats['total_smart_matches'] += 1
            self.stats['auto_registrations'] += 1
            self.stats['total_match_time_ms'] += search_time_ms
            
            self.logger.info(f"✅ Face auto-registrada na posição: {index_position}")
            
            return SmartMatchResponse(
                query_embedding_hash=embedding_hash,
                search_performed=search_details,
                result_type="auto_registered",
                matches=None,
                auto_registered_index=index_position,
                total_collections_searched=collections_searched,
                search_time_ms=search_time_ms,
                threshold_used=threshold,
                top_k_used=top_k
            )
        
        # Não encontrou em lugar nenhum
        search_time_ms = (time.time() - start_time) * 1000
        
        self.stats['total_smart_matches'] += 1
        self.stats['total_match_time_ms'] += search_time_ms
        
        self.logger.info(f"❌ Não encontrado em nenhuma collection")
        
        return SmartMatchResponse(
            query_embedding_hash=embedding_hash,
            search_performed=search_details,
            result_type="not_found",
            matches=None,
            auto_registered_index=None,
            total_collections_searched=collections_searched,
            search_time_ms=search_time_ms,
            threshold_used=threshold,
            top_k_used=top_k
        )
    
    async def _search_multiple_collections(self, embedding: np.ndarray, collections: List[Collection],
                                         threshold: float, top_k: int) -> tuple[List[Dict], Dict[str, Dict[str, List[Dict]]]]:
        """
        Busca em múltiplas collections em paralelo.
        
        Returns:
            tuple: (consolidated_results, results_by_category)
                - consolidated_results: Top results consolidados ordenados por similaridade
                - results_by_category: Resultados organizados por categoria (public/private) e company
        """
        if not collections:
            return [], {}
        
        # Buscar em paralelo (limitando concorrência)
        semaphore = asyncio.Semaphore(5)
        
        async def search_one(collection):
            async with semaphore:
                results = collection.search(embedding, top_k, threshold)
                return collection, results
        
        tasks = [search_one(collection) for collection in collections]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organizar resultados por categoria (public/private) e company
        results_by_category = {"public": {}, "private": {}}
        all_results = []
        
        for result in results_list:
            if isinstance(result, tuple) and len(result) == 2:
                collection, matches = result
                
                # Determinar categoria baseada no tipo da empresa
                category = "public" if collection.company_type == "public" else "private"
                company_id = collection.company_id
                
                # Simplificar matches (usar apenas dados do FAISS)
                simplified_matches = []
                for match in matches:
                    simplified_matches.append({
                        "index_position": match["index_position"],
                        "similarity": match["similarity"], 
                        "confidence": match["confidence"]
                    })
                
                # Armazenar por categoria e company
                if company_id not in results_by_category[category]:
                    results_by_category[category][company_id] = []
                results_by_category[category][company_id].extend(simplified_matches)
                
                # Adicionar ao consolidado (manter estrutura original para compatibilidade interna)
                all_results.extend(matches)
        
        # Ordenar consolidado por similaridade (maior primeiro)
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return all_results[:top_k], results_by_category
    
    async def add_face_to_collection(self, company_id: int, company_type: str, collection_type: str,
                                    embedding: List[float]) -> int:
        """Adiciona uma face a uma collection."""
        collection = await self.collection_manager.get_or_create_collection(
            company_id, company_type, collection_type
        )
        
        face_id = str(uuid.uuid4())
        
        # Converter embedding para numpy
        embedding_array = np.array(embedding, dtype=np.float32)
        
        # Posição que será ocupada no índice (antes de adicionar)
        index_position = collection.face_count
        
        # Adicionar à collection
        collection.add_face(embedding_array, face_id)
        
        # Salvar alterações
        collection.save_to_disk()
        
        self.logger.info(f"👤 Face adicionada na posição {index_position} à collection {collection.collection_key}")
        
        return index_position
    
    async def remove_face_from_collection(self, company_id: int, company_type: str, 
                                        collection_type: str, index_position: int) -> bool:
        """Remove uma face de uma collection."""
        collection = await self.collection_manager.get_or_create_collection(
            company_id, company_type, collection_type
        )
        
        success = collection.remove_face(index_position)
        
        if success:
            self.logger.info(f"🗑️ Face removida da posição {index_position} da collection {collection.collection_key}")
        else:
            self.logger.warning(f"❌ Falha ao remover face da posição {index_position} da collection {collection.collection_key}")
        
        return success
    
    async def get_stats(self) -> ServiceStats:
        """Retorna estatísticas do serviço."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        avg_match_time = 0
        if self.stats['total_smart_matches'] > 0:
            avg_match_time = self.stats['total_match_time_ms'] / self.stats['total_smart_matches']
        
        # Informações de memória
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return ServiceStats(
            uptime_seconds=uptime,
            total_smart_matches=self.stats['total_smart_matches'],
            total_collections=len(self.collection_manager.collections_cache),
            total_faces=sum(col.face_count for col in self.collection_manager.collections_cache.values()),
            average_match_time_ms=avg_match_time,
            auto_registrations=self.stats['auto_registrations'],
            memory_usage_mb=memory_mb
        )
    
    async def cleanup(self) -> None:
        """Limpa recursos do serviço."""
        self.logger.info("🧹 Limpando recursos do Match Service...")
        
        # Salvar todas as collections
        for collection in self.collection_manager.collections_cache.values():
            try:
                collection.save_to_disk()
            except Exception as e:
                self.logger.error(f"❌ Erro ao salvar collection {collection.collection_key}: {e}")
        
        self.logger.info("✅ Cleanup concluído") 