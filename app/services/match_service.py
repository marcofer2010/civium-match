"""
Serviço principal de match FAISS para o Civium Match Service
"""

import asyncio
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import os
import numpy as np
import faiss
import psutil

from app.config import settings
from app.models.api_models import (
    SmartMatchResponse,
    MatchResult,
    ServiceStats
)
from app.utils.logger import setup_logger

# Logger global para a Collection
logger = setup_logger("collection")


class Collection:
    """Collection FAISS com sistema de invalidação local."""
    
    def __init__(self, company_id: int, company_type: str, collection_type: str):
        self.company_id = company_id
        self.company_type = company_type
        self.collection_type = collection_type
        self.index = None
        self.invalidated_positions: Set[int] = set()  # Posições invalidadas localmente
        self._initialize_index()
    
    @property
    def collection_key(self) -> str:
        """Chave única da collection."""
        return f"{self.company_type}_{self.company_id}_{self.collection_type}"
    
    @property
    def collection_path(self) -> str:
        """Path do arquivo FAISS no sistema."""
        return f"collections/{self.company_type}/{self.company_id}/{self.collection_type}"
    
    def _initialize_index(self):
        """Inicializar índice FAISS."""
        # IndexFlatIP para similaridade coseno (produto interno)
        self.index = faiss.IndexFlatIP(512)  # 512 dimensões
    
    def add_face(self, embedding: np.ndarray) -> int:
        """
        Adicionar embedding ao índice FAISS.
        
        Returns:
            index_position: Posição no índice FAISS (para gravar no PostgreSQL)
        """
        # Validar embedding
        if embedding.shape[0] != 512:
            raise ValueError(f"Embedding deve ter 512 dimensões, recebeu {embedding.shape[0]}")
        
        # Validar que não é vetor zero (má prática!)
        if np.allclose(embedding, 0):
            raise ValueError("Embedding não pode ser vetor zero! Use invalidate_position() para invalidar.")
        
        # Normalizar para similaridade coseno
        embedding = embedding / np.linalg.norm(embedding)
        
        # Posição que será ocupada (antes de adicionar)
        index_position = self.index.ntotal
        
        # Adicionar ao FAISS
        self.index.add(embedding.reshape(1, -1))
        
        return index_position
    
    def invalidate_position(self, index_position: int) -> bool:
        """
        Invalidar uma posição específica.
        
        Em vez de alterar o FAISS (impossível), mantemos uma lista local
        de posições invalidadas e filtramos nos resultados.
        
        Args:
            index_position: Posição no índice FAISS para invalidar
            
        Returns:
            True se invalidou com sucesso
        """
        if index_position < 0 or index_position >= self.index.ntotal:
            return False
            
        self.invalidated_positions.add(index_position)
        return True
    
    def revalidate_position(self, index_position: int) -> bool:
        """
        Revalidar uma posição (remover da lista de invalidadas).
        
        Args:
            index_position: Posição para revalidar
            
        Returns:
            True se revalidou com sucesso
        """
        if index_position in self.invalidated_positions:
            self.invalidated_positions.remove(index_position)
            return True
        return False
    
    def search(self, embedding: np.ndarray, top_k: int = 10, threshold: float = 0.4) -> List[Dict]:
        """
        Buscar embeddings similares, filtrando posições invalidadas.
        
        Returns:
            Lista de matches com index_position, similarity e confidence (sem invalidadas)
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalizar query embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Buscar mais resultados para compensar filtros
        search_k = min(top_k * 3, self.index.ntotal)  # 3x mais para filtrar
        similarities, indices = self.index.search(embedding.reshape(1, -1), search_k)
        
        results = []
        for i, (similarity, index_position) in enumerate(zip(similarities[0], indices[0])):
            # Filtrar: threshold, posições válidas e NÃO invalidadas
            if (similarity >= threshold and 
                index_position != -1 and 
                index_position not in self.invalidated_positions):
                
                confidence = min(similarity * 100, 100.0)
                results.append({
                    "index_position": int(index_position),
                    "similarity": float(similarity),
                    "confidence": float(confidence)
                })
                
                # Parar quando tivermos resultados suficientes
                if len(results) >= top_k:
                    break
        
        return results
    
    def save_to_disk(self) -> None:
        """Salvar índice FAISS e posições invalidadas."""
        import os
        import pickle
        
        # Criar diretório se não existir
        dir_path = f"collections/{self.company_type}/{self.company_id}"
        os.makedirs(dir_path, exist_ok=True)
        
        # Salvar índice FAISS
        index_path = f"{self.collection_path}.index"
        faiss.write_index(self.index, index_path)
        
        # Salvar posições invalidadas (se houver)
        if self.invalidated_positions:
            invalidated_path = f"{self.collection_path}.invalidated"
            with open(invalidated_path, 'wb') as f:
                pickle.dump(self.invalidated_positions, f)
    
    @classmethod
    def load_from_disk(cls, company_id: int, company_type: str, collection_type: str) -> 'Collection':
        """Carregar collection do disco - índice FAISS + posições invalidadas."""
        import pickle
        
        collection = cls(company_id, company_type, collection_type)
        
        index_path = f"{collection.collection_path}.index"
        invalidated_path = f"{collection.collection_path}.invalidated"
        
        try:
            # Carregar índice FAISS
            if os.path.exists(index_path):
                collection.index = faiss.read_index(index_path)
                
                # Carregar posições invalidadas
                if os.path.exists(invalidated_path):
                    with open(invalidated_path, 'rb') as f:
                        collection.invalidated_positions = pickle.load(f)
                
                valid_faces = collection.index.ntotal - len(collection.invalidated_positions)
                logger.info(f"📂 Collection carregada: {collection.collection_key} "
                           f"({collection.index.ntotal} total, {valid_faces} válidas, "
                           f"{len(collection.invalidated_positions)} invalidadas)")
            else:
                logger.info(f"📂 Nova collection criada: {collection.collection_key}")
                
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar collection {collection.collection_key}: {e}")
            logger.info("🔄 Criando nova collection...")
            collection._initialize_index()
            collection.invalidated_positions = set()
        
        return collection
    
    @property
    def face_count(self) -> int:
        """Número de faces VÁLIDAS na collection (total - invalidadas)."""
        total = self.index.ntotal if self.index else 0
        return total - len(self.invalidated_positions)
    
    @property  
    def total_face_count(self) -> int:
        """Número total de faces na collection (incluindo invalidadas)."""
        return self.index.ntotal if self.index else 0


class CollectionManager:
    """Gerenciador de collections com criação automática."""
    
    def __init__(self):
        self.logger = setup_logger("collection-manager")
        self.collections_cache: Dict[str, Collection] = {}
    
    async def get_or_create_collection(self, company_id: int, company_type: str, 
                                     collection_type: str) -> Collection:
        """Retorna collection existente ou cria nova se não existir."""
        collection_key = f"{company_type}_{company_id}_{collection_type}"
        
        if collection_key not in self.collections_cache:
            # Tentar carregar do disco ou criar nova
            self.logger.info(f"📁 Carregando/criando collection: {collection_key}")
            collection = Collection.load_from_disk(company_id, company_type, collection_type)
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
                        str(company_id): [MatchResult(
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
                
                # Armazenar por categoria e company (converter company_id para string)
                company_id_str = str(company_id)
                if company_id_str not in results_by_category[category]:
                    results_by_category[category][company_id_str] = []
                results_by_category[category][company_id_str].extend(simplified_matches)
                
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
        
        # Converter embedding para numpy
        embedding_array = np.array(embedding, dtype=np.float32)
        
        # Adicionar à collection e obter posição
        index_position = collection.add_face(embedding_array)
        
        # Salvar alterações
        collection.save_to_disk()
        
        self.logger.info(f"👤 Face adicionada na posição {index_position} à collection {collection.collection_key}")
        
        return index_position
    
    async def remove_face_from_collection(self, company_id: int, company_type: str, 
                                        collection_type: str, index_position: int) -> bool:
        """
        Remove (invalida) uma face de uma collection.
        
        Usa invalidação local: mantém o embedding no FAISS mas filtra dos resultados.
        Para remoção definitiva, você deve fazer no PostgreSQL.
        """
        try:
            collection = await self.collection_manager.get_or_create_collection(
                company_id, company_type, collection_type
            )
            
            # Invalidar localmente
            success = collection.invalidate_position(index_position)
            
            if success:
                # Salvar alterações (incluindo lista de invalidadas)
                collection.save_to_disk()
                
                self.logger.info(f"✅ Face na posição {index_position} invalidada em {company_type}/{company_id}/{collection_type}")
                self.logger.info(f"💡 Total: {collection.total_face_count}, Válidas: {collection.face_count}, Invalidadas: {len(collection.invalidated_positions)}")
                
                return True
            else:
                self.logger.warning(f"❌ Posição {index_position} inválida ou fora do range em {company_type}/{company_id}/{collection_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao invalidar face na posição {index_position}: {e}")
            return False
    
    async def get_stats(self) -> ServiceStats:
        """Retorna estatísticas do serviço."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        avg_match_time = 0
        if self.stats['total_smart_matches'] > 0:
            avg_match_time = self.stats['total_match_time_ms'] / self.stats['total_smart_matches']
        
        # Informações de memória
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Contar faces válidas vs totais
        total_faces = 0
        valid_faces = 0
        invalidated_faces = 0
        
        for collection in self.collection_manager.collections_cache.values():
            total_faces += collection.total_face_count
            valid_faces += collection.face_count
            invalidated_faces += len(collection.invalidated_positions)
        
        return ServiceStats(
            uptime_seconds=uptime,
            total_smart_matches=self.stats['total_smart_matches'],
            total_collections=len(self.collection_manager.collections_cache),
            total_faces=valid_faces,  # Mostrar apenas faces válidas no total
            average_match_time_ms=avg_match_time,
            auto_registrations=self.stats['auto_registrations'],
            memory_usage_mb=memory_mb,
            # Adicionar informações extras para debug
            total_faces_including_invalidated=total_faces,
            invalidated_faces=invalidated_faces
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
    
    # NOTA: Métodos promote/demote removidos
    # Com FAISS puro, promoção/rebaixamento deve ser feita:
    # 1. Obter embedding da posição original via PostgreSQL
    # 2. Adicionar na nova collection  
    # 3. Marcar original como removida no PostgreSQL

    # Novos métodos para trabalhar com paths de collections
    
    def _parse_collection_path(self, collection_path: str) -> tuple[str, int, str]:
        """
        Fazer parse do path da collection.
        
        Args:
            collection_path: Path no formato "company_type/company_id/collection_type"
            
        Returns:
            tuple: (company_type, company_id, collection_type)
        """
        parts = collection_path.split('/')
        if len(parts) != 3:
            raise ValueError('Path deve ter formato: company_type/company_id/collection_type')
        
        company_type, company_id_str, collection_type = parts
        
        if company_type not in ['public', 'private']:
            raise ValueError('company_type deve ser "public" ou "private"')
        
        try:
            company_id = int(company_id_str)
        except ValueError:
            raise ValueError('company_id deve ser um número')
            
        if collection_type not in ['known', 'unknown']:
            raise ValueError('collection_type deve ser "known" ou "unknown"')
            
        return company_type, company_id, collection_type

    async def add_face_by_path(self, collection_path: str, embedding: List[float]) -> int:
        """
        Adicionar face usando path da collection.
        
        Args:
            collection_path: Path no formato "company_type/company_id/collection_type"
            embedding: Embedding da face
            
        Returns:
            Posição no índice FAISS
        """
        company_type, company_id, collection_type = self._parse_collection_path(collection_path)
        
        return await self.add_face_to_collection(
            company_id=company_id,
            company_type=company_type,
            collection_type=collection_type,
            embedding=embedding
        )

    async def remove_face_by_path(self, collection_path: str, index_position: int) -> bool:
        """
        Remover face usando path da collection.
        
        Args:
            collection_path: Path no formato "company_type/company_id/collection_type"
            index_position: Posição no índice FAISS
            
        Returns:
            True se removido com sucesso
        """
        company_type, company_id, collection_type = self._parse_collection_path(collection_path)
        
        return await self.remove_face_from_collection(
            company_id=company_id,
            company_type=company_type,
            collection_type=collection_type,
            index_position=index_position
        )

    async def transfer_face(self, origin_path: str, target_path: str, index_position: int) -> Dict:
        """
        Transferir face entre collections.
        
        NOTA: Com FAISS puro, você deve:
        1. Obter embedding da posição original via PostgreSQL (não do FAISS)
        2. Adicionar na nova collection
        3. Marcar original como removida no PostgreSQL
        
        Args:
            origin_path: Path da collection de origem
            target_path: Path da collection de destino
            index_position: Posição no índice FAISS da collection de origem
            
        Returns:
            Dicionário com informações da transferência
        """
        self.logger.error(f"❌ Transfer não suportado com FAISS puro!")
        self.logger.error(f"💡 Implemente via PostgreSQL:")
        self.logger.error(f"   1. SELECT embedding FROM faces WHERE index_position = {index_position}")
        self.logger.error(f"   2. POST /api/v2/faces com embedding + {target_path}")
        self.logger.error(f"   3. UPDATE faces SET removed = true WHERE index_position = {index_position}")
        
        raise ValueError(
            "Transfer não suportado com FAISS puro. "
            "Use PostgreSQL para obter embedding e adicionar na nova collection."
        ) 