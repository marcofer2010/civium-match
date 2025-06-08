# 🎯 **Civium Match Service**

Serviço de match facial inteligente usando FAISS com arquitetura multi-tenant e busca federada.

## 🚀 **Funcionalidades Principais**

### **🧠 Smart Match - Busca em Cascata com Paths**
Sistema inteligente de busca facial usando paths simplificados para identificar collections:

**Path Format:** `company_type/company_id/collection_type`
- **Exemplos:** `private/123/known`, `public/456/unknown`

**Etapas da Busca:**
1. **Collections 'Known'**: Busca pessoas conhecidas (federada se órgão público + câmera compartilhada)
2. **Collection 'Unknown'**: Busca na própria empresa (se `search_unknown=True`)
3. **Auto-Registro**: Cadastra automaticamente se não encontrar (se `auto_register=True`)

### **🏢 Arquitetura Multi-Tenant Simplificada**
```
collections/
├── private/
│   ├── 123/
│   │   ├── known.index       # Apenas índice FAISS
│   │   └── unknown.index     # Apenas índice FAISS
└── public/
    ├── 100/                  # Polícia Civil
    │   ├── known.index       # Apenas índice FAISS
    │   └── unknown.index     # Apenas índice FAISS
    └── 200/                  # Polícia Federal
        ├── known.index
        └── unknown.index
```

### **🗄️ Integração com PostgreSQL**
```
┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     FAISS       │
│                 │    │                 │
│ face_id         │    │ index_position  │
│ person_id       │    │ embedding_512d  │
│ index_position  │◄──►│ similarity      │
│ company_id      │    │                 │
│ collection_type │    │                 │
│ embedding       │    │                 │
│ is_removed      │    │                 │
│ created_at      │    │                 │
│ metadata        │    │                 │
└─────────────────┘    └─────────────────┘
```

**Responsabilidades:**
- **FAISS**: Busca vetorial ultrarrápida, retorna `index_position`
- **PostgreSQL**: Metadados, mapeamento `index_position` → dados reais

### **🌐 Lógica de Busca Federada**

| Company Type | Camera Shared | Busca em Collections 'Known' |
|--------------|---------------|------------------------------|
| **private** | `false` | ✅ Apenas própria collection 'known' |
| **private** | `true` | ✅ Apenas própria collection 'known' |
| **public** | `false` | ✅ Apenas própria collection 'known' |
| **public** | `true` | 🌐 **TODAS** as collections 'known' públicas + própria |

**Exemplo:** Órgão público com câmera compartilhada pode acessar base de dados de todos os órgãos públicos!

## 📚 **API Endpoints**

### **🧠 Smart Match**
```http
POST /api/smart-match
```

**Request:**
```json
{
    "collection_path": "private/123/known",  // Path da collection base
    "embedding": [0.1, 0.2, ...],          // 512 dimensões
    "camera_shared": false,                  // Câmera compartilhada?
    "search_unknown": true,                  // Buscar em 'unknown'?
    "auto_register": true,                   // Auto-cadastrar se não encontrar?
    "threshold": 0.4,                        // Threshold de similaridade
    "top_k": 5                              // Máximo de resultados
}
```

**Response:**
```json
{
    "query_embedding_hash": "a1b2c3d4",
    "search_performed": {
        "collection_path": "private/123/known",
        "camera_shared": false,
        "search_unknown": true,
        "auto_register": true
    },
    "result_type": "found_known",
    "matches": {
        "private": {
            "123": [
                {
                    "index_position": 0,
                    "similarity": 0.95,
                    "confidence": 95.0
                }
            ]
        }
    },
    "auto_registered_index": null,
    "total_collections_searched": 2,
    "search_time_ms": 45.2,
    "threshold_used": 0.4,
    "top_k_used": 5
}
```

### **👤 Adicionar Face**
```http
POST /api/v2/faces
```

**Request:**
```json
{
    "collection_path": "private/123/known",
    "embedding": [0.1, 0.2, ...]
}
```

**Response:**
```json
{
    "index_position": 5,
    "collection_path": "private/123/known",
    "added_at": "2024-01-15T10:30:00Z"
}
```

### **🗑️ Remover Face (Soft Delete)**
```http
DELETE /api/v2/faces
```

**Request:**
```json
{
    "collection_path": "private/123/known",
    "index_position": 5
}
```

**Response:**
```json
{
    "success": true,
    "collection_path": "private/123/known",
    "index_position": 5,
    "removed_at": "2024-01-15T10:35:00Z"
}
```

**⚠️ Importante:** FAISS não suporta remoção real. Marque como removida no PostgreSQL!

### **📊 Estatísticas**
```http
GET /api/stats
```

**Response:**
```json
{
    "uptime_seconds": 3600.5,
    "total_smart_matches": 1250,
    "total_collections": 12,
    "total_faces": 5430,
    "average_match_time_ms": 23.8,
    "auto_registrations": 89,
    "memory_usage_mb": 512.3
}
```

### **🏥 Health Check**
```http
GET /health
```

## 🎯 **Cenários de Uso**

### **1. Empresa Privada - Busca Isolada**
```python
# Buscar apenas na própria base
{
    "collection_path": "private/123/known",
    "embedding": [...],
    "camera_shared": false,     # Não importa para empresas privadas
    "search_unknown": True,     # Buscar também em unknown
    "auto_register": True       # Auto-registrar se não encontrar
}
```

### **2. Órgão Público - Busca Federada**
```python
# Buscar em todas as bases públicas
{
    "collection_path": "public/100/known", 
    "embedding": [...],
    "camera_shared": True,      # ATIVA busca federada
    "search_unknown": False,    # Não buscar em unknown
    "auto_register": False      # Não auto-registrar
}
```

### **3. Fluxo Completo com PostgreSQL**
```python
# 1. Adicionar face
POST /api/v2/faces
{
    "collection_path": "private/123/unknown",
    "embedding": [...]
}
# Response: {"index_position": 42}

# 2. Salvar no PostgreSQL
INSERT INTO faces (face_id, person_id, index_position, collection_path, embedding, company_id)
VALUES ('uuid', 'person_123', 42, 'private/123/unknown', [...], 123);

# 3. Buscar face
POST /api/smart-match
{
    "collection_path": "private/123/known",
    "embedding": [...],
    "search_unknown": True
}
# Response: {"matches": {"private": {"123": [{"index_position": 42, ...}]}}}

# 4. Mapear resultado
SELECT face_id, person_id, metadata 
FROM faces 
WHERE index_position = 42 AND collection_path = 'private/123/unknown';
```

## 🚀 **Executar**

### **Docker**
```bash
# Build e run
docker build -t civium-match .
docker run -p 8000:8000 civium-match

# Com variáveis de ambiente
docker run -p 8000:8000 -e DEBUG=true civium-match
```

### **Local**
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar
python main.py

# Executar com reload (desenvolvimento)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Teste**
```bash
# Testar endpoints com paths
python test_path_api.py

# Testes existentes
python test_basic.py
python test_smart_match.py
```

## 🔧 **Configuração**

### **Variáveis de Ambiente**
```bash
# .env
DEBUG=false
LOG_LEVEL=INFO
COLLECTIONS_DIR=./collections
MAX_COLLECTIONS_CACHE=100
ALLOWED_ORIGINS=["*"]
```

### **Estrutura do Projeto**
```
civium-match/
├── main.py                 # FastAPI app
├── app/
│   ├── models/
│   │   └── api_models.py   # Modelos Pydantic
│   ├── services/
│   │   └── match_service.py # Lógica de match e collections
│   ├── utils/
│   │   └── logger.py       # Configuração de logging
│   └── config.py           # Configurações
├── collections/            # Armazenamento FAISS (auto-criado)
│   ├── private/
│   │   └── 123/
│   │       ├── known.index
│   │       └── unknown.index
│   └── public/
│       └── 100/
│           ├── known.index
│           └── unknown.index
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## 📋 **Response Types**

| `result_type` | Descrição | `matches` | `auto_registered_index` |
|---------------|-----------|-----------|------------------------|
| `found_known` | Encontrado em collection 'known' | ✅ Presente | `null` |
| `found_unknown` | Encontrado em collection 'unknown' própria | ✅ Presente | `null` |
| `auto_registered` | Auto-registrado em 'unknown' | `null` | ✅ Presente |
| `not_found` | Não encontrado | `null` | `null` |

## 🎯 **Benefícios da Arquitetura Simplificada**

✅ **Performance**: FAISS puro sem overhead de metadados  
✅ **Simplicidade**: Apenas arquivos `.index`, sem `.pkl`  
✅ **Flexibilidade**: PostgreSQL gerencia todos os metadados  
✅ **Escalabilidade**: Separação clara de responsabilidades  
✅ **Manutenibilidade**: Código mais limpo e focado  
✅ **Consistência**: Paths uniformes em toda a API  

## ⚠️ **Limitações Importantes**

❌ **Sem Remoção Real**: FAISS não suporta remoção de vetores  
❌ **Sem Transferência**: Use PostgreSQL para mover faces entre collections  
❌ **Sem Metadados**: Todos os metadados devem estar no PostgreSQL  

**Solução:** Marque faces como removidas no PostgreSQL e ignore nas buscas! 