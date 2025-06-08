# 🎯 **Civium Match Service**

Serviço de match facial inteligente usando FAISS com arquitetura multi-tenant e busca federada.

## 🚀 **Funcionalidades Principais**

### **🧠 Smart Match - Busca em Cascata**
Sistema inteligente de busca facial com controle granular por empresa e tipo de câmera:

**Etapas da Busca:**
1. **Collections 'Known'**: Busca pessoas conhecidas (federada se órgão público + câmera compartilhada)
2. **Collection 'Unknown'**: Busca na própria empresa (se `search_unknown=True`)
3. **Auto-Registro**: Cadastra automaticamente se não encontrar (se `auto_register=True`)

### **🏢 Arquitetura Multi-Tenant**
```
collections/
├── private/
│   ├── empresa_001/
│   │   ├── known.index       # Funcionários conhecidos
│   │   ├── known.pkl
│   │   ├── unknown.index     # Auto-detect
│   │   └── unknown.pkl
└── public/
    ├── policia_civil/
    │   ├── known.index       # Criminosos/Pessoas de interesse
    │   ├── known.pkl  
    │   ├── unknown.index     # Auto-detect
    │   └── unknown.pkl
    └── policia_militar/
        ├── known.index
        └── ...
```

### **🏢 Lógica de Busca por Tipo de Empresa e Câmera**

| Empresa | Câmera | Busca em Collections 'Known' |
|---------|--------|------------------------------|
| **Privada** | `camera_shared=false` | ✅ Apenas própria collection 'known' |
| **Privada** | `camera_shared=true` | 🌐 **TODAS** as collections 'known' públicas + própria |
| **Órgão Público** | `camera_shared=false` | ✅ Apenas própria collection 'known' |
| **Órgão Público** | `camera_shared=true` | 🌐 **TODAS** as collections 'known' públicas + própria |

**Exemplo:** Shopping center com câmera em área pública pode identificar criminosos procurados!

### **📊 Matches Detalhados por Collection**

Quando `camera_shared=true`, o sistema retorna matches organizados por **categoria** e **company**:

**Nova Estrutura Simplificada:**
```json
{
    "matches": {                        // Só presente quando há matches encontrados
        "public": {
            "policia_civil": [              // company_id do órgão público
                {"index_position": 0, "similarity": 0.95, "confidence": 95.0},
                {"index_position": 5, "similarity": 0.87, "confidence": 87.0}
            ],
            "policia_federal": [            // company_id do órgão público
                {"index_position": 2, "similarity": 0.82, "confidence": 82.0}
            ]
        },
        "private": {
            "empresa_shopping": [           // company_id da empresa privada
                {"index_position": 1, "similarity": 0.70, "confidence": 70.0}
            ]
        }
    }
}
```

**⚡ Arquitetura de Microserviços:**
- **civium-match**: Retorna apenas `index_position` (posição no índice FAISS)
- **Outro container**: Faz mapeamento `index_position` → dados PostgreSQL (`person_id`, `metadata`, etc.)
- **Benefício**: Separação clara de responsabilidades e performance otimizada

**Quando o campo `matches` aparece:**
- ✅ **`result_type: "found_known"`**: Faces encontradas nas collections 'known'
- ✅ **`result_type: "found_unknown"`**: Faces encontradas na collection 'unknown' própria
- ❌ **`result_type: "auto_registered"`**: Campo `matches` = `null` (nada encontrado)
- ❌ **`result_type: "not_found"`**: Campo `matches` = `null` (nada encontrado)

**Organização:**
- `public` / `private`: **Categoria** baseada no `company_type`
- `policia_civil`, `empresa_shopping`, etc.: **company_id** da empresa/órgão
- Array de matches: **Resultados** encontrados nesta company

**Vantagens:**
- ✅ **Organização clara**: Separação entre `public` e `private`
- ✅ **Dados reais**: Apenas `face_id`, `similarity` e `confidence` (que existem no FAISS)
- ✅ **Escalável**: Suporta múltiplas companies por categoria
- ✅ **Performance**: Não requer consultas extras ao banco

**Exemplo:** `top_k=3` + 4 companies = até **12 matches detalhados**
- `public/policia_civil`: até 3 matches  
- `public/policia_federal`: até 3 matches
- `public/policia_militar`: até 3 matches
- `private/empresa_shopping`: até 3 matches

## 📚 **API Endpoints**

### **🧠 Smart Match**
```http
POST /api/smart-match
```

**Request:**
```json
{
    "embedding": [0.1, 0.2, ...],           // 512 dimensões
    "company_id": "empresa_001",             // ID da empresa
    "company_type": "private",               // "private" ou "public_org"
    "camera_shared": false,                  // Câmera compartilhada?
    "search_unknown": true,                  // Buscar em 'unknown'?
    "auto_register": true,                   // Auto-cadastrar se não encontrar?
    "threshold": 0.4,                        // Threshold de similaridade
    "top_k": 10,                            // Máximo de resultados
    "metadata": {                           // Metadata adicional
        "source": "camera_001"
    }
}
```

**Response:**
```json
{
    "query_embedding_hash": "a1b2c3d4",
    "search_performed": {
        "company_id": "empresa_001",
        "company_type": "private",
        "camera_shared": true,
        "search_unknown": true,
        "auto_register": true,
        "top_k": 5
    },
    "result_type": "found_known",
    
    // Matches detalhados por categoria e company
    "matches": {
        "public": {                         // Categoria: órgãos públicos
            "policia_civil": [              // company_id
                {
                    "index_position": 0,
                    "similarity": 0.95,
                    "confidence": 95.0
                },
                {
                    "index_position": 5, 
                    "similarity": 0.87,
                    "confidence": 87.0
                }
                // ... mais até top_k
            ],
            "policia_federal": [            // company_id
                {
                    "index_position": 2,
                    "similarity": 0.82,
                    "confidence": 82.0
                }
                // ... mais até top_k
            ]
        },
        "private": {                        // Categoria: empresas privadas
            "empresa_001": [                // company_id
                {
                    "index_position": 1,
                    "similarity": 0.75,
                    "confidence": 75.0
                }
                // ... mais até top_k
            ]
        }
    },
    
    "auto_registered_index": null,
    "total_collections_searched": 4,
    "search_time_ms": 45.2,
    "threshold_used": 0.4,
    "top_k_used": 5
}
```

### **📋 Estrutura da Resposta**

**`matches`** é organizado em **2 níveis**:

1. **Nível 1 - Categoria**: `"public"` ou `"private"`
   - Baseado no `company_type` da empresa
   - `"public"`: órgãos públicos (`company_type: "public_org"`)
   - `"private"`: empresas privadas (`company_type: "private"`)

2. **Nível 2 - Company ID**: chave é o `company_id` real
   - Exemplos: `"policia_civil"`, `"empresa_shopping"`, `"policia_federal"`
   - Cada `company_id` contém array de matches encontrados nesta empresa

**Exemplo de navegação:**
```javascript
// Acessar matches da Polícia Civil
const matchesPoliciaCivil = response.matches.public.policia_civil;

// Acessar matches de empresa privada  
const matchesEmpresa = response.matches.private.empresa_shopping;

// Iterar todas as categories e companies
for (const [category, companies] of Object.entries(response.matches)) {
    console.log(`Categoria: ${category}`); // "public" ou "private"
    
    for (const [companyId, matches] of Object.entries(companies)) {
        console.log(`  Company: ${companyId}`);  // company_id real
        console.log(`  Matches: ${matches.length}`);
    }
}
```

### **👤 Adicionar Face**
```http
POST /api/faces
```

**Request:**
```json
{
    "embedding": [0.1, 0.2, ...],
    "company_id": "empresa_001",
    "company_type": "private",
    "collection_type": "known",              // "known" ou "unknown"
    "person_id": "pessoa_001",               // Opcional
    "metadata": {
        "name": "João Silva",
        "role": "funcionario"
    }
}
```

### **📊 Health Check**
```http
GET /health
```

### **📈 Estatísticas**
```http
GET /api/stats
```

## 🎯 **Cenários de Uso**

### **1. Empresa Privada - Busca Restrita**
```json
{
    "company_type": "private",
    "search_unknown": false,
    "auto_register": false
}
```
**Comportamento:** Busca apenas na collection 'known' da própria empresa.

### **2. Empresa Privada - Busca Completa**
```json
{
    "company_type": "private", 
    "search_unknown": true,
    "auto_register": true
}
```
**Comportamento:** Busca 'known' → 'unknown' próprias → auto-cadastra.

### **3. Órgão Público - Câmera Privada**
```json
{
    "company_type": "public_org",
    "camera_shared": false,
    "search_unknown": true,
    "auto_register": true
}
```
**Comportamento:** Busca apenas collections próprias (como empresa privada).

### **4. Órgão Público - Câmera Compartilhada**
```json
{
    "company_type": "public_org",
    "camera_shared": true,
    "search_unknown": true, 
    "auto_register": true
}
```
**Comportamento:** 🌐 **Busca federada** em TODAS as collections 'known' públicas → 'unknown' própria → auto-cadastra.

## 🛠️ **Instalação e Execução**

### **Pré-requisitos**
```bash
# Python 3.11+ e pip
pip install -r requirements.txt
```

### **Executar Serviço**
```bash
# Desenvolvimento
python main.py

# Produção
uvicorn main:app --host 0.0.0.0 --port 8002
```

### **Docker**
```bash
# Build
docker build -t civium-match .

# Run
docker run -p 8002:8002 -v $(pwd)/collections:/app/collections civium-match
```

### **Teste**
```bash
# Testar funcionalidades
python test_smart_match.py

# Teste básico
python test_basic.py
```

## ⚙️ **Configuração**

### **Variáveis de Ambiente**
```bash
# .env
DEBUG=true
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8002
EMBEDDING_DIMENSION=512
DEFAULT_MATCH_THRESHOLD=0.4
DEFAULT_TOP_K=10
ALLOWED_ORIGINS=["*"]
```

### **Ajustar Parâmetros**
```python
# app/config.py
class Settings:
    DEFAULT_MATCH_THRESHOLD: float = 0.4    # Threshold padrão
    DEFAULT_TOP_K: int = 10                  # Resultados padrão
    EMBEDDING_DIMENSION: int = 512           # Dimensão dos embeddings
```

## 📊 **Performance**

### **Otimizações FAISS**
- **IndexFlatIP**: Produto interno para similaridade cosseno
- **Busca Paralela**: Multiple collections em paralelo
- **Cache de Collections**: Collections ficam em memória
- **Lazy Loading**: Collections carregadas sob demanda

### **Benchmarks Típicos**
- **Busca Simples**: ~5-15ms (1 collection, 1K faces)
- **Busca Federada**: ~20-50ms (5 collections públicas)
- **Auto-Registro**: ~10-25ms (criação + adição)

## 🔧 **Integração com Worker**

Para integrar com o `facial_recognition_worker.py`:

```python
# Substituir lógica de identificação existente
async def identify_face_via_match_service(embedding: np.ndarray, 
                                        company_id: str, 
                                        company_type: str,
                                        camera_shared: bool = False) -> Dict:
    """Usar civium-match service para identificação."""
    
    match_request = {
        "embedding": embedding.tolist(),
        "company_id": company_id,
        "company_type": company_type,
        "camera_shared": camera_shared,
        "search_unknown": True,      # Buscar em unknown
        "auto_register": True,       # Auto-cadastrar
        "threshold": 0.4
    }
    
    response = requests.post(
        "http://civium-match:8002/api/smart-match",
        json=match_request,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result["result_type"] == "found_known":
            return {
                "action": "identified",
                "identified": True,
                "face_id": result["match"]["face_id"],
                "person_id": result["match"]["person_id"],
                "confidence": result["match"]["confidence"]
            }
        elif result["result_type"] == "auto_registered":
            return {
                "action": "auto_detected", 
                "identified": False,
                "face_id": result["auto_registered_face_id"]
            }
        else:
            return {
                "action": "not_found",
                "identified": False
            }
```

## 🎛️ **Controle de Comportamento**

| Parâmetro | Descrição | Casos de Uso |
|-----------|-----------|--------------|
| `search_unknown` | Buscar na collection 'unknown' da empresa | Evitar duplicatas, otimizar performance |
| `auto_register` | Cadastrar automaticamente se não encontrar | Empresas que querem auto-detecção vs. apenas verificação |
| `camera_shared` | Ativar busca federada para órgãos públicos | Câmeras em locais públicos vs. privados |
| `company_type` | Tipo da empresa/órgão | Determinar se pode participar de busca federada |

## 📋 **Roadmap**

- [ ] Cache Redis para performance
- [ ] Índices FAISS otimizados (IVF, HNSW)
- [ ] Métricas detalhadas (Prometheus)
- [ ] Backup/restore de collections
- [ ] Interface web de administração
- [ ] Clustering para alta disponibilidade

## 🤝 **Contribuição**

Este serviço é parte do ecossistema Civium. Para contribuições:

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/nova-funcionalidade`
3. Commit: `git commit -m 'Adiciona nova funcionalidade'`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abra um Pull Request

---

**Civium Match Service** - Reconhecimento facial inteligente com arquitetura multi-tenant 🎯 