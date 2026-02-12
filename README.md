# Customer Service RAG System
# 客服RAG系统

A production-ready customer service system using Corrective RAG to reduce hallucinations and improve answer quality.

一个使用矫正式RAG的生产就绪客服系统，用于减少幻觉并提高答案质量。

![System Architecture](https://via.placeholder.com/800x200/2196F3/FFFFFF?text=Customer+Service+RAG+System)

## 🌟 Features / 功能特性

✅ **Corrective RAG Pipeline** - Automatic relevance grading and query rewriting  
   矫正式RAG流程 - 自动相关性评分和查询重写

✅ **Source Citation** - Every answer includes confidence scores and sources  
   来源引用 - 每个答案包含置信度分数和来源

✅ **Bilingual Support** - English and Chinese interface and processing  
   双语支持 - 英中文界面和处理

✅ **Knowledge Base Management** - Upload, delete, and organize documents  
   知识库管理 - 上传、删除和组织文档

✅ **Persistent Storage** - Qdrant vector database for reliable storage  
   持久化存储 - Qdrant向量数据库提供可靠存储

✅ **Conversation History** - Context-aware multi-turn conversations  
   对话历史 - 上下文感知的多轮对话

## 📋 System Architecture / 系统架构

```
User Input → Streamlit Interface → Chat Service
                ↓
        ┌───────┴────────┬─────────┬──────────┐
        ↓                ↓         ↓          ↓
    Retriever      Reranker  Query Rewriter  Generator
        ↓                ↓         ↓          ↓
        └────────────────┴─────────┴──────────┘
                        ↓
                  OpenAI API + Qdrant
                        ↓
                 Final Response
```

## 🚀 Quick Start / 快速开始

### Prerequisites / 前提条件

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key

### Installation / 安装

1. **Clone the repository / 克隆仓库**
```bash
git clone <your-repo-url>
cd customer-service-rag
```

2. **Create virtual environment / 创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies / 安装依赖**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables / 设置环境变量**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# 编辑.env并添加你的OPENAI_API_KEY
```

5. **Start Qdrant / 启动Qdrant**
```bash
docker-compose up -d
```

6. **Initialize vector database / 初始化向量数据库**
```bash
python scripts/init_vector_db.py
```

7. **Load knowledge base / 加载知识库**
```bash
python scripts/load_knowledge_base.py
```

8. **Run the application / 运行应用**
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 📁 Project Structure / 项目结构

```
customer-service-rag/
├── app.py                    # Streamlit application / 主应用
├── chat_service.py          # RAG pipeline orchestration / RAG流程编排
├── knowledge_service.py     # Knowledge base management / 知识库管理
├── retriever.py             # Document retrieval / 文档检索
├── reranker.py              # Relevance scoring / 相关性评分
├── query_rewriter.py        # Query rewriting / 查询重写
├── generator.py             # Answer generation / 答案生成
├── vector_store.py          # Qdrant interface / Qdrant接口
├── llm_client.py            # OpenAI wrapper / OpenAI封装
├── config.py                # Configuration / 配置
├── document_loader.py       # Document processing / 文档处理
├── logger_config.py         # Logging setup / 日志配置
├── data/
│   ├── knowledge_base/      # Sample documents / 示例文档
│   └── uploads/             # User uploads / 用户上传
├── scripts/
│   ├── init_vector_db.py    # Initialize Qdrant / 初始化Qdrant
│   └── load_knowledge_base.py  # Load documents / 加载文档
└── requirements.txt         # Dependencies / 依赖项
```

## 🔧 Configuration / 配置

Edit `.env` file to customize:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# RAG Parameters
TOP_K=5
RELEVANCE_THRESHOLD=0.7
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## 📚 Usage Examples / 使用示例

### Chat Interface / 对话界面

1. Open the application at `http://localhost:8501`
2. Type your question in English or Chinese
3. View the answer with sources and confidence score
4. Continue the conversation with context awareness

### Knowledge Base Management / 知识库管理

1. Navigate to "Knowledge Base" tab
2. View existing documents and statistics
3. Upload new documents (.txt, .pdf, .docx)
4. Delete documents you no longer need

### Programmatic Usage / 编程使用

```python
from vector_store import create_vector_store
from llm_client import create_llm_client
from chat_service import create_chat_service

# Initialize services
vector_store = create_vector_store()
llm_client = create_llm_client()
chat_service = create_chat_service(vector_store, llm_client)

# Process query
response = chat_service.process_query("What is the refund policy?")

print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']}")
print(f"Sources: {response['sources']}")
```

## 🔄 Corrective RAG Workflow / 矫正式RAG工作流

```
1. User Query → Initial Retrieval
   用户查询 → 初始检索

2. Relevance Grading → Filter Documents
   相关性评分 → 过滤文档

3. Decision: Enough Relevant Docs?
   判断：相关文档是否足够？
   
   ├─ YES → Generate Answer
   │         生成答案
   │
   └─ NO  → Rewrite Query → Re-retrieve → Generate Answer
            重写查询 → 重新检索 → 生成答案
```

## 📊 Performance / 性能

- **Response Time**: < 3s (P95) / 响应时间
- **Retrieval Accuracy**: > 80% / 检索准确率
- **Answer Relevance**: > 85% / 答案相关性

## 🤝 Contributing / 贡献

Contributions are welcome! Please feel free to submit a Pull Request.

欢迎贡献！请随时提交Pull Request。

## 📝 License / 许可证

This project is open source and available under the MIT License.

本项目是开源的，采用MIT许可证。

## 🙏 Acknowledgments / 致谢

- OpenAI for GPT-4 and embeddings API
- Qdrant for vector database
- Streamlit for the amazing UI framework
- LangChain for RAG utilities

## 📧 Contact / 联系方式

For questions or support, please open an issue on GitHub.

如有问题或需要支持，请在GitHub上提出issue。

---

**Note**: This is a demonstration project for job applications. Ensure you have appropriate API keys and comply with all terms of service.

**注意**：这是一个用于求职申请的演示项目。请确保你有适当的API密钥并遵守所有服务条款。
