# Customer Service RAG Application

A structured Retrieval-Augmented Generation (RAG) application implementing a corrective retrieval pipeline to improve answer reliability and reduce hallucinations.
This project demonstrates practical RAG system design, vector database integration, query rewriting, and relevance-based document filtering.

[▶ Watch the Demo](demo.mp4)

## Overview

This application implements a corrective RAG workflow for customer service scenarios.

It supports:

- Context-aware multi-turn conversations
- Relevance grading and document filtering
- Query rewriting when retrieval quality is insufficient
- Source citation with confidence scoring
- Persistent vector storage with Qdrant
- The focus of this project is system design and retrieval quality optimization rather than UI complexity.

## Architecture

The system is organized into modular services to separate retrieval, ranking, generation, and storage.

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

## Technical Stack

- Python 3.8+
- OpenAI GPT models
- OpenAI Embeddings API
- Qdrant (Vector Database)
- Streamlit (UI layer)
- Docker / Docker Compose

## Engineering Highlights

- Corrective RAG pipeline with query rewriting
- Modular service-based architecture
- Separation of retrieval, ranking, and generation logic
- Configurable retrieval parameters
- Persistent vector storage
- Confidence-aware response generation
- Multi-turn conversation context handling

## Quick Start

### Setup

1. **Clone the repositor**
```bash
git clone <your-repo-url>
cd customer-service-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. **Start Qdrant**
```bash
docker-compose up -d
```

6. **Initialize vector database**
```bash
python scripts/init_vector_db.py
```

7. **Load knowledge base**
```bash
python scripts/load_knowledge_base.py
```

8. **Run the application**
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## Project Structure

```
customer-service-rag/
│
├── app.py
├── chat_service.py
├── knowledge_service.py
├── retriever.py
├── reranker.py
├── query_rewriter.py
├── generator.py
├── vector_store.py
├── llm_client.py
├── config.py
├── document_loader.py
├── logger_config.py
├── scripts/
│   ├── init_vector_db.py
│   └── load_knowledge_base.py
├── data/
└── requirements.txt
```

## Design Considerations

- Corrective retrieval to reduce hallucination risk
- Query rewriting to improve recall
- Separation of retrieval and generation for maintainability
- Configurable hyperparameters for experimentation
- Persistent vector storage for production-like behavior

## Corrective RAG Workflow

```
1. User Query → Initial Retrieval
2. Relevance Grading → Filter Documents
3. Decision: Enough Relevant Docs?
   ├─ YES → Generate Answer
   └─ NO  → Rewrite Query → Re-retrieve → Generate Answer
```

