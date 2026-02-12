"""
Configuration and Prompt Templates

"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# OpenAI Configuration
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# ============================================================================
# Qdrant Configuration
# ============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "customer_service_docs")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # Optional / 可选

# Vector dimension for text-embedding-3-small / text-embedding-3-small的向量维度
VECTOR_DIMENSION = 1536

# ============================================================================
# RAG Parameters
# ============================================================================

TOP_K = int(os.getenv("TOP_K", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Minimum number of relevant documents before triggering query rewrite
MIN_RELEVANT_DOCS = 2

# ============================================================================
# System Settings
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "中文"
}

# ============================================================================
# Prompt Templates 
# ============================================================================

# Relevance Grading Prompt
RELEVANCE_GRADING_PROMPT = """You are evaluating the relevance of a retrieved document to a user question.

Question: {question}

Document:
{document}

Evaluate if this document contains information that can help answer the question.
Respond with a JSON object in the following format:
{{
    "is_relevant": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}

Be strict - only mark as relevant if the document actually addresses the question.
"""

# Query Rewriting Prompt (Few-shot)
QUERY_REWRITING_PROMPT = """You are a query rewriting assistant. Your task is to rewrite user queries to improve document retrieval.

Examples:
- Original: "refund?"
  Rewritten: "What is the refund policy? How do I request a refund?"

- Original: "not working"
  Rewritten: "Product not functioning properly. Troubleshooting steps and common issues."

- Original: "账号问题"
  Rewritten: "账户相关问题：登录、密码重置、账号注册流程"

Now rewrite this query:
Original: {query}

Rewritten query (preserve the original language):"""

# Answer Generation Prompt
ANSWER_GENERATION_PROMPT = """You are a helpful customer service assistant. Use the provided context to answer the user's question.

Context Documents:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources by mentioning document names in [brackets]
3. If the context doesn't contain enough information, clearly state what information is missing
4. Be concise but complete
5. Use a professional and friendly tone
6. If asked in Chinese, respond in Chinese; if in English, respond in English

Answer:"""

# Answer Generation with History
ANSWER_GENERATION_WITH_HISTORY_PROMPT = """You are a helpful customer service assistant. Use the provided context and conversation history to answer the user's question.

Conversation History:
{history}

Context Documents:
{context}

Current Question: {question}

Instructions:
1. Consider the conversation history for context
2. Answer based on the provided context documents
3. Cite sources by mentioning document names in [brackets]
4. If the context doesn't contain enough information, clearly state what information is missing
5. Be concise but complete
6. Use a professional and friendly tone
7. Maintain language consistency with the conversation

Answer:"""

# Unable to Answer Template
UNABLE_TO_ANSWER_TEMPLATE = {
    "en": """I apologize, but I couldn't find enough information in our knowledge base to answer your question completely.

What I'm missing:
{missing_info}

Would you like me to:
1. Connect you with a human agent
2. Rephrase your question for better results
3. Search for related information""",
    
    "zh": """抱歉，我在知识库中未能找到足够的信息来完整回答您的问题。

缺少的信息：
{missing_info}

您希望我：
1. 为您转接人工客服
2. 换个方式重新提问以获得更好的结果
3. 搜索相关信息"""
}

# ============================================================================
# UI Text
# ============================================================================

UI_TEXT = {
    "en": {
        "app_title": "🤖 Customer Service RAG System",
        "app_subtitle": "Intelligent customer support with source citation",
        "chat_tab": "💬 Chat",
        "knowledge_tab": "📚 Knowledge Base",
        "settings_tab": "⚙️ Settings",
        "user_input_placeholder": "Ask a question...",
        "sources": "Sources",
        "confidence": "Confidence",
        "upload_doc": "Upload Document",
        "delete_doc": "Delete Document",
        "doc_list": "Document List",
        "no_docs": "No documents found",
        "processing": "Processing...",
        "error": "Error",
        "success": "Success"
    },
    "zh": {
        "app_title": "🤖 客服RAG系统",
        "app_subtitle": "带来源引用的智能客户支持",
        "chat_tab": "💬 对话",
        "knowledge_tab": "📚 知识库",
        "settings_tab": "⚙️ 设置",
        "user_input_placeholder": "提出问题...",
        "sources": "来源",
        "confidence": "置信度",
        "upload_doc": "上传文档",
        "delete_doc": "删除文档",
        "doc_list": "文档列表",
        "no_docs": "未找到文档",
        "processing": "处理中...",
        "error": "错误",
        "success": "成功"
    }
}

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """
    Validate critical configuration values
    """
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")
    
    if TOP_K < 1 or TOP_K > 20:
        errors.append(f"TOP_K must be between 1 and 20, got {TOP_K}")
    
    if RELEVANCE_THRESHOLD < 0 or RELEVANCE_THRESHOLD > 1:
        errors.append(f"RELEVANCE_THRESHOLD must be between 0 and 1, got {RELEVANCE_THRESHOLD}")
    
    if CHUNK_SIZE < 100 or CHUNK_SIZE > 2000:
        errors.append(f"CHUNK_SIZE must be between 100 and 2000, got {CHUNK_SIZE}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(errors))

# Validate on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️ Configuration Warning: {e}")
