"""
Retriever
检索器

This module retrieves relevant documents from the vector database.
本模块从向量数据库中检索相关文档。
"""

from typing import List, Optional, Dict, Tuple
import time

from vector_store import VectorStore, Document
from llm_client import LLMClient
from config import TOP_K
from logger_config import setup_logger, log_retrieval, log_error

logger = setup_logger(__name__)

# ============================================================================
# Retriever Class / 检索器类
# ============================================================================

class Retriever:
    """
    Retrieve relevant documents from vector database
    从向量数据库检索相关文档
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        top_k: int = TOP_K
    ):
        """
        Initialize retriever
        初始化检索器
        
        Args:
            vector_store: Vector store instance / 向量存储实例
            llm_client: LLM client for generating embeddings / 用于生成嵌入的LLM客户端
            top_k: Number of documents to retrieve / 要检索的文档数量
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k
        
        logger.info(f"Retriever initialized with top_k={top_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query
        为查询检索相关文档
        
        Args:
            query: User query text / 用户查询文本
            top_k: Override default top_k / 覆盖默认top_k
            filter_dict: Optional metadata filters / 可选的元数据过滤器
            
        Returns:
            List of (Document, similarity_score) tuples / (Document, 相似度分数)元组列表
        """
        start_time = time.time()
        k = top_k or self.top_k
        
        try:
            # Generate query embedding / 生成查询嵌入
            logger.debug(f"Generating embedding for query: '{query[:50]}...'")
            query_vector = self.llm_client.get_embedding(query)
            
            # Search vector store / 搜索向量存储
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                filter_dict=filter_dict
            )
            
            # Log retrieval / 记录检索
            duration = time.time() - start_time
            log_retrieval(logger, query, len(results), duration)
            
            return results
        
        except Exception as e:
            log_error(logger, "retrieve", e)
            return []
    
    def retrieve_documents_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve documents without scores
        检索文档但不包含分数
        
        Args:
            query: User query text / 用户查询文本
            top_k: Override default top_k / 覆盖默认top_k
            filter_dict: Optional metadata filters / 可选的元数据过滤器
            
        Returns:
            List of Document objects / Document对象列表
        """
        results = self.retrieve(query, top_k, filter_dict)
        return [doc for doc, score in results]
    
    def retrieve_by_category(
        self,
        query: str,
        category: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents filtered by category
        检索按类别过滤的文档
        
        Args:
            query: User query text / 用户查询文本
            category: Category to filter by / 要过滤的类别
            top_k: Override default top_k / 覆盖默认top_k
            
        Returns:
            List of (Document, similarity_score) tuples / (Document, 相似度分数)元组列表
        """
        filter_dict = {"category": category}
        return self.retrieve(query, top_k, filter_dict)
    
    def get_retrieval_stats(self) -> Dict:
        """
        Get retrieval statistics
        获取检索统计信息
        
        Returns:
            Dictionary with stats / 包含统计信息的字典
        """
        collection_info = self.vector_store.get_collection_info()
        return {
            "total_documents": collection_info.get("points_count", 0),
            "top_k": self.top_k,
            "collection_name": self.vector_store.collection_name
        }

# ============================================================================
# Convenience Functions / 便利函数
# ============================================================================

def create_retriever(
    vector_store: VectorStore,
    llm_client: LLMClient,
    **kwargs
) -> Retriever:
    """
    Create and return retriever instance
    创建并返回检索器实例
    
    Args:
        vector_store: Vector store instance / 向量存储实例
        llm_client: LLM client instance / LLM客户端实例
        **kwargs: Additional arguments / 额外参数
        
    Returns:
        Retriever instance / 检索器实例
    """
    return Retriever(vector_store, llm_client, **kwargs)

# ============================================================================
# Example Usage / 使用示例
# ============================================================================

if __name__ == "__main__":
    from vector_store import create_vector_store
    from llm_client import create_llm_client
    
    try:
        # Initialize components / 初始化组件
        vector_store = create_vector_store()
        llm_client = create_llm_client()
        retriever = create_retriever(vector_store, llm_client, top_k=3)
        
        # Test retrieval / 测试检索
        query = "What is the refund policy?"
        results = retriever.retrieve(query)
        
        print(f"\nQuery: {query}")
        print(f"Retrieved {len(results)} documents:\n")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Content: {doc.content[:100]}...")
            print(f"   Metadata: {doc.metadata}\n")
        
        # Get stats / 获取统计
        stats = retriever.get_retrieval_stats()
        print(f"Retrieval stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
