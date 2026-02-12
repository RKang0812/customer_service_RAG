"""
Qdrant Vector Store
Qdrant向量存储

This module handles all interactions with the Qdrant vector database.
本模块处理与Qdrant向量数据库的所有交互。

Fixed for qdrant-client 1.16.2+
"""

from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
import uuid

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_API_KEY,
    VECTOR_DIMENSION
)
from logger_config import setup_logger, log_error

logger = setup_logger(__name__)

# ============================================================================
# Document Class / 文档类
# ============================================================================

class Document:
    """
    Represents a document with content and metadata
    表示带内容和元数据的文档
    """
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ):
        """
        Initialize document
        初始化文档
        
        Args:
            content: Document text content / 文档文本内容
            metadata: Additional metadata / 附加元数据
            doc_id: Unique document ID / 唯一文档ID
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or str(uuid.uuid4())
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, content_length={len(self.content)})"

# ============================================================================
# Vector Store Class / 向量存储类
# ============================================================================

class VectorStore:
    """
    Interface for Qdrant vector database operations
    Qdrant向量数据库操作接口
    """
    
    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = QDRANT_COLLECTION,
        api_key: Optional[str] = QDRANT_API_KEY
    ):
        """
        Initialize vector store connection
        初始化向量存储连接
        
        Args:
            host: Qdrant server host / Qdrant服务器主机
            port: Qdrant server port / Qdrant服务器端口
            collection_name: Name of the collection / 集合名称
            api_key: API key for authentication / 用于认证的API密钥
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        
        try:
            # Initialize Qdrant client / 初始化Qdrant客户端
            if api_key:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    timeout=30
                )
            else:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    timeout=30
                )
            
            logger.info(f"Connected to Qdrant at {host}:{port}")
            
            # Create collection if it doesn't exist / 如果集合不存在则创建
            self._ensure_collection_exists()
            
        except Exception as e:
            log_error(logger, "vector_store_init", e)
            raise
    
    def _ensure_collection_exists(self):
        """
        Create collection if it doesn't exist
        如果集合不存在则创建
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=VECTOR_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
        
        except Exception as e:
            log_error(logger, "ensure_collection_exists", e)
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        vectors: List[List[float]]
    ) -> bool:
        """
        Add documents with their vectors to the collection
        将文档及其向量添加到集合
        
        Args:
            documents: List of Document objects / Document对象列表
            vectors: List of embedding vectors / 嵌入向量列表
            
        Returns:
            Success status / 成功状态
        """
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors / 文档数量必须与向量数量匹配")
        
        try:
            points = []
            for doc, vector in zip(documents, vectors):
                point = PointStruct(
                    id=doc.doc_id,
                    vector=vector,
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to {self.collection_name}")
            return True
        
        except Exception as e:
            log_error(logger, "add_documents", e)
            return False
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents
        搜索相似文档
        
        Args:
            query_vector: Query embedding vector / 查询嵌入向量
            top_k: Number of results to return / 返回的结果数量
            filter_dict: Optional metadata filters / 可选的元数据过滤器
            
        Returns:
            List of (Document, score) tuples / (Document, 分数)元组列表
        """
        try:
            # Build filter if provided / 如果提供则构建过滤器
            search_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                search_filter = Filter(must=conditions)
            
            # Perform search using query method / 使用query方法执行搜索
            # In qdrant-client 1.16+, use query() instead of search()
            # 在qdrant-client 1.16+中，使用query()而不是search()
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True
            )
            
            # Convert to Document objects / 转换为Document对象
            documents_with_scores = []
            
            # Handle different response formats / 处理不同的响应格式
            points = results.points if hasattr(results, 'points') else results
            
            for result in points:
                doc = Document(
                    content=result.payload.get("content", ""),
                    metadata=result.payload.get("metadata", {}),
                    doc_id=str(result.id)
                )
                # Score is in result.score / 分数在result.score中
                score = result.score if hasattr(result, 'score') else 0.0
                documents_with_scores.append((doc, score))
            
            logger.debug(f"Found {len(documents_with_scores)} similar documents")
            return documents_with_scores
        
        except Exception as e:
            log_error(logger, "search", e)
            return []
    
    def delete_by_id(self, doc_id: str) -> bool:
        """
        Delete document by ID
        按ID删除文档
        
        Args:
            doc_id: Document ID to delete / 要删除的文档ID
            
        Returns:
            Success status / 成功状态
        """
        try:
            from qdrant_client.models import PointIdsList
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[doc_id])
            )
            logger.info(f"Deleted document: {doc_id}")
            return True
        
        except Exception as e:
            log_error(logger, "delete_by_id", e)
            return False
    
    def delete_by_metadata(self, metadata_filter: Dict) -> bool:
        """
        Delete documents matching metadata filter
        删除匹配元数据过滤器的文档
        
        Args:
            metadata_filter: Metadata key-value pairs / 元数据键值对
            
        Returns:
            Success status / 成功状态
        """
        try:
            from qdrant_client.models import FilterSelector
            
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            
            filter_obj = Filter(must=conditions)
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=filter_obj)
            )
            
            logger.info(f"Deleted documents matching filter: {metadata_filter}")
            return True
        
        except Exception as e:
            log_error(logger, "delete_by_metadata", e)
            return False
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection
        获取集合的信息
        
        Returns:
            Collection information / 集合信息
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status
            }
        
        except Exception as e:
            log_error(logger, "get_collection_info", e)
            return {}
    
    def list_all_documents(self, limit: int = 100) -> List[Document]:
        """
        List all documents in the collection
        列出集合中的所有文档
        
        Args:
            limit: Maximum number of documents to return / 返回的最大文档数量
            
        Returns:
            List of documents / 文档列表
        """
        try:
            # Scroll through all points / 滚动浏览所有点
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            documents = []
            for point in points:
                doc = Document(
                    content=point.payload.get("content", ""),
                    metadata=point.payload.get("metadata", {}),
                    doc_id=str(point.id)
                )
                documents.append(doc)
            
            logger.info(f"Listed {len(documents)} documents")
            return documents
        
        except Exception as e:
            log_error(logger, "list_all_documents", e)
            return []

# ============================================================================
# Convenience Functions / 便利函数
# ============================================================================

def create_vector_store(**kwargs) -> VectorStore:
    """
    Create and return vector store instance
    创建并返回向量存储实例
    
    Args:
        **kwargs: Arguments to pass to VectorStore / 传递给VectorStore的参数
        
    Returns:
        Vector store instance / 向量存储实例
    """
    return VectorStore(**kwargs)

# ============================================================================
# Example Usage / 使用示例
# ============================================================================

if __name__ == "__main__":
    # Test vector store / 测试向量存储
    try:
        store = VectorStore()
        
        # Get collection info / 获取集合信息
        info = store.get_collection_info()
        print(f"Collection info: {info}\n")
        
        # Create test documents / 创建测试文档
        test_docs = [
            Document(
                content="Our refund policy allows returns within 30 days.",
                metadata={"category": "policy", "source": "refund_policy.txt"}
            ),
            Document(
                content="To reset your password, click on 'Forgot Password'.",
                metadata={"category": "account", "source": "account_help.txt"}
            )
        ]
        
        # For actual use, you would generate real embeddings / 实际使用时，你需要生成真实的嵌入
        # test_vectors = [llm_client.get_embedding(doc.content) for doc in test_docs]
        
        print("Vector store initialized successfully")
        
    except Exception as e:
        print(f"Error: {e}")