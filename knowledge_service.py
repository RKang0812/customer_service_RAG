"""
Knowledge Service
知识库服务

This module manages the knowledge base (add, delete, list documents).
本模块管理知识库（添加、删除、列出文档）。
"""

from typing import List, Dict, Optional
from pathlib import Path
import time

from vector_store import VectorStore, Document
from llm_client import LLMClient
from document_loader import load_and_chunk_document, load_directory
from logger_config import setup_logger, log_error

logger = setup_logger(__name__)

# ============================================================================
# Knowledge Service Class / 知识库服务类
# ============================================================================

class KnowledgeService:
    """
    Manage knowledge base documents
    管理知识库文档
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient
    ):
        """
        Initialize knowledge service
        初始化知识库服务
        
        Args:
            vector_store: Vector store instance / 向量存储实例
            llm_client: LLM client for generating embeddings / 用于生成嵌入的LLM客户端
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        
        logger.info("Knowledge Service initialized")
    
    def upload_document(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict:
        """
        Upload and index a document
        上传并索引文档
        
        Args:
            file_path: Path to document file / 文档文件路径
            metadata: Additional metadata / 附加元数据
            chunk_size: Override default chunk size / 覆盖默认块大小
            chunk_overlap: Override default overlap / 覆盖默认重叠
            
        Returns:
            Result dictionary / 结果字典
        """
        start_time = time.time()
        
        try:
            logger.info(f"Uploading document: {file_path}")
            
            # Load and chunk document / 加载并分块文档
            kwargs = {}
            if chunk_size:
                kwargs['chunk_size'] = chunk_size
            if chunk_overlap:
                kwargs['chunk_overlap'] = chunk_overlap
            if metadata:
                kwargs['metadata'] = metadata
            
            documents = load_and_chunk_document(file_path, **kwargs)
            
            if not documents:
                return {
                    "success": False,
                    "message": "Failed to load document",
                    "num_chunks": 0
                }
            
            # Generate embeddings / 生成嵌入
            logger.info(f"Generating embeddings for {len(documents)} chunks")
            vectors = []
            
            for doc in documents:
                vector = self.llm_client.get_embedding(doc.content)
                vectors.append(vector)
            
            # Add to vector store / 添加到向量存储
            success = self.vector_store.add_documents(documents, vectors)
            
            duration = time.time() - start_time
            
            if success:
                logger.info(
                    f"Successfully uploaded {Path(file_path).name}: "
                    f"{len(documents)} chunks in {duration:.2f}s"
                )
                
                return {
                    "success": True,
                    "message": f"Document uploaded successfully",
                    "num_chunks": len(documents),
                    "duration": duration,
                    "file_name": Path(file_path).name
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to add documents to vector store",
                    "num_chunks": 0
                }
        
        except Exception as e:
            log_error(logger, f"upload_document({file_path})", e)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "num_chunks": 0,
                "error": str(e)
            }
    
    def upload_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None
    ) -> Dict:
        """
        Upload all documents from a directory
        从目录上传所有文档
        
        Args:
            directory_path: Path to directory / 目录路径
            file_extensions: List of extensions to include / 要包含的扩展名列表
            
        Returns:
            Result dictionary / 结果字典
        """
        start_time = time.time()
        
        try:
            logger.info(f"Uploading directory: {directory_path}")
            
            # Load all documents / 加载所有文档
            all_documents = load_directory(directory_path, file_extensions)
            
            if not all_documents:
                return {
                    "success": False,
                    "message": "No documents found in directory",
                    "num_documents": 0,
                    "num_chunks": 0
                }
            
            # Generate embeddings / 生成嵌入
            logger.info(f"Generating embeddings for {len(all_documents)} chunks")
            vectors = []
            
            for doc in all_documents:
                vector = self.llm_client.get_embedding(doc.content)
                vectors.append(vector)
            
            # Add to vector store / 添加到向量存储
            success = self.vector_store.add_documents(all_documents, vectors)
            
            duration = time.time() - start_time
            
            # Count unique source files / 计算唯一源文件数
            unique_sources = set(doc.metadata.get("source", "") for doc in all_documents)
            
            if success:
                logger.info(
                    f"Successfully uploaded directory: "
                    f"{len(unique_sources)} files, {len(all_documents)} chunks in {duration:.2f}s"
                )
                
                return {
                    "success": True,
                    "message": "Directory uploaded successfully",
                    "num_documents": len(unique_sources),
                    "num_chunks": len(all_documents),
                    "duration": duration
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to add documents to vector store",
                    "num_documents": 0,
                    "num_chunks": 0
                }
        
        except Exception as e:
            log_error(logger, f"upload_directory({directory_path})", e)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "num_documents": 0,
                "num_chunks": 0,
                "error": str(e)
            }
    
    def delete_document(self, doc_id: str) -> Dict:
        """
        Delete a document by ID
        按ID删除文档
        
        Args:
            doc_id: Document ID / 文档ID
            
        Returns:
            Result dictionary / 结果字典
        """
        try:
            success = self.vector_store.delete_by_id(doc_id)
            
            if success:
                return {
                    "success": True,
                    "message": f"Document {doc_id} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to delete document {doc_id}"
                }
        
        except Exception as e:
            log_error(logger, f"delete_document({doc_id})", e)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def delete_by_source(self, source_name: str) -> Dict:
        """
        Delete all documents from a source file
        删除来自源文件的所有文档
        
        Args:
            source_name: Source file name / 源文件名
            
        Returns:
            Result dictionary / 结果字典
        """
        try:
            success = self.vector_store.delete_by_metadata({"source": source_name})
            
            if success:
                return {
                    "success": True,
                    "message": f"All documents from {source_name} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to delete documents from {source_name}"
                }
        
        except Exception as e:
            log_error(logger, f"delete_by_source({source_name})", e)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def list_documents(self, limit: int = 100) -> List[Dict]:
        """
        List all documents in the knowledge base
        列出知识库中的所有文档
        
        Args:
            limit: Maximum number of documents / 最大文档数量
            
        Returns:
            List of document info dicts / 文档信息字典列表
        """
        try:
            documents = self.vector_store.list_all_documents(limit)
            
            # Convert to info dicts / 转换为信息字典
            doc_list = []
            sources_seen = set()
            
            for doc in documents:
                source = doc.metadata.get("source", "Unknown")
                
                # Group by source / 按来源分组
                if source not in sources_seen:
                    sources_seen.add(source)
                    
                    # Count chunks from this source / 计算此来源的块数
                    chunk_count = sum(
                        1 for d in documents
                        if d.metadata.get("source") == source
                    )
                    
                    doc_list.append({
                        "source": source,
                        "num_chunks": chunk_count,
                        "category": doc.metadata.get("category", "general"),
                        "first_chunk_id": doc.doc_id
                    })
            
            logger.info(f"Listed {len(doc_list)} unique sources")
            return doc_list
        
        except Exception as e:
            log_error(logger, "list_documents", e)
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get knowledge base statistics
        获取知识库统计信息
        
        Returns:
            Statistics dictionary / 统计字典
        """
        try:
            info = self.vector_store.get_collection_info()
            documents = self.list_documents()
            
            return {
                "total_chunks": info.get("points_count", 0),
                "total_documents": len(documents),
                "collection_name": info.get("name", ""),
                "status": info.get("status", "unknown")
            }
        
        except Exception as e:
            log_error(logger, "get_collection_stats", e)
            return {}

# ============================================================================
# Convenience Functions / 便利函数
# ============================================================================

def create_knowledge_service(
    vector_store: VectorStore,
    llm_client: LLMClient
) -> KnowledgeService:
    """
    Create and return knowledge service instance
    创建并返回知识库服务实例
    
    Args:
        vector_store: Vector store instance / 向量存储实例
        llm_client: LLM client instance / LLM客户端实例
        
    Returns:
        KnowledgeService instance / 知识库服务实例
    """
    return KnowledgeService(vector_store, llm_client)

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
        knowledge_service = create_knowledge_service(vector_store, llm_client)
        
        # Get stats / 获取统计
        stats = knowledge_service.get_collection_stats()
        print(f"Knowledge Base Statistics:")
        print(f"  Total documents: {stats.get('total_documents', 0)}")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Status: {stats.get('status', 'unknown')}\n")
        
        # List documents / 列出文档
        documents = knowledge_service.list_documents()
        print(f"Documents in knowledge base ({len(documents)}):")
        for doc in documents:
            print(f"  - {doc['source']} ({doc['num_chunks']} chunks)")
        
    except Exception as e:
        print(f"Error: {e}")
