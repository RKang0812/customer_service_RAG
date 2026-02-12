"""
Retriever
"""

from typing import List, Optional, Dict, Tuple
import time

from vector_store import VectorStore, Document
from llm_client import LLMClient
from config import TOP_K
from logger_config import setup_logger, log_retrieval, log_error

logger = setup_logger(__name__)

# ============================================================================
# Retriever Class
# ============================================================================

class Retriever:
    """
    Retrieve relevant documents from vector database
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        top_k: int = TOP_K
    ):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store instance
            llm_client: LLM client for generating embeddings 
            top_k: Number of documents to retrieve
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
        
        Args:
            query: User query text
            top_k: Override default top_k 
            filter_dict: Optional metadata filters
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        start_time = time.time()
        k = top_k or self.top_k
        
        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: '{query[:50]}...'")
            query_vector = self.llm_client.get_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                filter_dict=filter_dict
            )
            
            # Log retrieval
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
            query: User query text
            top_k: Override default top_k 
            filter_dict: Optional metadata filters
            
        Returns:
            List of Document objects
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
        
        Args:
            query: User query tex
            category: Category to filter by
            top_k: Override default top_k
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        filter_dict = {"category": category}
        return self.retrieve(query, top_k, filter_dict)
    
    def get_retrieval_stats(self) -> Dict:
        """
        Get retrieval statistics

        
        Returns:
            Dictionary with stats
        """
        collection_info = self.vector_store.get_collection_info()
        return {
            "total_documents": collection_info.get("points_count", 0),
            "top_k": self.top_k,
            "collection_name": self.vector_store.collection_name
        }

# ============================================================================
# Convenience Functions
# ============================================================================

def create_retriever(
    vector_store: VectorStore,
    llm_client: LLMClient,
    **kwargs
) -> Retriever:
    """
    Create and return retriever instance
    
    Args:
        vector_store: Vector store instance
        llm_client: LLM client instance
        **kwargs: Additional arguments
        
    Returns:
        Retriever instance
    """
    return Retriever(vector_store, llm_client, **kwargs)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from vector_store import create_vector_store
    from llm_client import create_llm_client
    
    try:
        # Initialize components
        vector_store = create_vector_store()
        llm_client = create_llm_client()
        retriever = create_retriever(vector_store, llm_client, top_k=3)
        
        # Test retrieval
        query = "What is the refund policy?"
        results = retriever.retrieve(query)
        
        print(f"\nQuery: {query}")
        print(f"Retrieved {len(results)} documents:\n")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Content: {doc.content[:100]}...")
            print(f"   Metadata: {doc.metadata}\n")
        
        # Get stats
        stats = retriever.get_retrieval_stats()
        print(f"Retrieval stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
