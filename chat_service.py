"""
Chat Service
This module orchestrates the Corrective RAG pipeline.

"""

from typing import Dict, List, Optional
import time

from vector_store import VectorStore, Document
from llm_client import LLMClient
from retriever import Retriever
from reranker import Reranker
from query_rewriter import QueryRewriter
from generator import Generator
from config import MIN_RELEVANT_DOCS
from logger_config import setup_logger, log_error

logger = setup_logger(__name__)

# ============================================================================
# Chat Service Class
# ============================================================================

class ChatService:
    """
    Orchestrates the Corrective RAG pipeline
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        retriever: Retriever,
        reranker: Reranker,
        query_rewriter: QueryRewriter,
        generator: Generator,
        min_relevant_docs: int = MIN_RELEVANT_DOCS
    ):
        """
        Initialize chat service
        
        Args:
            vector_store: Vector store instance 
            llm_client: LLM client instance
            retriever: Retriever instance
            reranker: Reranker instance 
            query_rewriter: Query rewriter instance 
            generator: Generator instance
            min_relevant_docs: Minimum relevant docs before rewriting 
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.retriever = retriever
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.generator = generator
        self.min_relevant_docs = min_relevant_docs
        
        logger.info(f"Chat Service initialized with min_relevant_docs={min_relevant_docs}")
    
    def corrective_rag_pipeline(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        language: Optional[str] = None
    ) -> Dict:
        """
        Main Corrective RAG workflow
        
        
        Workflow:
        1. Retrieve documents 
        2. Grade relevance / 评估相关性
        3. If not enough relevant docs: rewrite query & re-retrieve
        4. Generate answer 
        
        Args:
            user_query: User's question 
            conversation_history: Previous messages
            language: Query language (auto-detected if None) 
            
        Returns:
            Response dictionary with answer, confidence, sources, etc.
        """
        start_time = time.time()
        pipeline_log = []  # Track pipeline steps
        
        try:
            logger.info(f"Starting Corrective RAG for query: '{user_query}'")
            
            # Detect language if not provided
            if language is None:
                language = self.query_rewriter.detect_language(user_query)
            
            pipeline_log.append(f"Detected language: {language}")
            
            # ========================================================================
            # Step 1: Initial Retrieval
            # ========================================================================
            
            logger.info("Step 1: Initial retrieval")
            retrieved_docs = self.retriever.retrieve(user_query)
            
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                pipeline_log.append("Initial retrieval: 0 documents")
                
                return {
                    "answer": self._get_no_docs_message(language),
                    "confidence": 0.0,
                    "sources": [],
                    "unable_to_answer": True,
                    "pipeline_log": pipeline_log,
                    "query_rewritten": False,
                    "duration": time.time() - start_time
                }
            
            docs_only = [doc for doc, score in retrieved_docs]
            pipeline_log.append(f"Initial retrieval: {len(docs_only)} documents")
            
            # ========================================================================
            # Step 2: Grade Relevance 
            # ========================================================================
            
            logger.info("Step 2: Grading relevance")
            graded_docs = self.reranker.grade_documents(user_query, docs_only)
            relevant_docs = self.reranker.filter_by_threshold(graded_docs)
            
            pipeline_log.append(
                f"Relevance grading: {len(relevant_docs)}/{len(docs_only)} documents relevant"
            )
            
            # ========================================================================
            # Step 3: Query Rewriting (if needed)
            # ========================================================================
            
            query_rewritten = False
            final_query = user_query
            
            if len(relevant_docs) < self.min_relevant_docs:
                logger.warning(
                    f"Only {len(relevant_docs)} relevant docs found (threshold: {self.min_relevant_docs}). "
                    "Rewriting query..."
                )
                pipeline_log.append("Triggered query rewriting")
                
                # Rewrite query
                rewritten_query = self.query_rewriter.rewrite(user_query, language)
                query_rewritten = True
                final_query = rewritten_query
                
                pipeline_log.append(f"Rewritten query: '{rewritten_query}'")
                
                # Re-retrieve with rewritten query
                logger.info("Step 3a: Re-retrieving with rewritten query")
                retrieved_docs = self.retriever.retrieve(rewritten_query)
                
                if not retrieved_docs:
                    logger.warning("No documents retrieved after rewriting")
                    pipeline_log.append("Re-retrieval: 0 documents")
                    
                    return {
                        "answer": self._get_no_docs_message(language),
                        "confidence": 0.0,
                        "sources": [],
                        "unable_to_answer": True,
                        "pipeline_log": pipeline_log,
                        "query_rewritten": True,
                        "rewritten_query": rewritten_query,
                        "duration": time.time() - start_time
                    }
                
                docs_only = [doc for doc, score in retrieved_docs]
                pipeline_log.append(f"Re-retrieval: {len(docs_only)} documents")
                
                # Re-grade with original query
                logger.info("Step 3b: Re-grading with original query")
                graded_docs = self.reranker.grade_documents(user_query, docs_only)
                relevant_docs = self.reranker.filter_by_threshold(graded_docs)
                
                pipeline_log.append(
                    f"Re-grading: {len(relevant_docs)}/{len(docs_only)} documents relevant"
                )
            
            # ========================================================================
            # Step 4: Generate Answer
            # ========================================================================
            
            logger.info("Step 4: Generating answer")
            
            # Extract documents only (without scores)
            final_docs = [doc for doc, score in relevant_docs]
            
            # Generate answer
            response = self.generator.generate(
                query=user_query,  # Use original query for generation
                documents=final_docs,
                history=conversation_history
            )
            
            # Add pipeline metadata
            response["pipeline_log"] = pipeline_log
            response["query_rewritten"] = query_rewritten
            if query_rewritten:
                response["rewritten_query"] = final_query
            response["duration"] = time.time() - start_time
            
            logger.info(
                f"Corrective RAG completed in {response['duration']:.2f}s "
                f"(confidence: {response['confidence']:.2f})"
            )
            
            return response
        
        except Exception as e:
            log_error(logger, "corrective_rag_pipeline", e)
            
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "unable_to_answer": True,
                "pipeline_log": pipeline_log + [f"Error: {str(e)}"],
                "query_rewritten": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def process_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Process user query (convenience method)
        
        Args:
            query: User query 
            history: Conversation history 
            
        Returns:
            Response dictionary
        """
        return self.corrective_rag_pipeline(query, history)
    
    def _get_no_docs_message(self, language: str) -> str:
        """
        Get message for when no documents are found
        
        Args:
            language: Response language
            
        Returns:
            Message text
        """
        messages = {
            "en": "I apologize, but I couldn't find any relevant information in the knowledge base to answer your question. Could you please rephrase your question or ask something else?",
            "zh": "抱歉，我在知识库中找不到相关信息来回答您的问题。您能换个方式重新提问或询问其他内容吗？"
        }
        
        return messages.get(language, messages["en"])
    
    def get_stats(self) -> Dict:
        """
        Get service statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "retriever": self.retriever.get_retrieval_stats(),
            "llm_usage": self.llm_client.get_usage_stats(),
            "vector_store": self.vector_store.get_collection_info()
        }

# ============================================================================
# Convenience Functions
# ============================================================================

def create_chat_service(
    vector_store: VectorStore,
    llm_client: LLMClient,
    **kwargs
) -> ChatService:
    """
    Create fully configured chat service

    Args:
        vector_store: Vector store instance
        llm_client: LLM client instance
        **kwargs: Additional arguments
        
    Returns:
        ChatService instance
    """
    from retriever import create_retriever
    from reranker import create_reranker
    from query_rewriter import create_query_rewriter
    from generator import create_generator
    
    retriever = create_retriever(vector_store, llm_client)
    reranker = create_reranker(llm_client)
    query_rewriter = create_query_rewriter(llm_client)
    generator = create_generator(llm_client)
    
    return ChatService(
        vector_store=vector_store,
        llm_client=llm_client,
        retriever=retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        generator=generator,
        **kwargs
    )

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
        chat_service = create_chat_service(vector_store, llm_client)
        
        # Test queries
        test_queries = [
            "What is the refund policy?",
            "How do I reset my password?",
            "退款政策是什么？"
        ]
        
        print("Corrective RAG Examples:\n")
        print("=" * 80)
        
        for query in test_queries:
            print(f"\nQuery: {query}\n")
            
            response = chat_service.process_query(query)
            
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Confidence: {response['confidence']:.2f}")
            print(f"Sources: {', '.join(response['sources'])}")
            print(f"Query rewritten: {response['query_rewritten']}")
            print(f"Duration: {response['duration']:.2f}s")
            print("\nPipeline log:")
            for log in response['pipeline_log']:
                print(f"  - {log}")
            print("=" * 80)
        
        # Get stats
        stats = chat_service.get_stats()
        print(f"\n\nService Statistics:\n{stats}")
        
    except Exception as e:
        print(f"Error: {e}")
