"""
Chat Service
对话服务

This module orchestrates the Corrective RAG pipeline.
本模块编排矫正式RAG流程。
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
# Chat Service Class / 对话服务类
# ============================================================================

class ChatService:
    """
    Orchestrates the Corrective RAG pipeline
    编排矫正式RAG流程
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
        初始化对话服务
        
        Args:
            vector_store: Vector store instance / 向量存储实例
            llm_client: LLM client instance / LLM客户端实例
            retriever: Retriever instance / 检索器实例
            reranker: Reranker instance / 重排序器实例
            query_rewriter: Query rewriter instance / 查询重写器实例
            generator: Generator instance / 生成器实例
            min_relevant_docs: Minimum relevant docs before rewriting / 重写前的最小相关文档数
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
        主要的矫正式RAG工作流
        
        Workflow / 工作流:
        1. Retrieve documents / 检索文档
        2. Grade relevance / 评估相关性
        3. If not enough relevant docs: rewrite query & re-retrieve / 如果相关文档不足：重写查询并重新检索
        4. Generate answer / 生成答案
        
        Args:
            user_query: User's question / 用户的问题
            conversation_history: Previous messages / 之前的消息
            language: Query language (auto-detected if None) / 查询语言（None时自动检测）
            
        Returns:
            Response dictionary with answer, confidence, sources, etc. / 包含答案、置信度、来源等的响应字典
        """
        start_time = time.time()
        pipeline_log = []  # Track pipeline steps / 追踪流程步骤
        
        try:
            logger.info(f"Starting Corrective RAG for query: '{user_query}'")
            
            # Detect language if not provided / 如果未提供则检测语言
            if language is None:
                language = self.query_rewriter.detect_language(user_query)
            
            pipeline_log.append(f"Detected language: {language}")
            
            # ========================================================================
            # Step 1: Initial Retrieval / 步骤1：初始检索
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
            # Step 2: Grade Relevance / 步骤2：评估相关性
            # ========================================================================
            
            logger.info("Step 2: Grading relevance")
            graded_docs = self.reranker.grade_documents(user_query, docs_only)
            relevant_docs = self.reranker.filter_by_threshold(graded_docs)
            
            pipeline_log.append(
                f"Relevance grading: {len(relevant_docs)}/{len(docs_only)} documents relevant"
            )
            
            # ========================================================================
            # Step 3: Query Rewriting (if needed) / 步骤3：查询重写（如需要）
            # ========================================================================
            
            query_rewritten = False
            final_query = user_query
            
            if len(relevant_docs) < self.min_relevant_docs:
                logger.warning(
                    f"Only {len(relevant_docs)} relevant docs found (threshold: {self.min_relevant_docs}). "
                    "Rewriting query..."
                )
                pipeline_log.append("Triggered query rewriting")
                
                # Rewrite query / 重写查询
                rewritten_query = self.query_rewriter.rewrite(user_query, language)
                query_rewritten = True
                final_query = rewritten_query
                
                pipeline_log.append(f"Rewritten query: '{rewritten_query}'")
                
                # Re-retrieve with rewritten query / 使用重写后的查询重新检索
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
                
                # Re-grade with original query / 使用原始查询重新评分
                logger.info("Step 3b: Re-grading with original query")
                graded_docs = self.reranker.grade_documents(user_query, docs_only)
                relevant_docs = self.reranker.filter_by_threshold(graded_docs)
                
                pipeline_log.append(
                    f"Re-grading: {len(relevant_docs)}/{len(docs_only)} documents relevant"
                )
            
            # ========================================================================
            # Step 4: Generate Answer / 步骤4：生成答案
            # ========================================================================
            
            logger.info("Step 4: Generating answer")
            
            # Extract documents only (without scores) / 仅提取文档（不含分数）
            final_docs = [doc for doc, score in relevant_docs]
            
            # Generate answer / 生成答案
            response = self.generator.generate(
                query=user_query,  # Use original query for generation / 使用原始查询生成
                documents=final_docs,
                history=conversation_history
            )
            
            # Add pipeline metadata / 添加流程元数据
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
        处理用户查询（便利方法）
        
        Args:
            query: User query / 用户查询
            history: Conversation history / 对话历史
            
        Returns:
            Response dictionary / 响应字典
        """
        return self.corrective_rag_pipeline(query, history)
    
    def _get_no_docs_message(self, language: str) -> str:
        """
        Get message for when no documents are found
        获取未找到文档时的消息
        
        Args:
            language: Response language / 响应语言
            
        Returns:
            Message text / 消息文本
        """
        messages = {
            "en": "I apologize, but I couldn't find any relevant information in the knowledge base to answer your question. Could you please rephrase your question or ask something else?",
            "zh": "抱歉，我在知识库中找不到相关信息来回答您的问题。您能换个方式重新提问或询问其他内容吗？"
        }
        
        return messages.get(language, messages["en"])
    
    def get_stats(self) -> Dict:
        """
        Get service statistics
        获取服务统计信息
        
        Returns:
            Statistics dictionary / 统计字典
        """
        return {
            "retriever": self.retriever.get_retrieval_stats(),
            "llm_usage": self.llm_client.get_usage_stats(),
            "vector_store": self.vector_store.get_collection_info()
        }

# ============================================================================
# Convenience Functions / 便利函数
# ============================================================================

def create_chat_service(
    vector_store: VectorStore,
    llm_client: LLMClient,
    **kwargs
) -> ChatService:
    """
    Create fully configured chat service
    创建完全配置的对话服务
    
    Args:
        vector_store: Vector store instance / 向量存储实例
        llm_client: LLM client instance / LLM客户端实例
        **kwargs: Additional arguments / 额外参数
        
    Returns:
        ChatService instance / 对话服务实例
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
# Example Usage / 使用示例
# ============================================================================

if __name__ == "__main__":
    from vector_store import create_vector_store
    from llm_client import create_llm_client
    
    try:
        # Initialize components / 初始化组件
        vector_store = create_vector_store()
        llm_client = create_llm_client()
        chat_service = create_chat_service(vector_store, llm_client)
        
        # Test queries / 测试查询
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
        
        # Get stats / 获取统计
        stats = chat_service.get_stats()
        print(f"\n\nService Statistics:\n{stats}")
        
    except Exception as e:
        print(f"Error: {e}")
