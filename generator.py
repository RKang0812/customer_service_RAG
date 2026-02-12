"""
Generator
生成器

This module generates final answers with source citations.
本模块生成带来源引用的最终答案。
"""

from typing import List, Dict, Optional
import time

from vector_store import Document
from llm_client import LLMClient
from config import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_GENERATION_WITH_HISTORY_PROMPT,
    UNABLE_TO_ANSWER_TEMPLATE
)
from logger_config import setup_logger, log_generation, log_error

logger = setup_logger(__name__)

# ============================================================================
# Generator Class / 生成器类
# ============================================================================

class Generator:
    """
    Generate answers with source citations
    生成带来源引用的答案
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str = ANSWER_GENERATION_PROMPT,
        prompt_with_history: str = ANSWER_GENERATION_WITH_HISTORY_PROMPT
    ):
        """
        Initialize generator
        初始化生成器
        
        Args:
            llm_client: LLM client instance / LLM客户端实例
            prompt_template: Prompt template for generation / 用于生成的提示词模板
            prompt_with_history: Prompt template with conversation history / 带对话历史的提示词模板
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.prompt_with_history = prompt_with_history
        
        logger.info("Generator initialized")
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format documents into context string
        将文档格式化为上下文字符串
        
        Args:
            documents: List of documents / 文档列表
            
        Returns:
            Formatted context / 格式化的上下文
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.content.strip()
            
            context_parts.append(
                f"Document {i} [{source}]:\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def format_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history
        格式化对话历史
        
        Args:
            history: List of message dicts with 'role' and 'content' / 包含'role'和'content'的消息字典列表
            
        Returns:
            Formatted history / 格式化的历史
        """
        if not history:
            return "No previous conversation."
        
        history_parts = []
        
        for msg in history[-6:]:  # Keep last 6 messages / 保留最后6条消息
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                history_parts.append(f"User: {content}")
            elif role == "assistant":
                history_parts.append(f"Assistant: {content}")
        
        return "\n".join(history_parts)
    
    def extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract unique source names from documents
        从文档中提取唯一的来源名称
        
        Args:
            documents: List of documents / 文档列表
            
        Returns:
            List of source names / 来源名称列表
        """
        sources = []
        seen = set()
        
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources
    
    def calculate_confidence(
        self,
        documents: List[Document],
        answer: str
    ) -> float:
        """
        Calculate confidence score for the answer
        计算答案的置信度分数
        
        Args:
            documents: Source documents / 来源文档
            answer: Generated answer / 生成的答案
            
        Returns:
            Confidence score (0-1) / 置信度分数（0-1）
        """
        # Simple heuristic based on:
        # 1. Number of documents / 文档数量
        # 2. Answer length / 答案长度
        # 3. Presence of citations / 是否有引用
        
        confidence = 0.5  # Base confidence / 基础置信度
        
        # More documents = higher confidence / 更多文档 = 更高置信度
        if len(documents) >= 3:
            confidence += 0.2
        elif len(documents) >= 2:
            confidence += 0.1
        
        # Reasonable answer length / 合理的答案长度
        if 50 < len(answer) < 500:
            confidence += 0.1
        
        # Check for citations (brackets) / 检查引用（括号）
        if "[" in answer and "]" in answer:
            confidence += 0.2
        
        # Check for inability to answer / 检查是否无法回答
        unable_phrases = [
            "don't have enough information",
            "cannot find",
            "unable to answer",
            "没有足够的信息",
            "无法找到",
            "无法回答"
        ]
        
        if any(phrase in answer.lower() for phrase in unable_phrases):
            confidence = min(confidence, 0.3)
        
        return min(confidence, 1.0)
    
    def detect_unable_to_answer(self, answer: str) -> bool:
        """
        Detect if answer indicates inability to respond
        检测答案是否表示无法回答
        
        Args:
            answer: Generated answer / 生成的答案
            
        Returns:
            True if unable to answer / 如果无法回答返回True
        """
        unable_phrases = [
            "don't have enough information",
            "cannot find",
            "unable to answer",
            "insufficient information",
            "没有足够的信息",
            "无法找到",
            "无法回答",
            "信息不足"
        ]
        
        return any(phrase in answer.lower() for phrase in unable_phrases)
    
    def generate(
        self,
        query: str,
        documents: List[Document],
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Generate answer from documents
        从文档生成答案
        
        Args:
            query: User query / 用户查询
            documents: List of relevant documents / 相关文档列表
            history: Optional conversation history / 可选的对话历史
            
        Returns:
            Dictionary with answer, confidence, sources, etc. / 包含答案、置信度、来源等的字典
        """
        start_time = time.time()
        
        try:
            # Format context / 格式化上下文
            context = self.format_context(documents)
            
            # Choose prompt template / 选择提示词模板
            if history and len(history) > 0:
                history_text = self.format_history(history)
                prompt = self.prompt_with_history.format(
                    history=history_text,
                    context=context,
                    question=query
                )
            else:
                prompt = self.prompt_template.format(
                    context=context,
                    question=query
                )
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful customer service assistant. Always cite sources and be concise."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Generate answer / 生成答案
            answer = self.llm_client.chat_completion(
                messages,
                temperature=0.7
            )
            
            answer = answer.strip()
            
            # Extract sources / 提取来源
            sources = self.extract_sources(documents)
            
            # Calculate confidence / 计算置信度
            confidence = self.calculate_confidence(documents, answer)
            
            # Check if unable to answer / 检查是否无法回答
            unable_to_answer = self.detect_unable_to_answer(answer)
            
            # Log generation / 记录生成
            duration = time.time() - start_time
            log_generation(logger, confidence, len(sources))
            logger.info(f"Generation took {duration:.2f}s")
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "unable_to_answer": unable_to_answer,
                "num_documents": len(documents)
            }
        
        except Exception as e:
            log_error(logger, "generate", e)
            
            # Return error response / 返回错误响应
            return {
                "answer": "I apologize, but I encountered an error while generating the response. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "unable_to_answer": True,
                "num_documents": 0,
                "error": str(e)
            }
    
    def generate_with_fallback(
        self,
        query: str,
        documents: List[Document],
        language: str = "en",
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Generate answer with fallback message if no documents
        生成答案，如果没有文档则使用后备消息
        
        Args:
            query: User query / 用户查询
            documents: List of documents / 文档列表
            language: Response language / 响应语言
            history: Optional conversation history / 可选的对话历史
            
        Returns:
            Response dictionary / 响应字典
        """
        if not documents:
            # No documents found - return unable to answer / 未找到文档 - 返回无法回答
            unable_template = UNABLE_TO_ANSWER_TEMPLATE.get(language, UNABLE_TO_ANSWER_TEMPLATE["en"])
            
            return {
                "answer": unable_template.format(
                    missing_info="relevant information in the knowledge base"
                ),
                "confidence": 0.0,
                "sources": [],
                "unable_to_answer": True,
                "num_documents": 0
            }
        
        return self.generate(query, documents, history)

# ============================================================================
# Convenience Functions / 便利函数
# ============================================================================

def create_generator(llm_client: LLMClient, **kwargs) -> Generator:
    """
    Create and return generator instance
    创建并返回生成器实例
    
    Args:
        llm_client: LLM client instance / LLM客户端实例
        **kwargs: Additional arguments / 额外参数
        
    Returns:
        Generator instance / 生成器实例
    """
    return Generator(llm_client, **kwargs)

# ============================================================================
# Example Usage / 使用示例
# ============================================================================

if __name__ == "__main__":
    from llm_client import create_llm_client
    from vector_store import Document
    
    try:
        # Initialize components / 初始化组件
        llm_client = create_llm_client()
        generator = create_generator(llm_client)
        
        # Test documents / 测试文档
        test_docs = [
            Document(
                content="Our refund policy allows returns within 30 days of purchase with original receipt.",
                metadata={"source": "refund_policy.txt"}
            ),
            Document(
                content="Refunds are processed within 5-7 business days to your original payment method.",
                metadata={"source": "refund_policy.txt"}
            )
        ]
        
        # Test generation / 测试生成
        query = "What is the refund policy?"
        
        print(f"Query: {query}\n")
        print("=" * 80)
        
        response = generator.generate(query, test_docs)
        
        print(f"\nAnswer:\n{response['answer']}\n")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {', '.join(response['sources'])}")
        print(f"Unable to answer: {response['unable_to_answer']}")
        print(f"Number of documents: {response['num_documents']}")
        
    except Exception as e:
        print(f"Error: {e}")
