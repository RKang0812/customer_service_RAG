"""
Generator
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
# Generator Class
# ============================================================================

class Generator:
    """
    Generate answers with source citations

    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str = ANSWER_GENERATION_PROMPT,
        prompt_with_history: str = ANSWER_GENERATION_WITH_HISTORY_PROMPT
    ):
        """
        Initialize generator

        Args:
            llm_client: LLM client instance
            prompt_template: Prompt template for generation
            prompt_with_history: Prompt template with conversation history 
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.prompt_with_history = prompt_with_history
        
        logger.info("Generator initialized")
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format documents into context string
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context
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
        
        Args:
            history: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted history
        """
        if not history:
            return "No previous conversation."
        
        history_parts = []
        
        for msg in history[-6:]:  # Keep last 6 messages 
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
        
        Args:
            documents: List of documents
            
        Returns:
            List of source names
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
        
        Args:
            documents: Source documents
            answer: Generated answer
            
        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic based on:
        # 1. Number of documents
        # 2. Answer length 
        # 3. Presence of citations
        
        confidence = 0.5  # Base confidence 
        
        # More documents = higher confidence
        if len(documents) >= 3:
            confidence += 0.2
        elif len(documents) >= 2:
            confidence += 0.1
        
        # Reasonable answer length
        if 50 < len(answer) < 500:
            confidence += 0.1
        
        # Check for citations (brackets) 
        if "[" in answer and "]" in answer:
            confidence += 0.2
        
        # Check for inability to answer
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
        
        Args:
            answer: Generated answer
            
        Returns:
            True if unable to answer
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
        
        Args:
            query: User query 
            documents: List of relevant documents 
            history: Optional conversation history
            
        Returns:
            Dictionary with answer, confidence, sources, etc. 
        """
        start_time = time.time()
        
        try:
            # Format context 
            context = self.format_context(documents)
            
            # Choose prompt template 
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
            
            # Generate answer
            answer = self.llm_client.chat_completion(
                messages,
                temperature=0.7
            )
            
            answer = answer.strip()
            
            # Extract sources
            sources = self.extract_sources(documents)
            
            # Calculate confidence
            confidence = self.calculate_confidence(documents, answer)
            
            # Check if unable to answer
            unable_to_answer = self.detect_unable_to_answer(answer)
            
            # Log generation
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
            
            # Return error response
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
        
        Args:
            query: User query 
            documents: List of documents
            language: Response language
            history: Optional conversation history
            
        Returns:
            Response dictionary 
        """
        if not documents:
            # No documents found - return unable to answer
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
# Convenience Functions 
# ============================================================================

def create_generator(llm_client: LLMClient, **kwargs) -> Generator:
    """
    Create and return generator instance

    Args:
        llm_client: LLM client instance
        **kwargs: Additional arguments
        
    Returns:
        Generator instance
    """
    return Generator(llm_client, **kwargs)

# ============================================================================
# Example Usage 
# ============================================================================

if __name__ == "__main__":
    from llm_client import create_llm_client
    from vector_store import Document
    
    try:
        # Initialize components 
        llm_client = create_llm_client()
        generator = create_generator(llm_client)
        
        # Test documents
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
        
        # Test generation
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
