"""
Reranker

This module grades document relevance using LLM.
"""

import json
from typing import List, Tuple, Optional
import time

from vector_store import Document
from llm_client import LLMClient
from config import RELEVANCE_GRADING_PROMPT, RELEVANCE_THRESHOLD
from logger_config import setup_logger, log_reranking, log_error

logger = setup_logger(__name__)

# ============================================================================
# Relevance Score Class
# ============================================================================

class RelevanceScore:
    """
    Represents relevance score for a document
    """
    
    def __init__(
        self,
        is_relevant: bool,
        confidence: float,
        reason: str = ""
    ):
        """
        Initialize relevance score
        
        Args:
            is_relevant: Whether document is relevant
            confidence: Confidence score (0-1)
            reason: Explanation for the score 
        """
        self.is_relevant = is_relevant
        self.confidence = confidence
        self.reason = reason
    
    def __repr__(self):
        return f"RelevanceScore(relevant={self.is_relevant}, confidence={self.confidence:.2f})"

# ============================================================================
# Reranker Class
# ============================================================================

class Reranker:
    """
    Grade document relevance using LLM
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        threshold: float = RELEVANCE_THRESHOLD,
        prompt_template: str = RELEVANCE_GRADING_PROMPT
    ):
        """
        Initialize reranker

        Args:
            llm_client: LLM client instance
            threshold: Relevance threshold
            prompt_template: Prompt template for grading
        """
        self.llm_client = llm_client
        self.threshold = threshold
        self.prompt_template = prompt_template
        
        logger.info(f"Reranker initialized with threshold={threshold}")
    
    def grade_document(
        self,
        query: str,
        document: Document
    ) -> RelevanceScore:
        """
        Grade relevance of a single document
        
        Args:
            query: User query
            document: Document to grade
            
        Returns:
            Relevance score 
        """
        try:
            # Format prompt
            prompt = self.prompt_template.format(
                question=query,
                document=document.content
            )
            
            messages = [
                {"role": "system", "content": "You are a document relevance grader. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            # Get LLM response
            response = self.llm_client.chat_completion(messages, temperature=0.1)
            
            # Parse JSON response
            # Try to extract JSON from response 
            response_text = response.strip()
            
            # Handle markdown code blocks 
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to find JSON-like structure
                logger.warning(f"Failed to parse JSON, attempting fallback parsing")
                
                # Simple heuristic: look for true/false 
                is_relevant = "true" in response.lower()
                confidence = 0.5  # Default confidence
                
                return RelevanceScore(
                    is_relevant=is_relevant,
                    confidence=confidence,
                    reason="Parsed from text response"
                )
            
            # Extract values
            is_relevant = result.get("is_relevant", False)
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "")
            
            return RelevanceScore(
                is_relevant=is_relevant,
                confidence=confidence,
                reason=reason
            )
        
        except Exception as e:
            log_error(logger, f"grade_document", e)
            # Return low confidence irrelevant score on error
            return RelevanceScore(
                is_relevant=False,
                confidence=0.0,
                reason=f"Error: {str(e)}"
            )
    
    def grade_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[Document, RelevanceScore]]:
        """
        Grade relevance of multiple documents
        
        Args:
            query: User query 
            documents: List of documents to grade
            
        Returns:
            List of (Document, RelevanceScore) tuples 
        """
        start_time = time.time()
        results = []
        
        for doc in documents:
            score = self.grade_document(query, doc)
            results.append((doc, score))
            
            logger.debug(
                f"Document relevance: {score.is_relevant} "
                f"(confidence: {score.confidence:.2f}) - {doc.content[:50]}..."
            )
        
        # Log overall statistics
        relevant_count = sum(1 for _, score in results if score.is_relevant)
        duration = time.time() - start_time
        log_reranking(logger, relevant_count, len(documents))
        logger.info(f"Reranking took {duration:.2f}s")
        
        return results
    
    def filter_by_threshold(
        self,
        graded_documents: List[Tuple[Document, RelevanceScore]],
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Filter documents by relevance threshold
        
        Args:
            graded_documents: List of (Document, RelevanceScore) tuples 
            threshold: Override default threshold 
            
        Returns:
            List of (Document, confidence) tuples for relevant docs
        """
        thresh = threshold if threshold is not None else self.threshold
        
        filtered = [
            (doc, score.confidence)
            for doc, score in graded_documents
            if score.is_relevant and score.confidence >= thresh
        ]
        
        logger.info(
            f"Filtered {len(filtered)}/{len(graded_documents)} documents "
            f"above threshold {thresh}"
        )
        
        return filtered
    
    def rerank_and_filter(
        self,
        query: str,
        documents: List[Document],
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Grade documents and filter by threshold in one step
        
        Args:
            query: User query
            documents: List of documents
            threshold: Override default threshold
            
        Returns:
            List of (Document, confidence) tuples for relevant docs
        """
        graded = self.grade_documents(query, documents)
        return self.filter_by_threshold(graded, threshold)

# ============================================================================
# Convenience Functions
# ============================================================================

def create_reranker(llm_client: LLMClient, **kwargs) -> Reranker:
    """
    Create and return reranker instance

    Args:
        llm_client: LLM client instance
        **kwargs: Additional arguments
        
    Returns:
        Reranker instance
    """
    return Reranker(llm_client, **kwargs)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from llm_client import create_llm_client
    from vector_store import Document
    
    try:
        # Initialize components
        llm_client = create_llm_client()
        reranker = create_reranker(llm_client, threshold=0.7)
        
        # Test documents
        query = "What is the refund policy?"
        
        test_docs = [
            Document(
                content="Our refund policy allows returns within 30 days of purchase with original receipt.",
                metadata={"source": "policy.txt"}
            ),
            Document(
                content="To reset your password, click on 'Forgot Password' on the login page.",
                metadata={"source": "account.txt"}
            ),
            Document(
                content="Refunds are processed within 5-7 business days to your original payment method.",
                metadata={"source": "policy.txt"}
            )
        ]
        
        # Grade documents
        print(f"\nQuery: {query}\n")
        graded = reranker.grade_documents(query, test_docs)
        
        for doc, score in graded:
            print(f"Relevant: {score.is_relevant}")
            print(f"Confidence: {score.confidence:.2f}")
            print(f"Reason: {score.reason}")
            print(f"Content: {doc.content[:80]}...\n")
        
        # Filter by threshold
        filtered = reranker.filter_by_threshold(graded)
        print(f"\nFiltered {len(filtered)} relevant documents")
        
    except Exception as e:
        print(f"Error: {e}")
