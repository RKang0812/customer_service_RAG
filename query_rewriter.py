"""
Query Rewriter

"""

from typing import Optional
import re

from llm_client import LLMClient
from config import QUERY_REWRITING_PROMPT
from logger_config import setup_logger, log_query_rewrite, log_error

logger = setup_logger(__name__)

# ============================================================================
# Query Rewriter Class
# ============================================================================

class QueryRewriter:
    """
    Rewrite queries for better document retrieval
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str = QUERY_REWRITING_PROMPT
    ):
        """
        Initialize query rewriter
        
        Args:
            llm_client: LLM client instance
            prompt_template: Prompt template for rewriting 
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        
        logger.info("Query Rewriter initialized")
    
    def detect_language(self, query: str) -> str:
        """
        Detect if query is in English or Chinese
        
        Args:
            query: User query 
            
        Returns:
            Language code: 'en' or 'zh'
        """
        # Simple heuristic: check for Chinese characters 
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', query)
        
        if len(chinese_chars) > len(query) * 0.3:  # If >30% Chinese characters
            return 'zh'
        return 'en'
    
    def rewrite(
        self,
        query: str,
        language: Optional[str] = None
    ) -> str:
        """
        Rewrite query for better retrieval
        
        Args:
            query: Original user query
            language: Query language ('en' or 'zh'), auto-detected if None 
            
        Returns:
            Rewritten query
        """
        try:
            # Detect language if not provided
            if language is None:
                language = self.detect_language(query)
            
            # Format prompt
            prompt = self.prompt_template.format(query=query)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a query rewriting assistant. Expand and clarify queries while preserving the original language."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Get rewritten query
            rewritten = self.llm_client.chat_completion(
                messages,
                temperature=0.3  # Lower temperature for more consistent rewrites
            )
            
            # Clean up response
            rewritten = rewritten.strip()
            
            # Remove any quotes or extra formatting 
            rewritten = rewritten.strip('"\'')
            
            # If rewrite failed or is too short, return original 
            if not rewritten or len(rewritten) < 3:
                logger.warning("Rewrite produced empty or too short result, using original query")
                return query
            
            # Log the rewrite
            log_query_rewrite(logger, query, rewritten)
            
            return rewritten
        
        except Exception as e:
            log_error(logger, "rewrite", e)
            # Return original query on error 
            logger.warning("Query rewrite failed, using original query")
            return query
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with related terms (alternative approach)
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a query expansion assistant. Add related terms and synonyms to improve search."
                },
                {
                    "role": "user",
                    "content": f"Expand this query with related terms: {query}\n\nExpanded query:"
                }
            ]
            
            expanded = self.llm_client.chat_completion(messages, temperature=0.5)
            expanded = expanded.strip().strip('"\'')
            
            if expanded and len(expanded) > len(query):
                logger.info(f"Expanded query: '{query}' → '{expanded}'")
                return expanded
            
            return query
        
        except Exception as e:
            log_error(logger, "expand_query", e)
            return query
    
    def simplify_query(self, query: str) -> str:
        """
        Simplify complex query (useful for debugging)
        
        Args:
            query: Original query
            
        Returns:
            Simplified query
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a query simplification assistant. Extract the core question."
                },
                {
                    "role": "user",
                    "content": f"Simplify this query to its core question: {query}\n\nSimplified:"
                }
            ]
            
            simplified = self.llm_client.chat_completion(messages, temperature=0.2)
            simplified = simplified.strip().strip('"\'')
            
            if simplified:
                logger.info(f"Simplified query: '{query}' → '{simplified}'")
                return simplified
            
            return query
        
        except Exception as e:
            log_error(logger, "simplify_query", e)
            return query

# ============================================================================
# Convenience Functions
# ============================================================================

def create_query_rewriter(llm_client: LLMClient, **kwargs) -> QueryRewriter:
    """
    Create and return query rewriter instance
    
    Args:
        llm_client: LLM client instance
        **kwargs: Additional arguments
        
    Returns:
        QueryRewriter instance
    """
    return QueryRewriter(llm_client, **kwargs)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from llm_client import create_llm_client
    
    try:
        # Initialize components
        llm_client = create_llm_client()
        rewriter = create_query_rewriter(llm_client)
        
        # Test queries
        test_queries = [
            "refund?",
            "not working",
            "账号问题",
            "How do I cancel my subscription?",
            "产品保修多久？"
        ]
        
        print("Query Rewriting Examples:\n")
        print("=" * 80)
        
        for query in test_queries:
            lang = rewriter.detect_language(query)
            rewritten = rewriter.rewrite(query)
            
            print(f"\nOriginal ({lang}): {query}")
            print(f"Rewritten:      {rewritten}")
            print("-" * 80)
        
        # Test expansion 
        print("\n\nQuery Expansion Example:\n")
        print("=" * 80)
        
        test_query = "refund policy"
        expanded = rewriter.expand_query(test_query)
        print(f"Original: {test_query}")
        print(f"Expanded: {expanded}")
        
    except Exception as e:
        print(f"Error: {e}")
