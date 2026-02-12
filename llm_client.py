"""
OpenAI LLM Client
"""

import time
from typing import List, Dict, Optional
from openai import OpenAI, OpenAIError, RateLimitError
import tiktoken

from config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, TEMPERATURE
from logger_config import setup_logger, log_error

logger = setup_logger(__name__)

# ============================================================================
# LLM Client Class 
# ============================================================================

class LLMClient:
    """
    OpenAI API client with retry logic and error handling
    """
    
    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        llm_model: str = LLM_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        temperature: float = TEMPERATURE,
        max_retries: int = 3
    ):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenAI API key
            llm_model: Model name for chat completion
            embedding_model: Model name for embeddings 
            temperature: Sampling temperature 
            max_retries: Maximum number of retries
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Token usage tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        logger.info(f"LLM Client initialized with model: {llm_model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate chat completion with retry logic
        
        Args:
            messages: List of message dictionaries 
            temperature: Override default temperature 
            max_tokens: Maximum tokens to generate 
            
        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens
                )
                
                # Track token usage 
                if hasattr(response, 'usage'):
                    tokens = response.usage.total_tokens
                    self.total_tokens_used += tokens
                    logger.debug(f"Tokens used: {tokens} (total: {self.total_tokens_used})")
                
                return response.choices[0].message.content
            
            except RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                time.sleep(wait_time)
                
                if attempt == self.max_retries - 1:
                    log_error(logger, "chat_completion", e)
                    raise
            
            except OpenAIError as e:
                log_error(logger, "chat_completion", e)
                
                if attempt == self.max_retries - 1:
                    raise
                
                time.sleep(1)
        
        raise Exception("Max retries exceeded")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text with retry logic
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Truncate text if too long
        max_chars = 8000  # Safe limit for embedding model 
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters for embedding")
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                
                # Track token usage
                if hasattr(response, 'usage'):
                    tokens = response.usage.total_tokens
                    self.total_tokens_used += tokens
                
                return response.data[0].embedding
            
            except RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                time.sleep(wait_time)
                
                if attempt == self.max_retries - 1:
                    log_error(logger, "get_embedding", e)
                    raise
            
            except OpenAIError as e:
                log_error(logger, "get_embedding", e)
                
                if attempt == self.max_retries - 1:
                    raise
                
                time.sleep(1)
        
        raise Exception("Max retries exceeded")
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken
        
        Args:
            text: Input text 
            model: Model name (defaults to llm_model) 
            
        Returns:
            Number of tokens
        """
        try:
            model_name = model or self.llm_model
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Could not count tokens, estimating: {e}")
            # Fallback: rough estimate
            return len(text.split()) * 1.3
    
    def get_usage_stats(self) -> Dict:
        """
        Get token usage statistics
        
        Returns:
            Dictionary with usage stats
        """
        return {
            "total_tokens": self.total_tokens_used,
            "estimated_cost": self.total_cost
        }
    
    def reset_usage_stats(self):
        """
        Reset token usage tracking
        """
        self.total_tokens_used = 0
        self.total_cost = 0.0
        logger.info("Usage stats reset")

# ============================================================================
# Convenience Functions
# ============================================================================

def create_llm_client(**kwargs) -> LLMClient:
    """
    Create and return LLM client instance
    
    Args:
        **kwargs: Arguments to pass to LLMClient
        
    Returns:
        LLM client instance
    """
    return LLMClient(**kwargs)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test LLM client
    try:
        client = LLMClient()
        
        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = client.chat_completion(messages)
        print(f"Response: {response}\n")
        
        # Test embedding
        text = "This is a test sentence for embedding."
        embedding = client.get_embedding(text)
        print(f"Embedding dimension: {len(embedding)}\n")
        
        # Test token counting
        token_count = client.count_tokens(text)
        print(f"Token count: {token_count}\n")
        
        # Get usage stats
        stats = client.get_usage_stats()
        print(f"Usage stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
