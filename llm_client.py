"""
OpenAI LLM Client
OpenAI LLM客户端

This module provides a wrapper for OpenAI API with error handling and retry logic.
本模块提供带错误处理和重试逻辑的OpenAI API封装。
"""

import time
from typing import List, Dict, Optional
from openai import OpenAI, OpenAIError, RateLimitError
import tiktoken

from config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, TEMPERATURE
from logger_config import setup_logger, log_error

logger = setup_logger(__name__)

# ============================================================================
# LLM Client Class / LLM客户端类
# ============================================================================

class LLMClient:
    """
    OpenAI API client with retry logic and error handling
    带重试逻辑和错误处理的OpenAI API客户端
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
        初始化LLM客户端
        
        Args:
            api_key: OpenAI API key / OpenAI API密钥
            llm_model: Model name for chat completion / 用于聊天完成的模型名称
            embedding_model: Model name for embeddings / 用于嵌入的模型名称
            temperature: Sampling temperature / 采样温度
            max_retries: Maximum number of retries / 最大重试次数
        """
        if not api_key:
            raise ValueError("OpenAI API key is required / 需要OpenAI API密钥")
        
        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Token usage tracking / Token使用追踪
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
        生成带重试逻辑的聊天完成
        
        Args:
            messages: List of message dictionaries / 消息字典列表
            temperature: Override default temperature / 覆盖默认温度
            max_tokens: Maximum tokens to generate / 生成的最大token数
            
        Returns:
            Generated text / 生成的文本
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
                
                # Track token usage / 追踪token使用
                if hasattr(response, 'usage'):
                    tokens = response.usage.total_tokens
                    self.total_tokens_used += tokens
                    logger.debug(f"Tokens used: {tokens} (total: {self.total_tokens_used})")
                
                return response.choices[0].message.content
            
            except RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff / 指数退避
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
        
        raise Exception("Max retries exceeded / 超过最大重试次数")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text with retry logic
        生成带重试逻辑的文本嵌入
        
        Args:
            text: Input text / 输入文本
            
        Returns:
            Embedding vector / 嵌入向量
        """
        # Truncate text if too long / 如果文本过长则截断
        max_chars = 8000  # Safe limit for embedding model / 嵌入模型的安全限制
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters for embedding")
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                
                # Track token usage / 追踪token使用
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
        
        raise Exception("Max retries exceeded / 超过最大重试次数")
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken
        使用tiktoken计算文本中的token数
        
        Args:
            text: Input text / 输入文本
            model: Model name (defaults to llm_model) / 模型名称（默认为llm_model）
            
        Returns:
            Number of tokens / token数量
        """
        try:
            model_name = model or self.llm_model
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Could not count tokens, estimating: {e}")
            # Fallback: rough estimate / 后备方案：粗略估计
            return len(text.split()) * 1.3
    
    def get_usage_stats(self) -> Dict:
        """
        Get token usage statistics
        获取token使用统计
        
        Returns:
            Dictionary with usage stats / 包含使用统计的字典
        """
        return {
            "total_tokens": self.total_tokens_used,
            "estimated_cost": self.total_cost
        }
    
    def reset_usage_stats(self):
        """
        Reset token usage tracking
        重置token使用追踪
        """
        self.total_tokens_used = 0
        self.total_cost = 0.0
        logger.info("Usage stats reset")

# ============================================================================
# Convenience Functions / 便利函数
# ============================================================================

def create_llm_client(**kwargs) -> LLMClient:
    """
    Create and return LLM client instance
    创建并返回LLM客户端实例
    
    Args:
        **kwargs: Arguments to pass to LLMClient / 传递给LLMClient的参数
        
    Returns:
        LLM client instance / LLM客户端实例
    """
    return LLMClient(**kwargs)

# ============================================================================
# Example Usage / 使用示例
# ============================================================================

if __name__ == "__main__":
    # Test LLM client / 测试LLM客户端
    try:
        client = LLMClient()
        
        # Test chat completion / 测试聊天完成
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = client.chat_completion(messages)
        print(f"Response: {response}\n")
        
        # Test embedding / 测试嵌入
        text = "This is a test sentence for embedding."
        embedding = client.get_embedding(text)
        print(f"Embedding dimension: {len(embedding)}\n")
        
        # Test token counting / 测试token计数
        token_count = client.count_tokens(text)
        print(f"Token count: {token_count}\n")
        
        # Get usage stats / 获取使用统计
        stats = client.get_usage_stats()
        print(f"Usage stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
