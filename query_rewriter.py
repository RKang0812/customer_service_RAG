"""
Query Rewriter
查询重写器

This module rewrites user queries to improve retrieval quality.
本模块重写用户查询以提升检索质量。
"""

from typing import Optional
import re

from llm_client import LLMClient
from config import QUERY_REWRITING_PROMPT
from logger_config import setup_logger, log_query_rewrite, log_error

logger = setup_logger(__name__)

# ============================================================================
# Query Rewriter Class / 查询重写器类
# ============================================================================

class QueryRewriter:
    """
    Rewrite queries for better document retrieval
    重写查询以提升文档检索效果
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str = QUERY_REWRITING_PROMPT
    ):
        """
        Initialize query rewriter
        初始化查询重写器
        
        Args:
            llm_client: LLM client instance / LLM客户端实例
            prompt_template: Prompt template for rewriting / 用于重写的提示词模板
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        
        logger.info("Query Rewriter initialized")
    
    def detect_language(self, query: str) -> str:
        """
        Detect if query is in English or Chinese
        检测查询是英文还是中文
        
        Args:
            query: User query / 用户查询
            
        Returns:
            Language code: 'en' or 'zh' / 语言代码：'en'或'zh'
        """
        # Simple heuristic: check for Chinese characters / 简单启发式：检查中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', query)
        
        if len(chinese_chars) > len(query) * 0.3:  # If >30% Chinese characters / 如果>30%中文字符
            return 'zh'
        return 'en'
    
    def rewrite(
        self,
        query: str,
        language: Optional[str] = None
    ) -> str:
        """
        Rewrite query for better retrieval
        重写查询以提升检索效果
        
        Args:
            query: Original user query / 原始用户查询
            language: Query language ('en' or 'zh'), auto-detected if None / 查询语言，None时自动检测
            
        Returns:
            Rewritten query / 重写后的查询
        """
        try:
            # Detect language if not provided / 如果未提供则检测语言
            if language is None:
                language = self.detect_language(query)
            
            # Format prompt / 格式化提示词
            prompt = self.prompt_template.format(query=query)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a query rewriting assistant. Expand and clarify queries while preserving the original language."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Get rewritten query / 获取重写后的查询
            rewritten = self.llm_client.chat_completion(
                messages,
                temperature=0.3  # Lower temperature for more consistent rewrites / 较低温度以获得更一致的重写
            )
            
            # Clean up response / 清理响应
            rewritten = rewritten.strip()
            
            # Remove any quotes or extra formatting / 移除任何引号或额外格式
            rewritten = rewritten.strip('"\'')
            
            # If rewrite failed or is too short, return original / 如果重写失败或太短，返回原始查询
            if not rewritten or len(rewritten) < 3:
                logger.warning("Rewrite produced empty or too short result, using original query")
                return query
            
            # Log the rewrite / 记录重写
            log_query_rewrite(logger, query, rewritten)
            
            return rewritten
        
        except Exception as e:
            log_error(logger, "rewrite", e)
            # Return original query on error / 出错时返回原始查询
            logger.warning("Query rewrite failed, using original query")
            return query
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with related terms (alternative approach)
        使用相关术语扩展查询（替代方法）
        
        Args:
            query: Original query / 原始查询
            
        Returns:
            Expanded query / 扩展后的查询
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
        简化复杂查询（用于调试）
        
        Args:
            query: Original query / 原始查询
            
        Returns:
            Simplified query / 简化后的查询
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
# Convenience Functions / 便利函数
# ============================================================================

def create_query_rewriter(llm_client: LLMClient, **kwargs) -> QueryRewriter:
    """
    Create and return query rewriter instance
    创建并返回查询重写器实例
    
    Args:
        llm_client: LLM client instance / LLM客户端实例
        **kwargs: Additional arguments / 额外参数
        
    Returns:
        QueryRewriter instance / 查询重写器实例
    """
    return QueryRewriter(llm_client, **kwargs)

# ============================================================================
# Example Usage / 使用示例
# ============================================================================

if __name__ == "__main__":
    from llm_client import create_llm_client
    
    try:
        # Initialize components / 初始化组件
        llm_client = create_llm_client()
        rewriter = create_query_rewriter(llm_client)
        
        # Test queries / 测试查询
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
        
        # Test expansion / 测试扩展
        print("\n\nQuery Expansion Example:\n")
        print("=" * 80)
        
        test_query = "refund policy"
        expanded = rewriter.expand_query(test_query)
        print(f"Original: {test_query}")
        print(f"Expanded: {expanded}")
        
    except Exception as e:
        print(f"Error: {e}")
