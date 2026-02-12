"""
Logger Configuration
日志配置

This module sets up structured logging for the application.
本模块为应用设置结构化日志。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from colorlog import ColoredFormatter
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

from config import LOG_LEVEL

# Create logs directory / 创建日志目录
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Logger Setup / 日志设置
# ============================================================================

def setup_logger(name: str = "customer_service_rag") -> logging.Logger:
    """
    Set up logger with colored console output and file output
    设置带彩色控制台输出和文件输出的日志器
    
    Args:
        name: Logger name / 日志器名称
        
    Returns:
        Configured logger / 配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid duplicate handlers / 避免重复的处理器
    if logger.handlers:
        return logger
    
    # Console handler with colors / 带颜色的控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if COLORLOG_AVAILABLE:
        console_formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(
            "%(levelname)-8s %(name)s - %(message)s"
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler / 文件处理器
    log_filename = LOGS_DIR / f"rag_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# Logging Utilities / 日志工具函数
# ============================================================================

def log_retrieval(logger: logging.Logger, query: str, num_results: int, duration: float = None):
    """
    Log retrieval operation
    记录检索操作
    
    Args:
        logger: Logger instance / 日志器实例
        query: User query / 用户查询
        num_results: Number of results retrieved / 检索到的结果数量
        duration: Time taken in seconds / 耗时（秒）
    """
    msg = f"Retrieved {num_results} documents for query: '{query[:50]}...'"
    if duration:
        msg += f" (took {duration:.2f}s)"
    logger.info(msg)

def log_reranking(logger: logging.Logger, num_relevant: int, num_total: int):
    """
    Log reranking results
    记录重排序结果
    
    Args:
        logger: Logger instance / 日志器实例
        num_relevant: Number of relevant documents / 相关文档数量
        num_total: Total number of documents / 文档总数
    """
    relevance_rate = (num_relevant / num_total * 100) if num_total > 0 else 0
    logger.info(f"Reranking: {num_relevant}/{num_total} docs relevant ({relevance_rate:.1f}%)")

def log_query_rewrite(logger: logging.Logger, original: str, rewritten: str):
    """
    Log query rewriting
    记录查询重写
    
    Args:
        logger: Logger instance / 日志器实例
        original: Original query / 原始查询
        rewritten: Rewritten query / 重写后的查询
    """
    logger.info(f"Query rewritten:\n  Original: '{original}'\n  Rewritten: '{rewritten}'")

def log_generation(logger: logging.Logger, confidence: float, sources_count: int):
    """
    Log answer generation
    记录答案生成
    
    Args:
        logger: Logger instance / 日志器实例
        confidence: Confidence score / 置信度分数
        sources_count: Number of sources cited / 引用的来源数量
    """
    logger.info(f"Generated answer with confidence {confidence:.2f} from {sources_count} sources")

def log_error(logger: logging.Logger, operation: str, error: Exception):
    """
    Log error with context
    记录带上下文的错误
    
    Args:
        logger: Logger instance / 日志器实例
        operation: Operation that failed / 失败的操作
        error: Exception object / 异常对象
    """
    logger.error(f"Error in {operation}: {type(error).__name__}: {str(error)}", exc_info=True)

# Create default logger / 创建默认日志器
default_logger = setup_logger()

# ============================================================================
# Example Usage / 使用示例
# ============================================================================

if __name__ == "__main__":
    # Test logging / 测试日志
    logger = setup_logger("test_logger")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test utility functions / 测试工具函数
    log_retrieval(logger, "What is the refund policy?", 5, 0.23)
    log_reranking(logger, 3, 5)
    log_query_rewrite(logger, "refund?", "What is the refund policy? How do I request a refund?")
    log_generation(logger, 0.85, 3)
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(logger, "test_operation", e)
