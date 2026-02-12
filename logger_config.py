"""
Logger Configuration

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

# Create logs directory 
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Logger Setup 
# ============================================================================

def setup_logger(name: str = "customer_service_rag") -> logging.Logger:
    """
    Set up logger with colored console output and file output
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with colors
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
    
    # File handler
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
# Logging Utilities
# ============================================================================

def log_retrieval(logger: logging.Logger, query: str, num_results: int, duration: float = None):
    """
    Log retrieval operation
    
    Args:
        logger: Logger instance
        query: User query 
        num_results: Number of results retrieved 
        duration: Time taken in seconds
    """
    msg = f"Retrieved {num_results} documents for query: '{query[:50]}...'"
    if duration:
        msg += f" (took {duration:.2f}s)"
    logger.info(msg)

def log_reranking(logger: logging.Logger, num_relevant: int, num_total: int):
    """
    Log reranking results
    
    Args:
        logger: Logger instance 
        num_relevant: Number of relevant documents 
        num_total: Total number of documents 
    """
    relevance_rate = (num_relevant / num_total * 100) if num_total > 0 else 0
    logger.info(f"Reranking: {num_relevant}/{num_total} docs relevant ({relevance_rate:.1f}%)")

def log_query_rewrite(logger: logging.Logger, original: str, rewritten: str):
    """
    Log query rewriting
    
    Args:
        logger: Logger instance
        original: Original query
        rewritten: Rewritten query
    """
    logger.info(f"Query rewritten:\n  Original: '{original}'\n  Rewritten: '{rewritten}'")

def log_generation(logger: logging.Logger, confidence: float, sources_count: int):
    """
    Log answer generation
    
    Args:
        logger: Logger instance
        confidence: Confidence score 
        sources_count: Number of sources cited
    """
    logger.info(f"Generated answer with confidence {confidence:.2f} from {sources_count} sources")

def log_error(logger: logging.Logger, operation: str, error: Exception):
    """
    Log error with context
    
    Args:
        logger: Logger instance
        operation: Operation that failed
        error: Exception object
    """
    logger.error(f"Error in {operation}: {type(error).__name__}: {str(error)}", exc_info=True)

# Create default logger
default_logger = setup_logger()

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test logging
    logger = setup_logger("test_logger")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test utility functions
    log_retrieval(logger, "What is the refund policy?", 5, 0.23)
    log_reranking(logger, 3, 5)
    log_query_rewrite(logger, "refund?", "What is the refund policy? How do I request a refund?")
    log_generation(logger, 0.85, 3)
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(logger, "test_operation", e)
