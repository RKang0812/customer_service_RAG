"""
Initialize Qdrant Vector Database
初始化Qdrant向量数据库

This script creates the Qdrant collection if it doesn't exist.
此脚本在集合不存在时创建Qdrant集合。

Usage / 使用方法:
    python scripts/init_vector_db.py
"""

import sys
from pathlib import Path

# Add parent directory to path / 将父目录添加到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import create_vector_store
from logger_config import setup_logger

logger = setup_logger("init_vector_db")

def main():
    """
    Initialize vector database
    初始化向量数据库
    """
    try:
        logger.info("Starting Qdrant initialization...")
        logger.info("=" * 80)
        
        # Create vector store (automatically creates collection) / 创建向量存储（自动创建集合）
        vector_store = create_vector_store()
        
        # Get collection info / 获取集合信息
        info = vector_store.get_collection_info()
        
        logger.info("Qdrant initialized successfully!")
        logger.info(f"Collection name: {info.get('name')}")
        logger.info(f"Status: {info.get('status')}")
        logger.info(f"Vectors count: {info.get('vectors_count', 0)}")
        logger.info(f"Points count: {info.get('points_count', 0)}")
        logger.info("=" * 80)
        
        print("\n✅ Qdrant vector database initialized successfully!")
        print(f"📊 Collection: {info.get('name')}")
        print(f"📈 Status: {info.get('status')}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure Qdrant is running:")
        print("   docker-compose up -d")
        return 1

if __name__ == "__main__":
    exit(main())
