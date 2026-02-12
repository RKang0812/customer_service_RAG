"""
Initialize Qdrant Vector Database

This script creates the Qdrant collection if it doesn't exist.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import create_vector_store
from logger_config import setup_logger

logger = setup_logger("init_vector_db")

def main():
    """
    Initialize vector database
    """
    try:
        logger.info("Starting Qdrant initialization...")
        logger.info("=" * 80)

        vector_store = create_vector_store()

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
