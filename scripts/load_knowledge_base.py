"""
Load Knowledge Base Documents
加载知识库文档

This script loads all documents from data/knowledge_base/ into Qdrant.
此脚本将data/knowledge_base/中的所有文档加载到Qdrant。

Usage / 使用方法:
    python scripts/load_knowledge_base.py
"""

import sys
from pathlib import Path

# Add parent directory to path / 将父目录添加到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import create_vector_store
from llm_client import create_llm_client
from knowledge_service import create_knowledge_service
from logger_config import setup_logger

logger = setup_logger("load_knowledge_base")

def main():
    """
    Load all documents from knowledge base directory
    从知识库目录加载所有文档
    """
    try:
        logger.info("Starting knowledge base loading...")
        logger.info("=" * 80)
        
        # Initialize components / 初始化组件
        vector_store = create_vector_store()
        llm_client = create_llm_client()
        knowledge_service = create_knowledge_service(vector_store, llm_client)
        
        # Path to knowledge base / 知识库路径
        kb_path = Path(__file__).parent.parent / "data" / "knowledge_base"
        
        if not kb_path.exists():
            logger.error(f"Knowledge base directory not found: {kb_path}")
            print(f"\n❌ Directory not found: {kb_path}")
            print("💡 Create the directory and add your documents:")
            print(f"   mkdir -p {kb_path}")
            return 1
        
        # Check if directory has files / 检查目录是否有文件
        files = list(kb_path.glob("*.*"))
        if not files:
            logger.warning(f"No files found in {kb_path}")
            print(f"\n⚠️  No documents found in {kb_path}")
            print("💡 Add your documents (.txt, .pdf, .docx) to this directory")
            return 0
        
        logger.info(f"Found {len(files)} files in knowledge base")
        print(f"\n📚 Loading {len(files)} documents...")
        
        # Upload directory / 上传目录
        result = knowledge_service.upload_directory(
            str(kb_path),
            file_extensions=[".txt", ".pdf", ".docx"]
        )
        
        if result["success"]:
            logger.info("Knowledge base loaded successfully!")
            logger.info(f"Documents: {result.get('num_documents', 0)}")
            logger.info(f"Chunks: {result.get('num_chunks', 0)}")
            logger.info(f"Duration: {result.get('duration', 0):.2f}s")
            logger.info("=" * 80)
            
            print(f"\n✅ Knowledge base loaded successfully!")
            print(f"📄 Documents: {result.get('num_documents', 0)}")
            print(f"🔢 Chunks: {result.get('num_chunks', 0)}")
            print(f"⏱️  Duration: {result.get('duration', 0):.2f}s")
            
            # Show collection stats / 显示集合统计
            stats = knowledge_service.get_collection_stats()
            print(f"\n📊 Collection Stats:")
            print(f"   Total documents: {stats.get('total_documents', 0)}")
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            
            return 0
        else:
            logger.error(f"Failed to load knowledge base: {result.get('message')}")
            print(f"\n❌ Error: {result.get('message')}")
            return 1
    
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
