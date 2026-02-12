"""
Document Loader
"""

from typing import List, Optional
from pathlib import Path
import re

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False

from config import CHUNK_SIZE, CHUNK_OVERLAP
from vector_store import Document
from logger_config import setup_logger, log_error

logger = setup_logger(__name__)

# ============================================================================
# Text Loading Functions
# ============================================================================

def load_txt(file_path: str, encoding: str = "utf-8") -> str:
    """
    Load text from .txt file

    
    Args:
        file_path: Path to text file
        encoding: File encoding
        
    Returns:
        File content as string
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        logger.info(f"Loaded {len(content)} characters from {file_path}")
        return content
    
    except UnicodeDecodeError:
        # Try different encodings
        for enc in ["gbk", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                logger.warning(f"File loaded with {enc} encoding")
                return content
            except:
                continue
        
        logger.error(f"Could not decode file: {file_path}")
        raise
    
    except Exception as e:
        log_error(logger, f"load_txt({file_path})", e)
        raise

def load_pdf(file_path: str) -> str:
    """
    Load text from PDF file
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf is required to load PDF files. Install with: pip install pypdf")
    
    try:
        reader = PdfReader(file_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        content = "\n\n".join(text_parts)
        logger.info(f"Loaded {len(content)} characters from {len(reader.pages)} pages in {file_path}")
        
        return content
    
    except Exception as e:
        log_error(logger, f"load_pdf({file_path})", e)
        raise

def load_docx(file_path: str) -> str:
    """
    Load text from DOCX file
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        Extracted text
    """
    if not PYTHON_DOCX_AVAILABLE:
        raise ImportError("python-docx is required to load DOCX files. Install with: pip install python-docx")
    
    try:
        doc = DocxDocument(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        content = "\n\n".join(paragraphs)
        
        logger.info(f"Loaded {len(content)} characters from {len(paragraphs)} paragraphs in {file_path}")
        
        return content
    
    except Exception as e:
        log_error(logger, f"load_docx({file_path})", e)
        raise

def load_document(file_path: str) -> str:
    """
    Load document based on file extension
    
    Args:
        file_path: Path to document file
        
    Returns:
        Document content
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .txt, .pdf, .docx")

# ============================================================================
# Text Chunking Functions
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (optional)
    # text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'`-]', '', text)
    
    return text.strip()

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Split text into overlapping chunks

    
    Args:
        text: Input text 
        chunk_size: Maximum chunk size in characters 
        chunk_overlap: Overlap between chunks
        separators: List of separators to split on
        
    Returns:
        List of text chunks 
    """
    if separators is None:
        # Default separators: paragraph, sentence, word 
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    
    # Clean text first 
    text = clean_text(text)
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    def split_with_separator(text: str, sep: str) -> List[str]:
        """Split text by separator while keeping separator"""
        if sep not in text:
            return [text]
        parts = text.split(sep)
        # Add separator back except for the last part
        return [part + sep for part in parts[:-1]] + [parts[-1]]
    
    def merge_chunks(parts: List[str]) -> List[str]:
        """Merge parts into chunks respecting size and overlap"""
        result = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) <= chunk_size:
                current_chunk += part
            else:
                if current_chunk:
                    result.append(current_chunk)
                    # Start new chunk with overlap
                    if chunk_overlap > 0:
                        # Take last chunk_overlap characters 
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + part
                    else:
                        current_chunk = part
                else:
                    # Part is larger than chunk_size, force split
                    result.append(part[:chunk_size])
                    current_chunk = part[chunk_size:]
        
        if current_chunk:
            result.append(current_chunk)
        
        return result
    
    # Try splitting with each separator 
    parts = [text]
    for separator in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(split_with_separator(part, separator))
        parts = new_parts
        
        # Check if we can create good chunks 
        merged = merge_chunks(parts)
        if all(len(chunk) <= chunk_size for chunk in merged):
            chunks = merged
            break
    
    # If we still don't have good chunks, force split
    if not chunks:
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def load_and_chunk_document(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    metadata: Optional[dict] = None
) -> List[Document]:
    """
    Load document and split into chunks
    
    Args:
        file_path: Path to document
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata for all chunks 
        
    Returns:
        List of Document objects
    """
    try:
        # Load document
        content = load_document(file_path)
        
        # Split into chunks
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        
        # Create Document objects
        path = Path(file_path)
        base_metadata = metadata or {}
        base_metadata.update({
            "source": path.name,
            "file_path": str(path),
            "total_chunks": len(chunks)
        })
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            
            doc = Document(
                content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} document chunks from {file_path}")
        return documents
    
    except Exception as e:
        log_error(logger, f"load_and_chunk_document({file_path})", e)
        raise

# ============================================================================
# Batch Processing
# ============================================================================

def load_directory(
    directory_path: str,
    file_extensions: Optional[List[str]] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Load all documents from a directory
    
    Args:
        directory_path: Path to directory
        file_extensions: List of extensions to include
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all Document objects
    """
    if file_extensions is None:
        file_extensions = [".txt", ".pdf", ".docx"]
    
    path = Path(directory_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    all_documents = []
    
    for ext in file_extensions:
        files = list(path.glob(f"*{ext}"))
        logger.info(f"Found {len(files)} {ext} files")
        
        for file_path in files:
            try:
                docs = load_and_chunk_document(
                    str(file_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
    
    logger.info(f"Loaded total of {len(all_documents)} document chunks from {directory_path}")
    return all_documents

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test document loading
    test_text = """This is a test document. It contains multiple sentences.
    
    This is a new paragraph. We want to test the chunking functionality.
    
    And here's another paragraph with some more content to ensure we have enough text for chunking."""
    
    # Test chunking
    chunks = chunk_text(test_text, chunk_size=50, chunk_overlap=10)
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}\n")
