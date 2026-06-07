"""
BanglaLex — Knowledge Base Package
====================================
Exposes the four core components of the RAG pipeline:
  chunker   → load & chunk the Kaggle legal-acts CSV dataset
  embedder  → multilingual sentence-transformer wrapper
  indexer   → FAISS index build / save / load
  retriever → high-level query interface
"""

from .chunker   import load_kaggle_dataset, chunk_dataframe, save_chunks, load_chunks
from .embedder  import Embedder
from .indexer   import KnowledgeBaseIndex
from .retriever import Retriever

__all__ = [
    "load_kaggle_dataset",
    "chunk_dataframe",
    "save_chunks",
    "load_chunks",
    "Embedder",
    "KnowledgeBaseIndex",
    "Retriever",
]
