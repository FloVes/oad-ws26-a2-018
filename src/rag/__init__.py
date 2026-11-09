"""
RAG and catalog vector store indexing and retrieval package.
"""

from src.rag.indexing import get_embeddings, index_dataset
from src.rag.retriever import get_retriever

__all__ = ["get_embeddings", "index_dataset", "get_retriever"]
