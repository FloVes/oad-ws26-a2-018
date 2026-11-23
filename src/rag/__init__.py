"""
RAG and catalog vector store indexing and retrieval package.
"""

__all__ = ["get_embeddings", "index_dataset", "get_retriever", "CatalogRetriever", "Product"]


def __getattr__(name):
    # Preserve package-level imports without executing CLI modules prematurely.
    if name in {"get_embeddings", "index_dataset"}:
        from src.rag import indexing
        return getattr(indexing, name)
    if name in {"get_retriever", "CatalogRetriever"}:
        from src.rag import retriever
        return getattr(retriever, name)
    if name == "Product":
        from src.rag.catalog import Product
        return Product
    raise AttributeError(name)
