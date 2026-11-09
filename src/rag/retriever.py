"""
Catalog vector retriever interface for grounded item search.
"""

from langchain_chroma import Chroma
from src.utils import config
from src.rag.indexing import get_embeddings


def get_retriever(k: int = 5):
    """
    Loads existing vector store from disk and returns a LangChain retriever.
    
    TODO: Configure search type and parameters (e.g., similarity, MMR).
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(config.VECTOR_STORE_PATH),
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
