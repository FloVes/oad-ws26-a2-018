"""
Dataset ingestion, Document conversion, and Chroma vector database indexing.
"""

from typing import Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.utils import config


def get_embeddings() -> OpenAIEmbeddings:
    """
    Initializes and returns the OpenAI / LLMDuckt embedding model.
    """
    kwargs = {"model": config.EMBEDDING_MODEL_NAME}
    if config.LLMDUCKT_API_KEY:
        kwargs["api_key"] = config.LLMDUCKT_API_KEY
    if config.LLMDUCKT_BASE_URL:
        kwargs["base_url"] = config.LLMDUCKT_BASE_URL
    return OpenAIEmbeddings(**kwargs)


def index_dataset(raw_file_path: Optional[str] = None) -> Chroma:
    """
    Loads raw dataset, converts items into Documents, creates embeddings,
    and persists them into the vector database.
    
    TODO: Implement dataset parsing, text chunking/formatting, and vector indexing.
    """
    # 1. Load domain data (e.g., MovieLens, Recipes, or Amazon Reviews) and optional additional data.
    # 2. Convert records into LangChain Document objects with page_content and metadata
    # 3. Create and persist Chroma vector store
    raise NotImplementedError("TODO: Implement index_dataset")


if __name__ == "__main__":
    print("Running dataset indexing...")
    index_dataset()
    print("Indexing completed.")
