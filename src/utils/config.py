"""
Application configuration and environment settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Project root and directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw"
VECTOR_STORE_PATH = DATA_DIR / "vectorstore"
DEFAULT_RAW_FILE = RAW_DATA_PATH / "meta_Video_Games.jsonl.gz"
MODEL_PATH = Path(os.getenv("EMBEDDING_MODEL_PATH", str(DATA_DIR / "models" / "all-MiniLM-L6-v2")))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

# LLMDuckt / OpenAI API Configuration
LLMDUCKT_API_KEY = os.getenv("LLMDUCKT_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLMDUCKT_BASE_URL = os.getenv("LLMDUCKT_API_BASE_URL", os.getenv("LLM_BASE_URL", "https://llm.tugraz.at/llmapi/v1"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", os.getenv("LLM_API_MODEL", "azure/gpt-5-mini"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "azure/text-embedding-ada-002")
