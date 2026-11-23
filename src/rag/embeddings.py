"""Design decision: local MiniLM through LangChain's HuggingFaceEmbeddings.

For this OAD project, understandable application code and maintainability take
priority over implementing model-runtime optimizations ourselves. The maintained
LangChain adapter delegates model loading, inference, pooling and normalization to libraries.

We keep the compact, pretrained all-MiniLM-L6-v2 model: its 384-dimensional,
normalized vectors fit cosine search over the English catalog. Local execution
keeps bulk indexing independent of a remote embedding endpoint and its quotas.
A pinned model revision and a model identifier prevent incompatible index reuse.
The standard token splitter keeps long product texts within the model window.

Sources: https://docs.langchain.com/oss/python/integrations/embeddings/sentence_transformers
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
"""

from functools import cached_property
from importlib.metadata import version
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from src.utils import config

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def embedding_id(embeddings: Embeddings) -> str:
    if model_id := getattr(embeddings, "model_id", None):
        return model_id
    if model_name := getattr(embeddings, "model_name", None):
        kwargs = getattr(embeddings, "model_kwargs", {})
        encode = getattr(embeddings, "encode_kwargs", {})
        return (f"hf:{model_name}@{kwargs.get('revision', 'default')}:"
                f"{kwargs.get('backend', 'torch')}:normalize={encode.get('normalize_embeddings', False)}:"
                f"precision={encode.get('precision', 'float32')}")
    return (f"{type(embeddings).__module__}.{type(embeddings).__qualname__}:"
            f"{getattr(embeddings, 'model', '')}")


class LocalMiniLMEmbeddings(HuggingFaceEmbeddings):
    """embedding computation is inherited from LangChain."""

    def __init__(self, model_path: Path | None = None, batch_size: int = 64,
                 local_files_only: bool = False):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        kwargs = {"revision": MODEL_REVISION, "trust_remote_code": False,
                  "token": False, "local_files_only": local_files_only}
        if config.EMBEDDING_DEVICE != "auto":
            kwargs["device"] = config.EMBEDDING_DEVICE
        super().__init__(model_name=MODEL_NAME,
                         cache_folder=str(Path(model_path or config.MODEL_PATH) / "huggingface"),
                         model_kwargs=kwargs,
                         encode_kwargs={"normalize_embeddings": True, "precision": "float32",
                                        "batch_size": batch_size})

    @property
    def model_id(self) -> str:
        return (f"hf:{MODEL_NAME}@{MODEL_REVISION}:torch:float32:l2:"
                f"sentence-transformers-{version('sentence-transformers')}")

    @cached_property
    def splitter(self):
        return SentenceTransformersTokenTextSplitter(
            model_name=self.model_name, tokens_per_chunk=220, chunk_overlap=24,
            model_kwargs={**self.model_kwargs, "cache_folder": self.cache_folder})

    @property
    def tokenizer(self):
        return self.splitter.tokenizer

    def chunk_text(self, text: str, size: int = 220, overlap: int = 24) -> list[str]:
        if (size, overlap) != (220, 24):
            raise ValueError("This catalog uses 220 tokens with 24 overlap; change the policy and rebuild together")
        return self.splitter.split_text(text) or [text]
