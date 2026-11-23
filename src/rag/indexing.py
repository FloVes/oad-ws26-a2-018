"""Dataset ingestion, Doc conversion, Chroma vector database indexing."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Optional

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.rag.catalog import Product, normalize_product
from src.rag.embeddings import LocalMiniLMEmbeddings, embedding_id
from src.utils import config

COLLECTION_NAME = "video_games"


def get_embeddings() -> Embeddings:
    """Return local embeddings by default; preserve the explicit API option."""
    if config.EMBEDDING_PROVIDER == "local":
        return LocalMiniLMEmbeddings()
    if config.EMBEDDING_PROVIDER != "api":
        raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'api'")
    kwargs = {"model": config.EMBEDDING_MODEL_NAME}
    if config.LLMDUCKT_API_KEY:
        kwargs["api_key"] = config.LLMDUCKT_API_KEY
    if config.LLMDUCKT_BASE_URL:
        kwargs["base_url"] = config.LLMDUCKT_BASE_URL
    return OpenAIEmbeddings(**kwargs)


def _write_manifest(directory: Path, manifest: dict) -> None:
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def product_documents(product: Product, embeddings: Embeddings) -> list[Document]:
    """text + source metadata -> LangChain Documents."""
    text = product.search_text()
    if hasattr(embeddings, "chunk_text"):
        chunks = embeddings.chunk_text(text, size=220, overlap=24)
    else:
        # Test/API adapters use a bounded word split; local indexing uses exact tokens.
        words = text.split()
        chunks = []
        for start in range(0, len(words), 196):
            chunks.append(" ".join(words[start:start + 220]))
            if start + 220 >= len(words):
                break
    return [Document(page_content=chunk, metadata={**product.index_metadata(),
                     "chunk_index": i, "chunk_count": len(chunks)}) for i, chunk in enumerate(chunks)]


def index_dataset(raw_file_path: Optional[str] = None, *, persist_directory=None,
                  embeddings: Embeddings | None = None, limit: int | None = None,
                  batch_size: int = 256, min_products: int = 100_000) -> Chroma:
    """Clean, chunk, embed and persist real catalog items.

    Existing nonempty indexes are never overwritten. Build replacements
    in a new directory. An explicit limit produces a labelled development sample.
    """
    raw_path = Path(raw_file_path or config.DEFAULT_RAW_FILE).resolve()
    directory = Path(persist_directory or config.VECTOR_STORE_PATH).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if type(min_products) is not int or min_products < 1:
        raise ValueError("min_products must be a positive integer")
    if directory.exists() and any(p.name != ".gitkeep" for p in directory.iterdir()):
        raise FileExistsError("Index directory is not empty; use a new --persist-directory")
    directory.mkdir(parents=True, exist_ok=True)
    embeddings = embeddings if embeddings is not None else get_embeddings()
    started = time.monotonic()
    with raw_path.open("rb") as raw_file:
        source_digest = hashlib.file_digest(raw_file, "sha256").hexdigest()
    manifest = {"schema_version": 1, "status": "preparing", "collection_name": COLLECTION_NAME,
                "embedding_id": embedding_id(embeddings), "sample_limit": limit,
                "minimum_products": 1 if limit is not None and min_products == 100_000 else min_products,
                "chunking": {"size": 220, "overlap": 24,
                             "unit": "wordpiece" if hasattr(embeddings, "chunk_text") else "words"},
                "source_file": raw_path.name,
                "source_sha256": source_digest,
                "started_at": datetime.now(timezone.utc).isoformat()}
    stats = Counter()
    _write_manifest(directory, manifest)
    database = sqlite3.connect(directory / "catalog.sqlite3")
    try:
        database.execute("CREATE TABLE products(item_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        opener = gzip.open if raw_path.suffix == ".gz" else open
        with opener(raw_path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                stats["source_rows"] += 1
                if not line.strip():
                    stats["blank_rows"] += 1
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}") from exc
                product = normalize_product(record) if isinstance(record, dict) else None
                if product is None:
                    stats["invalid_identity_or_title"] += 1
                    continue
                cursor = database.execute("INSERT OR IGNORE INTO products VALUES (?, ?)",
                                          (product.item_id, product.model_dump_json()))
                if cursor.rowcount == 0:
                    stats["duplicate_products"] += 1
                    continue
                stats["products"] += 1
                for field in ("price", "average_rating", "description", "features", "categories"):
                    value = getattr(product, field)
                    if value is None or value == "" or value == []:
                        stats[f"missing_{field}"] += 1
                if limit is not None and stats["products"] >= limit:
                    break
        database.commit()
        if stats["products"] < manifest["minimum_products"]:
            raise ValueError(f"Only {stats['products']} unique usable products; require {manifest['minimum_products']}")
        manifest.update(status="indexing", product_count=stats["products"], cleaning=dict(stats), indexed_chunks=0)
        _write_manifest(directory, manifest)
        vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(directory),
                             embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"},
                             client_settings=Settings(anonymized_telemetry=False))
        pending = []
        def flush():
            if not pending:
                return
            ids = [f"{doc.metadata['item_id']}:{doc.metadata['chunk_index']}" for doc in pending]
            vectorstore.add_documents(pending, ids=ids)
            manifest["indexed_chunks"] += len(pending)
            _write_manifest(directory, manifest)
            pending.clear()
        for i, (payload,) in enumerate(database.execute("SELECT payload FROM products ORDER BY rowid"), 1):
            product = Product.model_validate_json(payload)
            for doc in product_documents(product, embeddings):
                pending.append(doc)
                if len(pending) >= batch_size:
                    flush()
            if i % 1000 == 0:
                print(f"Indexed {i}/{stats['products']} products; {manifest['indexed_chunks']} chunks", flush=True)
        flush()
        stored_chunks = vectorstore._collection.count()
        if stored_chunks != manifest["indexed_chunks"]:
            raise RuntimeError("Persisted vector count differs from indexed chunk count")
        first_vector = vectorstore.get(limit=1, include=["embeddings"])["embeddings"][0]
        manifest.update(status="complete", chunk_count=manifest["indexed_chunks"],
                        embedding_dimensions=len(first_vector), distance="cosine",
                        elapsed_seconds=round(time.monotonic() - started, 2),
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        assignment_size_met=stats["products"] >= 100_000)
        _write_manifest(directory, manifest)
        return vectorstore
    except BaseException as exc:
        manifest.update(status="failed", cleaning=dict(stats), error=type(exc).__name__)
        _write_manifest(directory, manifest)
        raise
    finally:
        database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file")
    parser.add_argument("--persist-directory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-products", type=int, default=100_000)
    args = parser.parse_args()
    print("Running dataset indexing...", flush=True)
    index_dataset(args.raw_file, persist_directory=args.persist_directory, limit=args.limit,
                  batch_size=args.batch_size, min_products=args.min_products)
    print("Indexing completed.", flush=True)
