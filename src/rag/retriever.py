"""Semantic product search, hard metadata filters and canonical ID lookup."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableConfig

from src.rag.catalog import Product, category_key
from src.rag.embeddings import embedding_id
from src.rag.indexing import get_embeddings
from src.utils import config


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 50:
        raise ValueError("k must be an integer between 1 and 50")
    return k


def _where(filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Translate the public filter interface into Chroma predicates."""
    if filters is None:
        return None
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be a mapping")
    unknown = set(filters) - {"min_price", "max_price", "min_rating", "category"}
    if unknown:
        raise ValueError(f"Unknown filters: {', '.join(sorted(map(str, unknown)))}")
    clauses = []
    numbers = {}
    for name, field, operator in [
        ("min_price", "price", "$gte"),
        ("max_price", "price", "$lte"),
        ("min_rating", "average_rating", "$gte"),
    ]:
        value = filters.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite nonnegative number")
        try:
            value = float(value)
        except OverflowError:
            raise ValueError(f"{name} must be a finite nonnegative number") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        if name == "min_rating" and value > 5:
            raise ValueError("min_rating must be between 0 and 5")
        numbers[name] = value
        clauses.append({field: {operator: value}})
    if numbers.get("min_price", 0) > numbers.get("max_price", math.inf):
        raise ValueError("min_price must not exceed max_price")
    category = filters.get("category")
    if category is not None:
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a nonempty category label")
        clauses.append({category_key(category): {"$eq": True}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class CatalogRetriever(Runnable[str, list[Document]]):
    """Read a completed index; return at most k distinct, grounded products.

    At most 500 chunks are retrieved per query. Many matching chunks of one
    product can therefore yield fewer than k products; hard constraints are
    never relaxed just to fill the result list.
    """

    def __init__(self, persist_directory: str | Path | None = None,
                 embeddings: Embeddings | None = None, k: int = 5):
        self.k = _validate_k(k)
        self.persist_directory = Path(persist_directory or config.VECTOR_STORE_PATH).resolve()
        manifest_path = self.persist_directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("No completed index found. Run python -m src.rag.indexing first.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("The index manifest cannot be read") from error
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise ValueError("The index is incomplete; finish indexing before searching")
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported catalog schema version")
        self.catalog_path = self.persist_directory / "catalog.sqlite3"
        if not self.catalog_path.is_file():
            raise FileNotFoundError("The canonical product catalog is missing")
        embeddings = embeddings if embeddings is not None else get_embeddings()
        if manifest.get("embedding_id") != embedding_id(embeddings):
            raise ValueError("Embedding model differs from the model used to build this index")
        self.vectorstore = Chroma(
            collection_name="video_games", persist_directory=str(self.persist_directory),
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False),
            create_collection_if_not_exists=False,
        )

    def get_by_id(self, item_id: str) -> Document | None:
        """Return the authoritative whole product, or None for an unknown ID."""
        if not isinstance(item_id, str):
            raise ValueError("item_id must be a string")
        if not item_id.strip():
            return None
        with closing(sqlite3.connect(self.catalog_path.as_uri() + "?mode=ro", uri=True)) as connection:
            row = connection.execute(
                "SELECT payload FROM products WHERE item_id = ?", (item_id.strip(),)
            ).fetchone()
        return Product.model_validate_json(row[0]).as_document() if row else None

    def search(self, query: str, filters: Mapping[str, Any] | None = None,
               k: int | None = None) -> list[Document]:
        """Search by meaning with inclusive USD-price/rating/category filters.

        Category labels match exactly without case differences. Unknown prices
        and ratings cannot satisfy numeric constraints. Filtering happens inside
        Chroma before ranking, unlike the notebook's tiny all-record fixture.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        count = self.k if k is None else _validate_k(k)
        candidates = self.vectorstore.similarity_search(
            query.strip(), k=count * 10, filter=_where(filters)
        )
        results = []
        seen = set()
        for hit in candidates:
            item_id = hit.metadata.get("item_id")
            if not isinstance(item_id, str) or item_id in seen:
                continue
            product = self.get_by_id(item_id)
            if product is None:
                raise RuntimeError("Vector index contains an ID absent from the canonical catalog")
            seen.add(item_id)
            results.append(Document(page_content=hit.page_content, metadata=product.metadata))
            if len(results) == count:
                break
        return results

    def invoke(self, input: str, config: RunnableConfig | None = None,
               **kwargs: Any) -> list[Document]:
        """Preserve the LangChain retriever.invoke(query) calling convention."""
        unknown = set(kwargs) - {"filters", "k"}
        if unknown:
            raise ValueError(f"Unknown retrieval arguments: {', '.join(sorted(unknown))}")
        return self.search(input, filters=kwargs.get("filters"), k=kwargs.get("k"))


def get_retriever(k: int = 5, *, persist_directory: str | Path | None = None,
                  embeddings: Embeddings | None = None) -> CatalogRetriever:
    """Load the existing store and return a LangChain-compatible retriever.

    The original factory name, k argument and invoke usage are preserved. The
    concrete Runnable additionally provides product deduplication and get_by_id.
    """
    return CatalogRetriever(persist_directory=persist_directory, embeddings=embeddings, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local Video_Games catalog")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--min-price", type=float)
    parser.add_argument("--min-rating", type=float)
    parser.add_argument("--category")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--persist-directory", type=Path)
    args = parser.parse_args()
    filters = {name: value for name, value in {
        "min_price": args.min_price, "max_price": args.max_price,
        "min_rating": args.min_rating, "category": args.category,
    }.items() if value is not None}
    retriever = get_retriever(k=args.k, persist_directory=args.persist_directory)
    results = retriever.search(args.query, filters=filters)
    print(json.dumps([
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
