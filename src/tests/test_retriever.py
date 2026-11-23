"""Behavioral tests with a real persisted Chroma index and deterministic vectors.
The fake model proves indexing/search/filter plumbing only, not semantic quality.
"""
import json
import math
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
import pytest

from src.rag.indexing import index_dataset
from src.rag.retriever import CatalogRetriever, get_retriever


class FixtureEmbeddings(Embeddings):
    model_id = "fixture-keywords-v1"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        text = text.casefold()
        vector = [float(text.count(word)) for word in ["headset", "keyboard", "adventure"]] + [0.1]
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    directory = tmp_path_factory.mktemp("retrieval")
    raw = directory / "products.jsonl"
    products = [
        {"parent_asin": "HEADSET20", "title": "Wireless gaming headset", "description": ["Headset with soft ear pads. " * 200], "price": 20, "average_rating": 4.5, "categories": ["Video Games", "PC", "Headsets"]},
        {"parent_asin": "HEADSET_UNKNOWN", "title": "Another headset", "description": ["Headset microphone"], "price": None, "average_rating": 4.7, "categories": ["Video Games", "PC", "Headsets"]},
        {"parent_asin": "KEYBOARD100", "title": "Gaming keyboard", "description": ["Mechanical keyboard"], "price": 100, "average_rating": 4.8, "categories": ["Video Games", "PC", "Keyboards"]},
        {"parent_asin": "HEADSET60", "title": "Premium headset", "description": ["Headset for console players"], "price": 60, "average_rating": 3.2, "categories": ["Video Games", "Headsets"]},
        {"parent_asin": "GAME30", "title": "Adventure game", "description": ["Adventure in a fantasy world"], "price": 30, "average_rating": 4.4, "categories": ["Video Games", "Nintendo Switch", "Games"]},
    ]
    raw.write_text("\n".join(json.dumps(row) for row in products) + "\n", encoding="utf-8")
    store = directory / "index"
    index_dataset(str(raw), persist_directory=store, embeddings=FixtureEmbeddings(), min_products=1, batch_size=2)
    return store


def test_factory_invoke_and_product_deduplication(catalog):
    retriever = get_retriever(5, persist_directory=catalog, embeddings=FixtureEmbeddings())
    documents = retriever.invoke("headset", {"configurable": {"thread_id": "session"}})
    ids = [document.metadata["item_id"] for document in documents]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"HEADSET20", "HEADSET_UNKNOWN", "HEADSET60", "KEYBOARD100", "GAME30"}
    assert retriever.get_by_id("HEADSET20").metadata["price"] == 20


def test_combined_filters_are_hard_and_inclusive(catalog):
    retriever = get_retriever(persist_directory=catalog, embeddings=FixtureEmbeddings())
    documents = retriever.search("headset", filters={"min_price": 20, "max_price": 20, "min_rating": 4.5, "category": " headsets "})
    assert [document.metadata["item_id"] for document in documents] == ["HEADSET20"]
    assert documents[0].metadata["categories"] == ["Video Games", "PC", "Headsets"]


def test_retriever_can_be_used_in_a_langchain_pipeline(catalog):
    retriever = get_retriever(1, persist_directory=catalog, embeddings=FixtureEmbeddings())
    pipeline = retriever | RunnableLambda(lambda documents: documents[0].metadata["item_id"])
    assert pipeline.invoke("keyboard") == "KEYBOARD100"


def test_unknown_price_is_kept_but_cannot_match_a_budget(catalog):
    retriever = CatalogRetriever(catalog, FixtureEmbeddings())
    product = retriever.get_by_id("HEADSET_UNKNOWN")
    assert product is not None
    assert product.metadata["price"] is None
    documents = retriever.search("headset", {"max_price": 100})
    assert "HEADSET_UNKNOWN" not in {document.metadata["item_id"] for document in documents}


def test_get_by_id_returns_full_product_and_handles_unknown_id(catalog):
    retriever = CatalogRetriever(catalog, FixtureEmbeddings())
    product = retriever.get_by_id("HEADSET20")
    assert product is not None
    assert "Wireless gaming headset" in product.page_content
    assert "soft ear pads" in product.page_content
    assert retriever.get_by_id("DOES_NOT_EXIST") is None
    assert retriever.get_by_id("") is None


def test_unsatisfied_constraints_return_empty_list(catalog):
    retriever = CatalogRetriever(catalog, FixtureEmbeddings())
    assert retriever.search("headset", {"category": "No such category"}) == []
    assert retriever.search("headset", {"max_price": 1}) == []


@pytest.mark.parametrize("filters", [
    {"max_price": -1}, {"min_rating": 6}, {"max_price": float("nan")},
    {"max_price": True}, {"max_price": "20"},
    {"min_price": 30, "max_price": 20}, {"category": " "},
    {"category": []}, {"unknown": 1}, [],
])
def test_invalid_filters_are_rejected_before_retrieval(catalog, filters):
    retriever = CatalogRetriever(catalog, FixtureEmbeddings())
    with pytest.raises(ValueError):
        retriever.search("headset", filters)


@pytest.mark.parametrize("k", [0, 51, -1, True, 1.5])
def test_invalid_result_count(catalog, k):
    with pytest.raises(ValueError):
        CatalogRetriever(catalog, FixtureEmbeddings(), k=k)


def test_blank_query_is_rejected(catalog):
    with pytest.raises(ValueError):
        CatalogRetriever(catalog, FixtureEmbeddings()).search(" ")


def test_invoke_does_not_silently_ignore_misspelled_filters(catalog):
    with pytest.raises(ValueError, match="Unknown retrieval arguments"):
        CatalogRetriever(catalog, FixtureEmbeddings()).invoke("headset", filter={"max_price": 1})


def test_missing_or_incomplete_index_is_not_silently_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        CatalogRetriever(tmp_path, FixtureEmbeddings())
    assert not (tmp_path / "chroma.sqlite3").exists()
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "status": "building"}))
    with pytest.raises(ValueError, match="incomplete"):
        CatalogRetriever(tmp_path, FixtureEmbeddings())


def test_embedding_model_mismatch_is_rejected(catalog):
    manifest_path = Path(catalog) / "manifest.json"
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["embedding_id"] = "different-model"
    manifest_path.write_text(json.dumps(manifest))
    try:
        with pytest.raises(ValueError, match="Embedding model differs"):
            CatalogRetriever(catalog, FixtureEmbeddings())
    finally:
        manifest_path.write_text(original)


def test_huggingface_model_revision_and_normalization_affect_identity():
    """Equal vector dimensions must not make different encoders interchangeable."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.rag.embeddings import embedding_id

    def config_only(revision, normalize):
        return HuggingFaceEmbeddings.model_construct(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'revision': revision},
            encode_kwargs={'normalize_embeddings': normalize})

    original = embedding_id(config_only('revision-a', True))
    assert original != embedding_id(config_only('revision-b', True))
    assert original != embedding_id(config_only('revision-a', False))
