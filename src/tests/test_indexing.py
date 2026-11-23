"""Offline ingestion/persistence tests using tiny injected embeddings."""

import gzip
import json
import math
import sqlite3
import urllib.request

from langchain_core.embeddings import Embeddings
import pytest

from src.rag import indexing
from src.rag.embeddings import LocalMiniLMEmbeddings, MODEL_NAME, MODEL_REVISION
from src.rag.retriever import get_retriever
from src.utils import config


class OfflineEmbeddings(Embeddings):
    model_id = "phase1-tests:offline-v1"

    def __init__(self):
        self.batch_sizes = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        vectors = []
        for text in texts:
            words = text.casefold()
            vector = [float(token in words) for token in ("controller", "headset", "mouse")]
            vector.append(0.1)
            length = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / length for value in vector])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture(autouse=True)
def forbid_network_and_default_embedding_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline ingestion tests must not download or use API embeddings")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(indexing, "get_embeddings", forbidden)


def write_source(path, records):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def read_manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def catalog_payloads(directory):
    with sqlite3.connect(directory / "catalog.sqlite3") as connection:
        return dict((item_id, json.loads(payload)) for item_id, payload in
                    connection.execute("SELECT item_id, payload FROM products"))


@pytest.mark.parametrize("filename", ["products.jsonl", "products.jsonl.gz"])
def test_ingestion_cleans_deduplicates_and_reopens_without_network(tmp_path, filename):
    source = write_source(tmp_path / filename, [
        {"parent_asin": "FIRST", "title": "<b>Controller</b>", "price": None},
        {"parent_asin": "FIRST", "title": "Later conflicting title", "price": 999},
        {"parent_asin": "SECOND", "title": "Headset", "price": 25, "average_rating": 4.2},
        {"parent_asin": "BAD", "title": "  "},
        {"title": "No ID"},
        ["not", "an", "object"],
    ])
    directory = tmp_path / "store"
    embeddings = OfflineEmbeddings()
    indexing.index_dataset(source, persist_directory=directory, embeddings=embeddings,
                           min_products=1, batch_size=1)

    manifest = read_manifest(directory)
    assert manifest["status"] == "complete"
    assert manifest["product_count"] == 2
    assert manifest["chunk_count"] >= 2
    assert manifest["cleaning"]["duplicate_products"] == 1
    assert manifest["cleaning"]["invalid_identity_or_title"] == 3
    assert manifest["assignment_size_met"] is False
    payloads = catalog_payloads(directory)
    assert payloads["FIRST"]["title"] == "Controller"
    assert payloads["FIRST"]["price"] is None
    assert max(embeddings.batch_sizes) <= 1

    reopened = get_retriever(persist_directory=directory, embeddings=OfflineEmbeddings())
    assert reopened.get_by_id("FIRST").metadata["title"] == "Controller"
    assert reopened.get_by_id("missing") is None
    hits = reopened.invoke("controller")
    assert hits
    assert {hit.metadata["item_id"] for hit in hits} <= {"FIRST", "SECOND"}


def test_full_build_enforces_product_minimum_before_embedding(tmp_path):
    source = write_source(tmp_path / "tiny.jsonl", [{"parent_asin": "P", "title": "Game"}])
    directory = tmp_path / "store"
    embeddings = OfflineEmbeddings()

    with pytest.raises(ValueError, match="require 100000"):
        indexing.index_dataset(source, persist_directory=directory, embeddings=embeddings)

    assert not embeddings.batch_sizes
    manifest = read_manifest(directory)
    assert manifest["status"] == "failed"
    assert manifest["minimum_products"] == 100000
    assert manifest["cleaning"]["products"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        get_retriever(persist_directory=directory, embeddings=embeddings)


def test_explicit_sample_can_be_small_and_never_overwrites_completed_index(tmp_path):
    source = write_source(tmp_path / "tiny.jsonl", [
        {"parent_asin": "P1", "title": "Controller"},
        {"parent_asin": "P2", "title": "Headset"},
    ])
    directory = tmp_path / "sample"
    embeddings = OfflineEmbeddings()
    indexing.index_dataset(source, persist_directory=directory, embeddings=embeddings, limit=1)
    previous_manifest = (directory / "manifest.json").read_bytes()
    previous_payloads = catalog_payloads(directory)

    assert read_manifest(directory)["product_count"] == 1
    assert read_manifest(directory)["sample_limit"] == 1
    assert read_manifest(directory)["assignment_size_met"] is False
    with pytest.raises(FileExistsError, match="not empty"):
        indexing.index_dataset(source, persist_directory=directory, embeddings=embeddings, limit=2)
    assert (directory / "manifest.json").read_bytes() == previous_manifest
    assert catalog_payloads(directory) == previous_payloads


def test_nonempty_target_is_not_modified_even_if_it_is_not_an_index(tmp_path):
    source = write_source(tmp_path / "tiny.jsonl", [{"parent_asin": "P", "title": "Game"}])
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "user_notes.txt"
    marker.write_text("Keep my work")

    with pytest.raises(FileExistsError):
        indexing.index_dataset(source, persist_directory=directory, embeddings=OfflineEmbeddings(), limit=1)
    assert marker.read_text() == "Keep my work"
    assert sorted(path.name for path in directory.iterdir()) == ["user_notes.txt"]


def test_malformed_json_reports_line_and_leaves_failed_manifest(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"parent_asin":"P","title":"Controller"}\n{broken JSON}\n')
    directory = tmp_path / "store"
    embeddings = OfflineEmbeddings()

    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        indexing.index_dataset(source, persist_directory=directory, embeddings=embeddings, min_products=1)
    assert not embeddings.batch_sizes
    manifest = read_manifest(directory)
    assert manifest["status"] == "failed"
    assert manifest["error"] == "ValueError"
    with pytest.raises(ValueError, match="incomplete"):
        get_retriever(persist_directory=directory, embeddings=embeddings)


@pytest.mark.parametrize("status", ["preparing", "indexing", "failed"])
def test_incomplete_index_is_rejected_before_any_database_or_model_access(tmp_path, status):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError, match="incomplete"):
        get_retriever(persist_directory=tmp_path)


def test_embedding_failure_never_labels_a_partial_index_complete(tmp_path):
    class BrokenEmbeddings(OfflineEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("Deliberate offline embedding failure")

    source = write_source(tmp_path / "tiny.jsonl", [{"parent_asin": "P", "title": "Game"}])
    directory = tmp_path / "store"
    embeddings = BrokenEmbeddings()
    with pytest.raises(RuntimeError, match="Deliberate offline"):
        indexing.index_dataset(source, persist_directory=directory, embeddings=embeddings,
                               min_products=1, batch_size=1)
    assert read_manifest(directory)["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        get_retriever(persist_directory=directory, embeddings=embeddings)


def test_local_token_windows_keep_final_evidence_and_stay_within_model_window():
    model_path = config.MODEL_PATH
    snapshot = model_path / "huggingface" / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / MODEL_REVISION
    if not (snapshot / "model.safetensors").is_file():
        pytest.skip("Local Hugging Face model is absent; this test never downloads it")
    embeddings = LocalMiniLMEmbeddings(model_path=model_path, local_files_only=True)
    text = "wireless controller " * 500 + "finalevidence"
    chunks = embeddings.chunk_text(text)
    assert len(chunks) > 1
    assert "finalevidence" in chunks[-1]
    assert all(len(embeddings.tokenizer.encode(chunk, add_special_tokens=False)) <= 254
               for chunk in chunks)
