# OAD 2026 - Assignment 2: Agentic Recommender System

For complete assignment requirements, grading criteria, and deliverables, please refer to the assignment specification in `OAD_Assignment_2_2026.pdf`.

---

## Datasets & Domains

Select one of the supported domains:
- **Movie Recommender**: [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- **Recipe Recommender**: [Recipe Dataset](https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life)
- **Shopping Recommender**: [Amazon Reviews Dataset](https://amazon-reviews-2023.github.io)

You may enrich the dataset with any additional data (e.g., using external APIs like [TMDB](https://developer.themoviedb.org/) or generating images) to enhance recommendations and provide more context. Place raw dataset files in `data/raw/`.

---

## Project Structure

> **Note:** The directory and file structure below is provided as an initial guide. You are free to change, rename, extend, or reorganize everything (classes, files, modules, graphs, and UI) to best fit your group's architecture and design.

```text
oad-ws26-a2-018/
├── .env.example               # Configuration template; .env stays local
├── requirements.txt           # Python dependencies
├── README.md                  # Setup and Phase 1 handoff
├── data/
│   ├── raw/                   # Downloaded Amazon metadata (local)
│   ├── models/                # Downloaded embedding model cache (generated)
│   └── vectorstore/           # Chroma, catalog.sqlite3, manifest.json (generated)
├── docs/
│   └── phase1_verification.json # Recorded Hugging Face verification results
└── src/
    ├── agents/                # Agent workflow and state stubs for Phase 2
    ├── guardrails/            # Input/output guardrail stubs for Phase 3
    ├── rag/
    │   ├── catalog.py         # Product schema and data cleaning
    │   ├── embeddings.py      # Local MiniLM embeddings and token splitter
    │   ├── indexing.py        # Raw data -> persistent catalog and vector index
    │   └── retriever.py       # Semantic search, filters, exact ID lookup
    ├── tests/                 # Catalog, indexing, retrieval and starter tests
    ├── ui/                    # UI placeholder for Phase 4
    └── utils/
        └── config.py          # Paths, embedding and API settings
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Access to an LLM endpoint (LLMDuckt API key from TeachCenter, or OpenAI API)

### Installation
1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API credentials:
   ```bash
   cp .env.example .env
   ```

4. **Prepare Dataset**:
   Place your chosen domain dataset in `data/raw/`.

---

## Running the Application

### 1. Run Tests
```bash
pytest
```

### 2. Ingest & Index Data
```bash
python -m src.rag.indexing
```

### 3. Launch the Application
```bash
python -m src.ui.app
```

---

# Phase 1 Handoff: Foundation and Data Grounding

See the [Requirements Specification](docs/requirements_specification.md) for the project requirements.

## Implemented by Flo

- **Catalog:** Amazon Reviews 2023 / `Video_Games` item metadata (McAuley Lab).
  Cleaning removes HTML, normalizes missing values, rejects unusable IDs/titles,
  and deduplicates products while preserving the original `parent_asin` as `item_id`.
- **RAG pipeline:** 220-token chunks with 24-token overlap, local
  `all-MiniLM-L6-v2` through LangChain `HuggingFaceEmbeddings`, 384-dimensional
  normalized vectors, and persistent Chroma storage. Maintained LangChain
  integrations keep the implementation small and explainable.
- **Retriever:** semantic search, price/rating/category filters, and exact
  `get_by_id` lookup. Full product metadata is stored separately in SQLite.
- **Recorded full build:** 137,260 products and 235,473 chunks; previous checks
  are documented in [the verification report](docs/phase1_verification.json).

## Run locally

Use **Python 3.12** for the environment in Getting Started above (tested version;
indexing requires Python 3.11+). Local indexing and search require no LLM API key.
Run these commands from the repository root with the environment activated:

```bash
export EMBEDDING_PROVIDER=local
mkdir -p data/raw
# Download once; keep the file compressed.
curl --fail --location --output data/raw/meta_Video_Games.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz

python -m pytest -q
# Build once; skip this if data/vectorstore/ already contains a complete index.
python -m src.rag.indexing
python -m src.rag.retriever --query "mechanical gaming keyboard" \
  --category PC --max-price 100 --min-rating 4 --k 3
```

Raw data, model weights, and the index are excluded from Git, so a fresh checkout
needs its own local build. The model downloads on first use and then runs locally.
The indexer refuses a nonempty output directory; use a different
`--persist-directory` for a new build. Indexing and retrieval must use the same
embedding model/settings. The generic UI command above belongs to a later phase;
Phase 1 is usable through the retriever CLI and Python interface below.

## Integration for Phase 2

Initialize the retriever once and reuse it in the agent's retrieval node/tool:

```python
from src.rag.retriever import get_retriever

retriever = get_retriever(k=5)
hits = retriever.search(
    "gaming headset with microphone",
    filters={"max_price": 100, "min_rating": 4, "category": "PC"},
)
product = retriever.get_by_id(hits[0].metadata["item_id"]) if hits else None
# LangChain Runnable interface: retriever.invoke("gaming headset with microphone")
```

`search(query, filters=None, k=None)` returns `list[Document]` with distinct
products; `get_by_id(item_id)` returns `Document | None`. No matches return `[]`.
Supported filters are `min_price`, `max_price`, `min_rating`, and `category`
(an exact existing category label, case-insensitive). Unknown prices/ratings do
not satisfy the corresponding numeric filters. Keep product IDs with candidates
so later output guardrails can verify them.

The fixed `Document.metadata` schema is defined by `Product` in
[src/rag/catalog.py](src/rag/catalog.py):

| Fields | Type |
|---|---|
| `item_id`, `title`, `description`, `currency`, `source` | `str` |
| `features`, `categories`, `review_snippets` | `list[str]` |
| `price`, `average_rating` | `float` or `None` |
| `rating_number` | `int` or `None` |
| `brand`, `image_url` | `str` or `None` |

Missing descriptions are `""`, missing lists are `[]`; reviews are not ingested (yet...),
so `review_snippets` is empty.
Search `page_content` contains the matching chunk; metadata contains all canonical
product fields. `get_by_id` also returns the complete assembled product text.
Similarity selects related candidates; the agent must check concrete user needs.