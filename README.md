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
oad2026-assignment2/
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── README.md                  # Project instructions
├── OAD_Assignment_2_2026.pdf  # Assignment specification
├── data/                      # Dataset files & vector store persistence
│   └── README.md              # Instructions for datasets
├── docs/                      # Required project documentation & deliverables
│   ├── requirements_specification.md # Document 1: Requirements specification
│   ├── class_diagrams.md             # Document 2: Conceptual design & UML diagrams
│   └── system_description.md         # Document 3: System architecture & design patterns
└── src/                       # Application source code
    ├── __init__.py
    ├── agents/                # Agent workflow & state
    │   ├── __init__.py
    │   ├── graph.py           # StateGraph builder stub
    │   ├── nodes.py           # Workflow node stubs (examples)
    │   ├── prompts.py         # System prompt templates (examples)
    │   ├── schemas.py         # Pydantic schemas (examples)
    │   └── state.py           # AgentState definition (example)
    ├── guardrails/            # Input & output validation stubs
    │   ├── __init__.py
    │   ├── input_guardrails.py  # Input safety & off-topic filtering
    │   └── output_guardrails.py # Output catalog grounding verification
    ├── rag/                   # RAG dataset indexing & catalog retrieval
    │   ├── __init__.py
    │   ├── indexing.py        # Dataset preprocessing & vector store persistence
    │   └── retriever.py       # Catalog retriever interface
    ├── tests/                 # Unit & integration tests
    │   ├── __init__.py
    │   └── test_stubs.py      # Baseline smoke tests
    ├── ui/                    # User interface & interaction
    │   ├── __init__.py
    │   └── app.py             # Application entry point
    └── utils/                 # Utilities and configuration
        ├── __init__.py
        └── config.py          # Configuration & environment variable loader
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
