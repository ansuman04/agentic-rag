# Agentic RAG — Local PDF Question Answering with CrewAI and Ollama

A learning and portfolio project that builds a **local Retrieval-Augmented Generation (RAG)** pipeline, then incrementally adds **CrewAI agents** for routing and document retrieval.

**Current status:** Traditional PDF RAG works end-to-end. A CrewAI router and a document retriever are connected through an orchestrator: document questions are routed to ChromaDB retrieval. Live web search, grading, and answer validation are planned—not yet implemented.

## Architecture

### Implemented: traditional RAG

```text
PDF → PyPDF extraction → overlapping text chunks → BGE Small embeddings
                                                       ↓
                                                ChromaDB (local)
                                                       ↑
User question → BGE Small query embedding → top-k matching chunks
                                                       ↓
                                     Qwen2.5 7B via local Ollama
                                                       ↓
                                         Answer + source references
```

### Implemented: CrewAI agent orchestration

```text
User question → AgentOrchestrator → CrewAI Router Agent
                                          │
                     ┌────────────────────┴────────────────────┐
                "vectorstore"                              "websearch"
                     ↓                                          ↓
     CrewAI Document Retriever Agent              "Web search is not
     → DocumentSearchTool → ChromaDB               implemented yet."
     → source passages with page metadata          (no evidence returned)
```

`AgentOrchestrator` (`src/orchestrator.py`) takes the router and retriever as plain callables, so the routing logic can be unit-tested without CrewAI or Ollama. `build_local_orchestrator()` wires in the real CrewAI agents and the local ChromaDB collection. An unrecognized route or an empty question raises `ValueError`.

The `websearch` branch is **a placeholder** for now: there is no web search tool yet.

## Technology stack

| Area | Technology | Purpose |
| --- | --- | --- |
| Runtime | Python 3.12 | Application and tests |
| PDF extraction | PyPDF | Read PDF text |
| Embeddings | `BAAI/bge-small-en-v1.5` via Sentence Transformers | 384-dimensional text vectors |
| Vector database | ChromaDB | Persistent local similarity search |
| Local inference | Ollama + `qwen2.5:7b` | Answer generation and agent reasoning |
| Agents | CrewAI | Router and document retriever |
| Tests | pytest, unittest.mock, ReportLab | Automated service/tool tests and generated PDF fixtures |
| Version control | Git / GitHub | Feature branches and pull requests |

## Features implemented

- Extract text from PDFs, preserving page number and source filename on each chunk.
- Split page text into overlapping character chunks (default: 1,000 characters, 200 overlap).
- Generate embeddings with BGE Small and persist them in ChromaDB.
- Index document chunks with content-hash-based IDs and metadata.
- Replace existing indexed chunks for the same source filename during re-indexing.
- Search by semantic similarity, returning text, cosine distance, and metadata.
- Generate locally hosted answers using retrieved passages, with requested `[n]` source citations.
- Route questions with a CrewAI Router Agent to `vectorstore` or `websearch` labels.
- Run a CrewAI Document Retriever Agent that can use a `DocumentSearchTool` wrapping the vector store.
- Orchestrate router → retriever: `vectorstore` questions run document retrieval; `websearch` questions skip it and return a placeholder message.
- Run 11 automated tests (as of the orchestrator feature); manual agent scripts are separate.

**Limitations:** The existing indexing replacement deletes old chunks before upserting the new version and is not atomic; files with the same filename can collide. Citations and factual grounding are requested from the LLM but not independently verified. Retrieval is character-chunk-based. The orchestrator returns retrieved evidence, not a generated answer, and the `websearch` route has no search tool behind it.

## Project structure

```text
agentic-rag/
├── data/                         # Local PDFs (gitignored)
├── src/
│   ├── __init__.py
│   ├── document_loader.py        # PDF reading and page-level chunks
│   ├── vector_store.py           # Embeddings and ChromaDB
│   ├── llm_client.py             # Ollama API wrapper
│   ├── rag_service.py            # Retrieve context and generate answer
│   ├── agents.py                 # CrewAI Router and Retriever agents
│   ├── tasks.py                  # CrewAI routing and retrieval tasks
│   ├── tools.py                  # CrewAI DocumentSearchTool
│   └── orchestrator.py           # Router → retriever orchestration
├── tests/
│   ├── conftest.py               # Temporary PDF fixture
│   ├── test_vector_store.py
│   ├── test_rag_service.py
│   ├── test_document_search_tool.py
│   └── test_orchestrator.py
├── main.py                       # Traditional RAG demo
├── test_router_manual.py         # Manual router check
├── test_retriever_manual.py      # Manual retriever check
├── test_orchestrator_manual.py   # Manual end-to-end routing check
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started (Windows / PowerShell)

### Prerequisites

- Python **3.12**
- Git
- [Ollama for Windows](https://ollama.com/download/windows) installed and running
- A PDF that you are allowed to use locally

### 1. Clone and create an environment

```powershell
git clone https://github.com/ansuman04/agentic-rag.git
cd agentic-rag
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either use the virtual environment's interpreter directly (`.\.venv\Scripts\python.exe`) or choose an execution policy appropriate to your system's security policy.

### 2. Download the local model

```powershell
ollama pull qwen2.5:7b
ollama list
```

Ollama must be running and reachable locally at `http://localhost:11434`. The BGE embedding model downloads and caches on first use.

### 3. Add a PDF

Place a text-based PDF at:

```text
data/sample.pdf
```

This path is **not** committed to GitHub. You can use the [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762) for personal study, subject to its terms.

### 4. Run traditional RAG

```powershell
python main.py
```

The demo ingests and indexes the PDF, retrieves relevant chunks for its configured question, then asks Qwen2.5 to answer from them. The first run may take longer while downloading the embedding model. The current demo re-indexes the PDF on every execution.

### 5. Run the CrewAI demos

```powershell
python test_router_manual.py
python test_retriever_manual.py
python test_orchestrator_manual.py
```

The router script checks route selection on its own, and the retriever script checks document retrieval on its own. The orchestrator script runs the connected flow on two questions: a document question (routed to `vectorstore`, returns evidence) and a weather question (routed to `websearch`, returns the not-implemented message).

The retriever and orchestrator scripts need a local ChromaDB collection populated by `python main.py`. If it is empty, the orchestrator stops with "No indexed documents found."

### 6. Run automated tests

```powershell
python -m pytest tests/ -v
```

The test suite generates its own small PDF using a pytest fixture. It does not require `data/sample.pdf` or an Ollama server: the RAG-service tests mock the LLM, and the orchestrator tests use stub router and retriever functions. The vector store tests may download the local embedding model if it is not already cached.

## Example

**Question:** Why do Transformers use positional encoding?

**Expected workflow:** ChromaDB retrieves relevant passages from the Transformer PDF; the locally running model composes an explanation and references retrieved pages. The source references are informative, not yet independently audited for support.

## Privacy and repository hygiene

- `.env`, `.venv/`, `data/` contents, `chroma_db/`, model caches and `.crewai/` runtime files are ignored by Git.
- Do not commit API keys, confidential documents or customer data.
- CrewAI may offer to **share execution traces** with its service. Decline unless you intend to upload the prompt, outputs and other trace details.
- Ollama model inference is local, but dependencies and model downloads require network access during setup.

## Roadmap

- [x] PDF extraction, chunking and page metadata
- [x] BGE embeddings and persistent ChromaDB retrieval
- [x] Local Ollama answer generation
- [x] CrewAI Router Agent (independent test)
- [x] CrewAI Document Retriever Agent and ChromaDB tool (independent test)
- [x] Automated tests with generated PDF fixtures
- [x] Connect Router → Retriever in an orchestration flow
- [ ] Implement real web-search retrieval
- [ ] Add relevance and hallucination/grounding checks
- [ ] Improve citation validation, retrieval ranking, and test coverage
- [ ] Make indexing replacement failure-safe and handle filename collisions
- [ ] Add deployment, monitoring and evaluation tooling

## Development workflow

Changes are developed on feature branches, tested with pytest, and merged into `main` through pull requests. GitHub can automatically delete remote feature branches after merging.

## Disclaimer

This is an educational project, not a production-ready knowledge assistant. Verify answers against original sources before relying on them for consequential decisions.
