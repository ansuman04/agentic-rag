# Agentic RAG — Local PDF and Web Retrieval with CrewAI, ChromaDB, Tavily, and Ollama

An educational and portfolio project that combines **Retrieval-Augmented Generation (RAG)** with **CrewAI-based question routing**. It supports two information sources: locally indexed PDF documents in ChromaDB and public web search through Tavily. Retrieval evidence is kept separate from LLM-generated summaries so future grading, citation verification, and evaluation can use the actual retrieved material.

**Current status (November 10, 2026):** Traditional local PDF RAG works end-to-end. The newer CrewAI orchestration routes questions to either document retrieval or live web retrieval, then summarizes the returned evidence with local Qwen2.5 through Ollama. Both paths return structured evidence and a separate agent-generated summary. Recognized LLM HTTP 500 failures during summarization can be reported without discarding successfully retrieved evidence. **29 automated tests passed** in the most recent local run; both manual routing scenarios completed successfully.

> **Scope:** The orchestration flow currently *retrieves and summarizes* evidence. It does not yet run a dedicated relevance grader, hallucination/grounding grader, or independently verified final-answer generator.

## Architecture

### 1. Traditional PDF RAG (`main.py`)

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

This is the original, separate `RAGService` implementation: it retrieves document passages and asks the local model to answer using them. Source references are requested from the model but not independently verified.

### 2. CrewAI orchestration (`test_orchestrator_manual.py`)

```text
                         User question
                               |
                               v
                    CrewAI Router Agent
                       (Qwen2.5 / Ollama)
                               |
                  +------------+------------+
                  |                         |
             "vectorstore"               "websearch"
                  |                         |
                  v                         v
       DocumentRetrievalService      WebRetrievalService
                  |                         |
                  v                         v
        ChromaDB similarity search     WebSearchTool → Tavily
                  |                         |
                  v                         v
        Structured PDF passages       Structured web results
       (text, source, page, etc.)    (title, URL, snippet)
                  |                         |
                  v                         v
       Document Evidence             Web Evidence
       Summarizer (CrewAI)           Summarizer (CrewAI)
                  |                         |
                  +------------+------------+
                               |
                               v
                    Structured response:
                    - question
                    - route
                    - status
                    - evidence       (original retrieved data)
                    - agent_output   (LLM-generated summary)
                    - error          (if applicable)
```

**Separation of responsibilities:**

- The **Router Agent** decides between `vectorstore` and `websearch`; it does not retrieve documents or websites itself.
- **Python retrieval services** call ChromaDB or Tavily directly, once per normal branch execution. They convert results to a structured `list[dict]` **without asking the LLM to rewrite the evidence**.
- **CrewAI summarizers** receive the already-retrieved evidence as task input. They have no retrieval tools attached, so they do not initiate another Tavily or ChromaDB search.
- `AgentOrchestrator` dispatches each question and returns the evidence separately from the generated summary. This enables later grounding checks against retrieved material rather than an earlier LLM paraphrase.

`AgentOrchestrator` (`src/orchestrator.py`) accepts plain Python callables for its router and retrieval branches, making orchestration testable without live services. `build_local_orchestrator()` wires in the actual components. Empty questions or unrecognized route labels raise `ValueError`.

### 3. Retrieval-response contract

A successful web request has the following *shape* (sample content is illustrative):

```json
{
  "question": "What is the weather in Bengaluru today?",
  "route": "websearch",
  "status": "retrieved",
  "evidence": [
    {
      "title": "Example source",
      "url": "https://example.com/weather",
      "snippet": "Example retrieved information"
    }
  ],
  "agent_output": "Summary based on the sources.",
  "error": null
}
```

For `vectorstore`, each evidence item instead includes `text`, `source`, `page`, `chunk_index`, `document_id`, and `distance` (cosine-distance output from the retrieval layer).

The current response statuses are:

| Status | Meaning |
| --- | --- |
| `retrieved` | Evidence was found and the summarizer returned output. This **does not** mean facts were independently validated. |
| `no_evidence` | The retrieval service found no results; `agent_output` is `null`. |
| `generation_failed` | Retrieval succeeded, but a recognized LLM server error interrupted summarization; structured evidence is still returned. |

Recovery currently catches `openai.InternalServerError` from the Ollama/OpenAI-compatible summarization call (including the observed HTTP 500 token-repetition failure). Other errors generally propagate. A Router Agent failure happens **before** retrieval and is not covered by this post-retrieval recovery.

The evidence is retained **in memory for the request**, not saved to a new evidence database. Web snippets are capped at **500 characters each** before being returned from `WebSearchTool`; this preserves the exact normalized tool results, **not** full original web-page contents.

## Technology stack

| Area | Technology | Purpose |
| --- | --- | --- |
| Runtime | Python 3.12 | Application and tests |
| PDF extraction | PyPDF (`pypdf`) | Read PDF text |
| Embeddings | `BAAI/bge-small-en-v1.5` via Sentence Transformers | 384-dimensional vectors |
| Vector database | ChromaDB | Persistent local similarity search |
| Local inference | Ollama + `qwen2.5:7b` | Traditional RAG and CrewAI agent reasoning |
| Agents and tasks | CrewAI | Routing, evidence summarization, and legacy tool-based demos |
| Live web search | Tavily (`tavily-python`) | Public web results with titles, URLs, and snippets |
| LLM-compatible client | OpenAI Python client (via CrewAI) | Connect CrewAI to local Ollama endpoint |
| Tests | pytest, unittest.mock, ReportLab | Unit tests, mocked integrations, generated PDF fixtures |
| Version control | Git / GitHub | Feature branches and pull requests |

**Anthropic is not required** for this setup. The currently tested orchestration uses local Ollama; provider switching can be added later.

## Features implemented

- Read PDF text page by page, retaining source filename and page metadata.
- Split pages into overlapping character chunks (defaults: 1,000 characters with 200-character overlap).
- Generate BGE Small embeddings and index/retrieve chunks from persistent ChromaDB.
- Use content-derived document identifiers and replace old chunks for the same filename when re-indexing.
- Retrieve semantically similar document chunks with source, page, chunk, document ID, and cosine distance.
- Run traditional local PDF RAG using `RAGService` and Ollama.
- Route CrewAI requests to `vectorstore` or `websearch`.
- Retrieve structured document evidence via `DocumentRetrievalService` and structured web evidence via `WebRetrievalService`.
- Call Tavily through `WebSearchTool`, returning up to five results with a 500-character snippet cap.
- Run tool-free CrewAI **Document Evidence Summarizer** and **Web Evidence Summarizer** over supplied evidence.
- Return `evidence` and `agent_output` as separate fields with a consistent response format on both routes.
- Preserve successfully retrieved evidence if an `openai.InternalServerError` occurs during summarization.
- Retain the original `DocumentSearchTool`, tool-based Document Retriever Agent, and related manual tests for learning/reference.
- Run **29 automated tests** (latest verified local result: November 10, 2026). Manual agent/integration scripts are separate.

### Known limitations

- **No relevance or grounding grader yet.** A fluent summary can still ignore relevant data or make unsupported claims.
- **Web result quality varies.** Tavily may return monthly forecasts, old pages, or otherwise irrelevant snippets for a current-information question; results are not yet reranked or freshness-validated.
- **Retrieval ranking is imperfect.** The lowest-distance ChromaDB chunk is not always the most useful passage.
- **Web evidence is truncated.** The preserved snippet is the tool's 500-character representation, not the full Tavily result or page text.
- **PDF ingestion is not failure-atomic.** Existing chunks are deleted before replacements are written, and different files with the same name can collide.
- **Citations are not independently checked.** Returned titles/URLs/page numbers reflect retrieved metadata, but summaries may still misinterpret them.
- **LLM recovery is limited.** Recognized HTTP 500 summarization errors are handled after retrieval; router failures, other LLM exceptions, malformed search data, and external API failures may still raise errors.
- **Not yet a production service.** No API server, authentication, deployment, persistent evaluation results, or operational monitoring is included.

## Project structure

```text
agentic-rag/
├── data/                              # Local PDFs (should be gitignored)
├── chroma_db/                         # Persisted local vector index (should be gitignored)
├── src/
│   ├── __init__.py
│   ├── document_loader.py             # PDF extraction and chunking
│   ├── vector_store.py                # Embeddings, ChromaDB indexing/search
│   ├── document_retrieval.py          # Structured ChromaDB evidence
│   ├── web_tools.py                   # Tavily WebSearchTool; 500-char snippets
│   ├── web_retrieval.py               # Structured Tavily evidence
│   ├── llm_client.py                  # Ollama client for traditional RAG
│   ├── rag_service.py                 # Traditional RAG answering
│   ├── agents.py                      # Router, summarizers, legacy agents
│   ├── tasks.py                       # CrewAI task instructions
│   ├── tools.py                       # Legacy CrewAI DocumentSearchTool
│   └── orchestrator.py                # Routing, response format, LLM recovery
├── tests/
│   ├── conftest.py                    # Generated PDF fixture
│   ├── test_vector_store.py
│   ├── test_rag_service.py
│   ├── test_document_search_tool.py
│   ├── test_document_retrieval.py
│   ├── test_web_search_tool.py
│   ├── test_web_retrieval.py
│   ├── test_orchestrator.py
│   └── test_llm_failure_recovery.py
├── main.py                            # Traditional RAG demo
├── test_router_manual.py              # Legacy manual Router test
├── test_retriever_manual.py           # Legacy tool-based Retriever test
├── test_orchestrator_manual.py        # Both routes, evidence + summary
├── test_web_retrieval_manual.py       # Direct Tavily retrieval (no LLM)
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started (Windows / PowerShell)

### Prerequisites

- Python **3.12** and Git
- [Ollama for Windows](https://ollama.com/download/windows), installed and running
- A [Tavily](https://tavily.com/) API key for the web-search branch
- A text-based PDF that you are permitted to process locally

### 1. Clone and create a Python environment

```powershell
git clone https://github.com/ansuman04/agentic-rag.git
cd agentic-rag
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> The changes described here are being developed in `feature/web-search`. Until the branch is merged, check out that branch before testing these new features: `git fetch origin; git switch feature/web-search` (assuming it has already been pushed). If the changes exist only in your local working tree, stay on your current feature branch.

If PowerShell blocks activation, run `\.venv\Scripts\python.exe` directly or select an execution-policy approach appropriate for your system.

### 2. Download and start the local LLM

```powershell
ollama pull qwen2.5:7b
ollama list
```

Ollama must be reachable at `http://localhost:11434`; start the application/service if it is not already running. The BGE embedding model downloads and caches on first use.

### 3. Configure the Tavily API key

For the current PowerShell session:

```powershell
$env:TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"
python -c "import os; print('Tavily configured:', bool(os.getenv('TAVILY_API_KEY')))"
```

Replace the placeholder with your actual key **locally**. Do not paste secrets into issues, screenshots, chat logs, or the repository. A PowerShell session variable must be configured again in a new session unless you adopt a local secret-loading mechanism.

Tavily is needed for the `websearch` route, not for document-only retrieval or mocked unit tests. The web tool uses basic-depth search with a maximum of five results and a 500-character snippet cap.

### 4. Add and index a PDF

Place a text-based PDF at:

```text
data/sample.pdf
```

You may use the [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762) for personal study, subject to its terms. Do not commit downloaded or confidential documents.

Run the traditional demo to ingest the PDF and populate ChromaDB:

```powershell
python main.py
```

This demo indexes the document and runs a traditional RAG question. It currently re-indexes the input PDF each time. A persisted ChromaDB index can subsequently be searched without re-extracting the PDF on every question.

### 5. Run the retrieval and agent demonstrations

```powershell
# Traditional PDF RAG
python main.py

# Legacy independent agent checks
python test_router_manual.py
python test_retriever_manual.py

# Tavily retrieval without an LLM
python test_web_retrieval_manual.py

# Current end-to-end orchestration: document + web question
python test_orchestrator_manual.py
```

The end-to-end script asks why Transformers use positional encoding (expected route: `vectorstore`) and asks about Bengaluru weather (expected route: `websearch`). For each, it prints the selected route, status, structured evidence, and the CrewAI-generated summary.

The document branch needs a populated local ChromaDB collection to return document passages. The web branch can run without indexed PDFs, provided Tavily credentials and Ollama are available. Real web-search results change between runs, and a successful response does not guarantee its factual accuracy.

### 6. Run automated tests

```powershell
python -m pytest tests/ -v
```

**Latest observed result:** `29 passed in 21.63s` (November 10, 2026). These are primarily unit tests with mocked LLM, Tavily, and orchestration dependencies. The suite also uses locally generated PDF fixtures; vector tests may download the embedding model on first use. Automated tests do **not** establish that live Ollama or Tavily will always work.

Important test areas include source-field preservation, empty/invalid retrieval results, routing, and recovery from a mocked `openai.InternalServerError` without losing evidence. The separate manual orchestration script exercises live local inference and external Tavily search.

## Example response

For the document question **“Why do Transformers use positional encoding?”**, the orchestrator returns a `vectorstore` route with document evidence and a separate summary. A document evidence item resembles:

```json
{
  "text": "The positional encodings have the same dimension as the embeddings...",
  "source": "sample.pdf",
  "page": 6,
  "chunk_index": 1,
  "document_id": "<content-derived identifier>",
  "distance": 0.26
}
```

This is a schematic example, not a guarantee of a particular ranking or score. The model may cite page 6 in its summary, but citation accuracy and claim grounding are **not yet automatically graded**.

## Privacy and repository hygiene

- **Verify** that `.gitignore` includes `.env`, `.venv/`, `data/`, `chroma_db/`, `.crewai/`, local model/cache files, and other generated artifacts as applicable. The example file paths above are intended to remain local.
- Never commit `TAVILY_API_KEY` or any other API credentials. Keep secrets in local environment variables or a gitignored secret file.
- Treat PDF content and retrieved website snippets as untrusted data. The summarizer tasks instruct the model not to follow embedded instructions, but prompt injection is not fully mitigated.
- CrewAI may offer execution traces hosted by its service. Review trace-sharing/telemetry settings before processing sensitive documents; prompts and evidence can contain private information.
- Ollama inference runs locally, but Tavily searches send queries to a third-party service; installing dependencies and downloading embedding models also involves network access.

## Roadmap

- [x] PDF extraction, chunking, and page metadata
- [x] BGE Small embeddings and persistent local ChromaDB retrieval
- [x] Traditional local Ollama answer generation
- [x] CrewAI Router Agent and original tool-based document retriever
- [x] Real Tavily web-search integration
- [x] Limit returned web snippets to 500 characters
- [x] Separate structured source evidence from LLM-generated summaries on both routes
- [x] Handle recognized Ollama/OpenAI-compatible HTTP 500 errors during summarization while retaining retrieved evidence
- [x] Automated tests for retrieval, routing, evidence preservation, and failure handling (29 passing locally)
- [ ] Add relevance grading and retrieval reranking
- [ ] Add dedicated final-answer generation and hallucination/grounding checks
- [ ] Validate source citations and freshness of live information
- [ ] Extend error handling for Router failures and other provider/network errors
- [ ] Make document replacement failure-safe and resolve filename collisions
- [ ] Improve request logging, evaluation, deployment, and monitoring
- [ ] Optionally support cloud LLM providers such as Anthropic without removing Ollama

## Development workflow

Changes are developed on feature branches, verified with `python -m pytest tests/ -v`, and merged into `main` through pull requests. The structured web/document retrieval and summarization changes described here are being prepared on `feature/web-search`; they should not be treated as merged until the pull request is completed.

## Disclaimer

This is an educational project, not a production-ready knowledge assistant. Retrieved search snippets may be incomplete, outdated, or inaccurate, and generated summaries are not independently fact-checked. Verify important claims using original sources.
