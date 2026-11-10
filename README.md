# Agentic RAG — Local PDF and Web Retrieval with CrewAI, ChromaDB, Tavily, and Ollama

An educational and portfolio project that combines **Retrieval-Augmented Generation (RAG)** with **CrewAI-based question routing**. It supports two information sources: locally indexed PDF documents in ChromaDB and public web search through Tavily. Retrieval evidence is preserved separately from relevance-grading decisions and LLM-generated answers, allowing inspection of both accepted and rejected sources.

**Current status (November 10, 2026):** Traditional local PDF RAG works end-to-end. CrewAI routes questions to ChromaDB document retrieval or Tavily web search. A local Qwen2.5 relevance grader evaluates retrieved evidence in a batch, preserving the original results while separating relevant and rejected items. Only relevant evidence is supplied to the document or web summarizer. Recognized grading errors and summarization HTTP 500 errors produce structured failure responses while retaining original evidence. **58 automated tests passed**, as reported from the latest local test run; recent manual executions of both routes succeeded.

> **Scope:** Relevance classification is implemented. Independent claim-by-claim grounding verification, citation auditing, retrieval reranking and controlled self-correction are **not** yet implemented.

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
                            |
               +------------+------------+
               |                         |
          "vectorstore"              "websearch"
               |                         |
               v                         v
   DocumentRetrievalService      WebRetrievalService
               |                         |
          ChromaDB                 Tavily WebSearchTool
               |                         |
               +------------+------------+
                            |
                   Structured evidence
                            |
                  CrewAI Relevance Grader
                      (Qwen2.5)
                            |
                +-----------+-----------+
                |                       |
        Relevant evidence         Rejected evidence
                |
                v
       Document / Web Summarizer
                |
                v
                  Structured response
        question, route, status, evidence,
        relevant_evidence, rejected_evidence,
                 agent_output, error
```

**Responsibilities:**

- The **Router Agent** selects `vectorstore` or `websearch`.
- The retrieval services query ChromaDB or Tavily and return structured evidence without an LLM rewriting it. Tavily returns up to five search results with snippets truncated to 500 characters.
- `LLMRelevanceScorer` runs a tool-free CrewAI grader using local Qwen2.5. The service returns an ordered list of boolean relevance decisions (also normalizing exact boolean strings such as `"true"`/`"false"`). `RelevanceGrader` splits results into accepted and rejected evidence while preserving the original objects.
- Document and web summarizers receive **only accepted evidence**. If nothing is relevant, summarization is skipped.
- `AgentOrchestrator` keeps all original evidence in the response. The orchestrator can be tested using plain callable stubs without live Ollama or Tavily.

### 3. Retrieval-response contract

Example *shape* for a successful web request (illustrative values):

```json
{
  "question": "What is the weather in Bengaluru today?",
  "route": "websearch",
  "status": "retrieved",
  "evidence": [
    {"title": "Source A", "url": "https://example.com/a", "snippet": "Relevant observation"},
    {"title": "Source B", "url": "https://example.com/b", "snippet": "Unrelated results"}
  ],
  "relevant_evidence": [
    {"title": "Source A", "url": "https://example.com/a", "snippet": "Relevant observation"}
  ],
  "rejected_evidence": [
    {"title": "Source B", "url": "https://example.com/b", "snippet": "Unrelated results"}
  ],
  "agent_output": "A concise answer citing Source A.",
  "error": null
}
```

For `vectorstore`, evidence items contain `text`, `source`, `page`, `chunk_index`, `document_id`, and `distance`.

| Status | Meaning |
| --- | --- |
| `retrieved` | Relevant evidence was found, and the summarizer returned output; **not** an independent factual-grounding guarantee. |
| `no_evidence` | Retrieval returned no results; grading and summarization are skipped. |
| `no_relevant_evidence` | Retrieval succeeded, but the grader rejected every item; summarization is skipped. |
| `grading_failed` | Grading returned invalid decisions or a recognized LLM server error. Original evidence is retained; no evidence is classified or summarized. |
| `generation_failed` | Grading succeeded, but a recognized LLM server error prevented summarization. Original and graded evidence are retained. |

The grading recovery catches expected `ValueError` and `openai.InternalServerError` failures, returning a `relevance_grading_error`. Summarization recovery handles `openai.InternalServerError` and returns an `llm_generation_error`. Unexpected exceptions and failures during routing or source retrieval may propagate. `rejected_evidence=[]` on `grading_failed` **does not** mean every source passed or failed grading; grading did not complete.

Evidence is retained in the response, not persisted to a separate evidence database. Web snippets represent tool-returned excerpts capped at 500 characters, **not** the full original web pages.

## Technology stack

| Area | Technology | Purpose |
| --- | --- | --- |
| Runtime | Python 3.12 | Application and tests |
| PDF extraction | PyPDF (`pypdf`) | Read PDF text |
| Embeddings | `BAAI/bge-small-en-v1.5` via Sentence Transformers | 384-dimensional vectors |
| Vector database | ChromaDB | Persistent local similarity search |
| Local inference | Ollama + `qwen2.5:7b` | Traditional RAG and CrewAI agent reasoning |
| Agents and tasks | CrewAI | Routing, batch relevance grading, summarization, and legacy tool-based demos |
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
- Grade retrieved results using a tool-free CrewAI **Evidence Relevance Grader**, separating relevant from rejected items before summarization.
- Run tool-free CrewAI **Document Evidence Summarizer** and **Web Evidence Summarizer** on relevant evidence only.
- Return original `evidence`, `relevant_evidence`, `rejected_evidence`, and `agent_output` as separate fields with a consistent response format on both routes.
- Preserve successfully retrieved evidence when expected grading errors occur, or when an `openai.InternalServerError` occurs during summarization.
- Retain the original `DocumentSearchTool`, tool-based Document Retriever Agent, and related manual tests for learning/reference.
- Run **58 automated tests** (latest user-reported result: November 10, 2026). Manual agent/integration scripts are separate.

### Known limitations

- **No grounding or hallucination verifier yet.** The relevance grader decides whether source snippets appear useful; it does not verify individual claims in the final answer.
- **Web result quality varies.** Tavily may return monthly forecasts and stale pages. The relevance grader can filter irrelevant snippets, but freshness is not independently validated.
- **Retrieval ranking is imperfect.** The lowest-distance ChromaDB chunk is not always the most useful passage.
- **Web evidence is truncated.** The preserved snippet is the tool's 500-character representation, not the full Tavily result or page text.
- **PDF ingestion is not failure-atomic.** Existing chunks are deleted before replacements are written, and different files with the same name can collide.
- **Citations are not independently checked.** Returned titles/URLs/page numbers reflect retrieved metadata, but summaries may still misinterpret them.
- **LLM recovery is limited.** Expected grading `ValueError`/HTTP 500 errors and HTTP 500 summarization errors are handled after retrieval; router failures, other LLM exceptions, malformed source retrieval data, and external API failures may still raise errors.
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
│   ├── relevance_grader.py            # Separate relevant/rejected evidence
│   ├── llm_relevance_scorer.py        # CrewAI grader and JSON decision parsing
│   ├── llm_client.py                  # Ollama client for traditional RAG
│   ├── rag_service.py                 # Traditional RAG answering
│   ├── agents.py                      # Router, relevance grader, summarizers
│   ├── tasks.py                       # CrewAI task instructions
│   ├── tools.py                       # Legacy CrewAI DocumentSearchTool
│   └── orchestrator.py                # Routing, grading, response, error recovery
├── tests/
│   ├── conftest.py                    # Generated PDF fixture
│   ├── test_vector_store.py
│   ├── test_rag_service.py
│   ├── test_document_search_tool.py
│   ├── test_document_retrieval.py
│   ├── test_web_search_tool.py
│   ├── test_web_retrieval.py
│   ├── test_orchestrator.py
│   ├── test_llm_failure_recovery.py
│   ├── test_relevance_grader.py
│   ├── test_llm_relevance_scorer.py
│   ├── test_grading_pipeline.py
│   ├── test_grading_failure_recovery.py
│   └── test_response_contract.py
├── main.py                            # Traditional RAG demo
├── test_router_manual.py              # Legacy manual Router test
├── test_retriever_manual.py           # Legacy tool-based Retriever test
├── test_orchestrator_manual.py        # Both routes, graded evidence + summary
├── test_relevance_manual.py           # Standalone live Qwen2.5 grader
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

> Relevance-grading work is being developed on `feature/relevance-grading`. The previously completed web-search changes have already been merged into `main`. Check out the feature branch to try grading before its own pull request is merged.

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

The end-to-end script asks why Transformers use positional encoding (expected route: `vectorstore`) and asks about Bengaluru weather (expected route: `websearch`). For each, it prints the selected route, status, total retrieved results, relevant/rejected counts, selected evidence, and the CrewAI-generated summary. The standalone `python test_relevance_manual.py` script checks live relevance classification against three sample passages.

The document branch needs a populated local ChromaDB collection to return document passages. The web branch can run without indexed PDFs, provided Tavily credentials and Ollama are available. Real web-search results change between runs, and a successful response does not guarantee its factual accuracy.

### 6. Run automated tests

```powershell
python -m pytest tests/ -v
```

**Latest reported result:** `58 passed` (November 10, 2026). These are primarily unit tests with mocked LLM, Tavily, and orchestration dependencies. The suite also uses locally generated PDF fixtures; vector tests may download the embedding model on first use. Automated tests do **not** establish that live Ollama or Tavily will always work.

Important test areas include source-field preservation, empty/invalid retrieval results, ordered relevance decisions, boolean-string normalization, evidence filtering, routing, five response statuses, and recovery from mocked grading and summarization failures without losing evidence. The separate manual orchestration script exercises live local inference and external Tavily search.

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

This is a schematic example, not a guarantee of a particular ranking or score. The relevance grader may retain this passage; citation accuracy and the grounding of generated claims are **not yet automatically audited**.

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
- [x] CrewAI/Qwen2.5 relevance grading for retrieved PDF and web evidence
- [x] Preserve original, relevant, and rejected evidence separately; skip summarization if no item is relevant
- [x] Handle expected grader failures without losing retrieved evidence
- [x] Automated tests for retrieval, routing, grading, response contracts, and failure handling (58 passing locally)
- [ ] Add retrieval reranking beyond boolean relevance classification
- [ ] Add dedicated final-answer generation and hallucination/grounding checks
- [ ] Validate source citations and freshness of live information
- [ ] Extend error handling for Router failures and other provider/network errors
- [ ] Make document replacement failure-safe and resolve filename collisions
- [ ] Improve request logging, evaluation, deployment, and monitoring
- [ ] Optionally support cloud LLM providers such as Anthropic without removing Ollama

## Development workflow

Changes are developed on feature branches, verified with `python -m pytest tests/ -v`, and merged into `main` through pull requests. The earlier `feature/web-search` pull request has been merged. Relevance grading is currently on `feature/relevance-grading`, pending review and merge.

## Disclaimer

This is an educational project, not a production-ready knowledge assistant. Retrieved search snippets may be incomplete, outdated, or inaccurate. LLM-based relevance decisions can be wrong, and generated summaries are not independently fact-checked. Verify important claims using original sources.
