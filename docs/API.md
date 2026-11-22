# Agentic RAG HTTP API

This is a **local development interface**, not an internet-facing production service.
The endpoint uses the same CrewAI orchestration, ChromaDB retrieval, Tavily
web search, grounding verification and one-attempt answer correction as the
manual scripts.

## Start the server

From the repository root, with the virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The API documentation is at http://127.0.0.1:8000/docs.
Check liveness at http://127.0.0.1:8000/health.

Ollama must be running with `qwen2.5:7b`. The local ChromaDB collection
must already have been populated for document questions. Configure
`TAVILY_API_KEY` in the server process environment for web searches.
Do not commit the API key.

## Ask a question

```powershell
$body = @{ question = "Why do Transformers use positional encoding?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/questions" -Method Post -ContentType "application/json" -Body $body
```

Successful `verified` results return the grounded answer in `answer` and
the accepted evidence in `sources`. Any other status returns `answer: null`,
even if the LLM produced a draft, so clients never mistake the draft for a
verified answer. The `status` field distinguishes `unverified`,
`no_evidence`, `no_relevant_evidence`, `grading_failed`,
`generation_failed`, `grounding_failed` and `correction_failed`.

The endpoint responds with HTTP 422 for invalid input, HTTP 503 when the
orchestrator raises an unexpected runtime error, and HTTP 200 for completed
pipeline executions (including structured nonverified outcomes). The
`/health` endpoint only reports process liveness, not dependency readiness.

## Tests

```powershell
python -m pytest tests/test_api.py -v
python -m pytest tests/ -v
```

API tests mock the orchestrator, and do not need Ollama or Tavily. Live
end-to-end testing requires those dependencies.

## Deployment notes

There is **no authentication, request throttling, or concurrency control**.
Bind only to localhost. Before deployment, add authentication/authorization,
rate and resource limits, observability, dependency readiness, and secure
request/evidence handling.
