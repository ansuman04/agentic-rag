"""Application factory and Uvicorn entry point."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .dependencies import OrchestratorProvider
from .routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic RAG API", version="0.1.0")
    app.state.orchestrator_provider = OrchestratorProvider()
    app.include_router(router)

    @app.exception_handler(Exception)
    async def backend_error(request: Request, exc: Exception):
        logger.exception("RAG request failed: %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "RAG processing failed. Check server logs."},
        )

    return app


app = create_app()
