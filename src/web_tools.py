
import json
import logging
import os
import random
import time

import requests
from tavily.errors import TimeoutError as TavilyTimeout
from src.resilience import (
    DependencyUnavailable, current_settings, check_deadline, remaining_budget, stage_timeout,
    log_failure,
)

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient


logger = logging.getLogger(__name__)
MAX_SNIPPET_LENGTH = 500


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Search query for finding information on the web",
        min_length=1,
    )


class WebSearchTool(BaseTool):
    name: str = "web_search"

    description: str = (
        "Search the public web for current information, "
        "news, and information outside the local document "
        "knowledge base. Returns titles, URLs, and snippets."
    )

    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """Search Tavily and return structured JSON evidence."""

        query = query.strip()

        if not query:
            raise ValueError("Search query cannot be empty.")

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise DependencyUnavailable(
                "TAVILY_API_KEY is not configured."
            )

        logger.info("Tavily web search started")

        try:
            client = TavilyClient(api_key=api_key)

            try:
                for attempt in range(2):
                    try:
                        response = client.search(
                            query=query, search_depth="basic", max_results=5,
                            include_answer=False,
                            timeout=stage_timeout(current_settings().tavily_timeout),
                        )
                        check_deadline()
                        break
                    except (TavilyTimeout, requests.exceptions.Timeout,
                            requests.exceptions.HTTPError) as exc:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        transient = isinstance(exc, (TavilyTimeout, requests.exceptions.Timeout)) or status == 503
                        if attempt == 1 or not transient:
                            raise
                        delay = random.uniform(0.1, 0.3)
                        # Reserve a full configured attempt rather than racing the deadline.
                        if remaining_budget() <= delay + current_settings().tavily_timeout:
                            raise
                        time.sleep(delay)
            finally:
                client.close()

            evidence = [
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": (
                        (result.get("content") or "")[:MAX_SNIPPET_LENGTH]
                    ),
                }
                for result in response.get("results", [])
                if result.get("url")
            ]

            logger.warning(
                "[WEB SEARCH TOOL] Returned %d results",
                len(evidence),
            )

            return json.dumps(
                evidence,
                ensure_ascii=False,
            )

        except Exception as exc:
            log_failure(exc, "tavily_search")
            raise
