
import json
import logging
import os

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
            raise RuntimeError(
                "TAVILY_API_KEY is not configured."
            )

        logger.info("Tavily web search started: %s", query)

        try:
            client = TavilyClient(api_key=api_key)

            response = client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_answer=False,
                timeout=20,
            )

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

        except Exception:
            logger.exception(
                "Tavily search failed for query: %s",
                query,
            )
            raise
