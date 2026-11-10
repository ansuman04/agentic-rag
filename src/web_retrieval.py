
import json

from src.web_tools import WebSearchTool


class WebRetrievalService:
    """Retrieve structured web evidence without an LLM."""

    def __init__(self, tool: WebSearchTool):
        self.tool = tool

    def retrieve(self, question: str) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Execute Tavily through our existing WebSearchTool.
        raw_result = self.tool._run(
            query=question.strip()
        )

        # Convert JSON into structured Python data.
        evidence = json.loads(raw_result)

        if not isinstance(evidence, list):
            raise ValueError(
                "Web search returned an invalid result format."
            )

        # Validate the essential evidence fields.
        for result in evidence:
            if not isinstance(result, dict):
                raise ValueError(
                    "Each web search result must be an object."
                )

            if not isinstance(result.get("url"), str):
                raise ValueError(
                    "Web search result is missing a valid URL."
                )

            if not result["url"].strip():
                raise ValueError(
                    "Web search result URL cannot be empty."
                )

        return evidence
