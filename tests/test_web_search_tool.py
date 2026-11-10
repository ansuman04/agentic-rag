
import json
from unittest.mock import patch

import pytest

from src.web_tools import WebSearchTool


def test_web_search_returns_results():
    mock_response = {
        "results": [
            {
                "title": "Python Documentation",
                "url": "https://docs.python.org/3/",
                "content": "Official Python documentation.",
            },
            {
                "title": "Python Software Foundation",
                "url": "https://www.python.org/",
                "content": "Python programming language.",
            },
        ]
    }

    with patch.dict(
        "os.environ",
        {"TAVILY_API_KEY": "test-key"},
    ):
        with patch("src.web_tools.TavilyClient") as mock_client:
            mock_client.return_value.search.return_value = (
                mock_response
            )

            tool = WebSearchTool()

            output = tool._run(
                query="Python programming language"
            )

            results = json.loads(output)

            assert len(results) == 2

            assert results[0]["title"] == (
                "Python Documentation"
            )
            assert results[0]["url"] == (
                "https://docs.python.org/3/"
            )
            assert results[0]["snippet"] == (
                "Official Python documentation."
            )

            mock_client.return_value.search.assert_called_once_with(
                query="Python programming language",
                search_depth="basic",
                max_results=5,
                include_answer=False,
                timeout=20,
            )


def test_web_search_empty_results():
    with patch.dict(
        "os.environ",
        {"TAVILY_API_KEY": "test-key"},
    ):
        with patch("src.web_tools.TavilyClient") as mock_client:
            mock_client.return_value.search.return_value = {
                "results": []
            }

            tool = WebSearchTool()

            assert json.loads(
                tool._run("unknown topic")
            ) == []


def test_web_search_rejects_empty_query():
    tool = WebSearchTool()

    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        tool._run("   ")


def test_web_search_requires_api_key():
    with patch.dict("os.environ", {}, clear=True):
        tool = WebSearchTool()

        with pytest.raises(
            RuntimeError,
            match="TAVILY_API_KEY",
        ):
            tool._run("Python release")


def test_web_search_limits_snippet_length():
    long_content = "A" * 2000

    mock_response = {
        "results": [
            {
                "title": "Long Search Result",
                "url": "https://example.com/article",
                "content": long_content,
            }
        ]
    }

    with patch.dict(
        "os.environ",
        {"TAVILY_API_KEY": "test-key"},
    ):
        with patch("src.web_tools.TavilyClient") as mock_client:
            mock_client.return_value.search.return_value = (
                mock_response
            )

            tool = WebSearchTool()

            output = tool._run(
                "example search question"
            )

            results = json.loads(output)

            assert len(results) == 1

            assert len(results[0]["snippet"]) == 500

            assert results[0]["snippet"] == (
                long_content[:500]
            )

            assert results[0]["title"] == (
                "Long Search Result"
            )

            assert results[0]["url"] == (
                "https://example.com/article"
            )

