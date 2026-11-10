
import json
from unittest.mock import Mock

import pytest

from src.web_retrieval import WebRetrievalService


def test_retrieval_preserves_structured_evidence():
    original_evidence = [
        {
            "title": "Weather Report",
            "url": "https://example.com/weather",
            "snippet": "Temperature is 24°C.",
        },
        {
            "title": "Second Weather Report",
            "url": "https://example.org/weather",
            "snippet": "Cloudy skies.",
        },
    ]

    mock_tool = Mock()
    mock_tool._run.return_value = json.dumps(
        original_evidence
    )

    service = WebRetrievalService(tool=mock_tool)

    result = service.retrieve(
        "What is the weather in Bengaluru?"
    )

    assert result == original_evidence
    assert isinstance(result, list)
    assert len(result) == 2

    mock_tool._run.assert_called_once_with(
        query="What is the weather in Bengaluru?"
    )


def test_retrieval_rejects_empty_question():
    mock_tool = Mock()
    service = WebRetrievalService(tool=mock_tool)

    with pytest.raises(ValueError):
        service.retrieve("   ")

    mock_tool._run.assert_not_called()


def test_retrieval_rejects_invalid_json_format():
    mock_tool = Mock()
    mock_tool._run.return_value = json.dumps(
        {"unexpected": "object"}
    )

    service = WebRetrievalService(tool=mock_tool)

    with pytest.raises(
        ValueError,
        match="invalid result format",
    ):
        service.retrieve("Test question")


def test_retrieval_rejects_missing_url():
    mock_tool = Mock()
    mock_tool._run.return_value = json.dumps(
        [
            {
                "title": "Invalid Result",
                "snippet": "Missing URL",
            }
        ]
    )

    service = WebRetrievalService(tool=mock_tool)

    with pytest.raises(
        ValueError,
        match="missing a valid URL",
    ):
        service.retrieve("Test question")
