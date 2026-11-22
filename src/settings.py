"""Small, validated settings module; no global logging configuration."""

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class Settings:
    tavily_timeout: float = 15.0
    chroma_timeout: float = 10.0
    llm_timeout: float = 60.0
    request_timeout: float = 180.0
    max_active: int = 1

    def __post_init__(self):
        for value in (self.tavily_timeout, self.chroma_timeout, self.llm_timeout, self.request_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be finite positive seconds.")
        if type(self.max_active) is not int or self.max_active < 1:
            raise ValueError("RAG_MAX_ACTIVE_REQUESTS must be a positive integer.")

    @classmethod
    def from_env(cls):
        return cls(
            tavily_timeout=float(os.getenv("TAVILY_TIMEOUT_SECONDS", "15")),
            chroma_timeout=float(os.getenv("CHROMA_TIMEOUT_SECONDS", "10")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            request_timeout=float(os.getenv("RAG_TIMEOUT_SECONDS", "180")),
            max_active=int(os.getenv("RAG_MAX_ACTIVE_REQUESTS", "1")),
        )
