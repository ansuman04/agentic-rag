"""CrewAI transport hook shared by its sync and async OpenAI clients."""

import httpx
from crewai.llms.hooks.base import BaseInterceptor

from src.resilience import llm_request_timeout


class DeadlineInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    def on_outbound(self, message):
        llm_request_timeout(message)
        return message

    def on_inbound(self, message):
        return message

    async def aon_outbound(self, message):
        return self.on_outbound(message)

    async def aon_inbound(self, message):
        return self.on_inbound(message)
