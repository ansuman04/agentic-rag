
from ollama import Client


class LocalLLM:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(
            host=host,
            timeout=120.0,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical AI assistant. "
                        "Answer clearly and accurately."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        return response.message.content or ""