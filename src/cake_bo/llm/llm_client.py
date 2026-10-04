"""LLM client interfaces and the OpenAI adapter."""

from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()


class LLMClient(Protocol):
    """Interface required by the kernel-evolution engine."""

    def generate(self, message: str, system_prompt: str) -> str:
        """Generate a response for ``message`` under ``system_prompt``."""


class OpenAIClient:
    """Small wrapper around the OpenAI chat-completions API."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.5,
        top_p: float = 1.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or inject an LLMClient into CAKE."
            )
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, message: str, system_prompt: str) -> str:
        """Return the model's text response."""
        if not message.strip():
            raise ValueError("message must not be empty")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )
        except OpenAIError as error:
            raise RuntimeError(f"OpenAI request failed: {error}") from error
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The OpenAI API returned an empty response")
        return content


__all__ = ["LLMClient", "OpenAIClient"]
