"""
OpenAIProvider
==============
Wraps the OpenAI SDK for use in cross-provider experiments.
Model: gpt-4o-mini  (fast + cheap for bulk experiments)
"""
from __future__ import annotations

import os
import json

from openai import OpenAI

from .base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    name = "openai"
    model = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"]
        )
        if model:
            self.model = model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            # OpenAI tool format: {"type": "function", "function": {...}}
            kwargs["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        try:
            def _call():
                return self._client.chat.completions.create(**kwargs)

            resp, latency = self._timed(_call)
            choice = resp.choices[0]
            msg = choice.message

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments or "{}"),
                    })

            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content=msg.content or "",
                tool_calls=tool_calls,
                latency_ms=latency,
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            )

        except Exception as exc:
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(exc),
            )
