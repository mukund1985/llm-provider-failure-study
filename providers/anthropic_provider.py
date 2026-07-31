"""
AnthropicProvider
=================
Wraps the Anthropic SDK for use in cross-provider experiments.
Model: claude-haiku-4-5  (fast + cheap for bulk experiments)
"""
from __future__ import annotations

import os

import anthropic

from .base import BaseProvider, ProviderResponse


class AnthropicProvider(BaseProvider):
    name = "claude"
    model = "claude-haiku-4-5"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
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
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        if tools:
            # Convert common format (parameters) to Anthropic format (input_schema)
            anthropic_tools = []
            for t in tools:
                converted = {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", t.get("input_schema", {"type": "object", "properties": {}})),
                }
                anthropic_tools.append(converted)
            kwargs["tools"] = anthropic_tools
        if temperature > 0:
            kwargs["temperature"] = temperature

        try:
            def _call():
                return self._client.messages.create(**kwargs)

            msg, latency = self._timed(_call)

            tool_calls = []
            content_text = ""
            for block in msg.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                    })

            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content=content_text,
                tool_calls=tool_calls,
                latency_ms=latency,
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
            )

        except Exception as exc:
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(exc),
            )
