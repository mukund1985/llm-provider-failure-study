"""
GeminiProvider
==============
Wraps the Google GenAI SDK (new) for use in cross-provider experiments.
Model: gemini-1.5-flash  (free tier, fast)
"""
from __future__ import annotations

import os

from google import genai
from google.genai import types

from .base import BaseProvider, ProviderResponse


class GeminiProvider(BaseProvider):
    name = "gemini"
    model = "gemini-3.1-flash-lite"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
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
        # Build contents
        contents = prompt
        if system:
            contents = f"{system}\n\n{prompt}"

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        # Build tool declarations if needed
        gemini_tools = None
        if tools:
            fn_decls = []
            for t in tools:
                props = {}
                for k, v in t.get("parameters", {}).get("properties", {}).items():
                    props[k] = types.Schema(
                        type="STRING",
                        description=v.get("description", ""),
                    )
                fn_decls.append(
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t.get("description", ""),
                        parameters=types.Schema(
                            type="OBJECT",
                            properties=props,
                        ),
                    )
                )
            gemini_tools = [types.Tool(function_declarations=fn_decls)]

        try:
            def _call():
                kw = {}
                if gemini_tools:
                    kw["tools"] = gemini_tools
                return self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs, **kw),
                )

            resp, latency = self._timed(_call)

            tool_calls = []
            content_text = ""

            for part in resp.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content_text += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {},
                    })

            usage = resp.usage_metadata
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content=content_text,
                tool_calls=tool_calls,
                latency_ms=latency,
                input_tokens=getattr(usage, "prompt_token_count", 0),
                output_tokens=getattr(usage, "candidates_token_count", 0),
            )

        except Exception as exc:
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(exc),
            )
