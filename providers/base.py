"""
BaseProvider
============
Abstract interface every cloud provider must implement.
All providers return a ProviderResponse so the experiment
runner can compare results uniformly.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    provider: str          # "claude" | "openai" | "gemini"
    model: str
    content: str           # raw text response
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.error is not None


class BaseProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Send a single completion request and return a ProviderResponse."""

    def complete_n(
        self,
        prompt: str,
        n: int,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
    ) -> list[ProviderResponse]:
        """Send the same prompt n times (for drift/consistency experiments)."""
        results = []
        for _ in range(n):
            results.append(self.complete(prompt, system=system,
                                         temperature=temperature,
                                         max_tokens=max_tokens))
        return results

    def _timed(self, fn):
        """Helper: wrap a call and return (result, elapsed_ms)."""
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed
