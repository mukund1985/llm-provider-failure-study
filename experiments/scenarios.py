"""
Experiment Scenarios
====================
Five failure modes tested across all three cloud providers.

Each scenario is a dict with:
  id          - short identifier
  name        - human-readable name (used in paper tables)
  description - what the scenario tests
  runs        - number of API calls per provider
  fn          - callable(provider) -> ScenarioResult
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Callable

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from providers.base import BaseProvider, ProviderResponse

# Shared embedding model — loaded once
_EMBEDDER: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _semantic_similarity(texts: list[str]) -> float:
    """Mean pairwise cosine similarity across a list of responses."""
    if len(texts) < 2:
        return 1.0
    model = _get_embedder()
    vecs = model.encode(texts)
    scores = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = cosine_similarity([vecs[i]], [vecs[j]])[0][0]
            scores.append(float(sim))
    return statistics.mean(scores)


@dataclass
class ScenarioResult:
    scenario_id: str
    provider: str
    model: str
    runs: int
    raw_responses: list[ProviderResponse] = field(default_factory=list)

    # Computed metrics (filled by each scenario fn)
    success_rate: float = 0.0       # fraction of runs that succeeded
    consistency_score: float = 0.0  # semantic similarity across runs (0-1)
    mean_latency_ms: float = 0.0
    error_count: int = 0
    notes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scenario 1 — Response Consistency
# ---------------------------------------------------------------------------
def _run_consistency(provider: BaseProvider) -> ScenarioResult:
    """
    Send the same factual question 20 times at temperature=1.
    Measure how much the answers vary semantically.
    Low score = high drift = bad for production reliability.
    """
    PROMPT = (
        "A user asks: 'What are the three main causes of the 2008 financial crisis?' "
        "Provide a concise answer in 2-3 sentences."
    )
    N = 20
    responses = provider.complete_n(PROMPT, n=N, temperature=1.0, max_tokens=200)

    texts = [r.content for r in responses if not r.failed and r.content.strip()]
    errors = sum(1 for r in responses if r.failed)
    latencies = [r.latency_ms for r in responses if not r.failed]

    consistency = _semantic_similarity(texts) if len(texts) >= 2 else 0.0

    return ScenarioResult(
        scenario_id="consistency",
        provider=provider.name,
        model=provider.model,
        runs=N,
        raw_responses=responses,
        success_rate=(N - errors) / N,
        consistency_score=consistency,
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        error_count=errors,
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Tool Call Reliability
# ---------------------------------------------------------------------------
_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. London",
            },
            "unit": {
                "type": "string",
                "description": "Temperature unit: celsius or fahrenheit",
            },
        },
        "required": ["city"],
    },
}


def _run_tool_reliability(provider: BaseProvider) -> ScenarioResult:
    """
    Ask the model to call a weather tool for London, 20 times.
    Measure: correct tool called, required arg present, no hallucinated args.
    """
    PROMPT = "What is the current weather in London? Use the get_weather tool."
    N = 20

    correct_tool = 0
    correct_args = 0
    hallucinated_args = 0
    errors = 0
    responses = []
    latencies = []

    for _ in range(N):
        r = provider.complete(PROMPT, tools=[_WEATHER_TOOL], max_tokens=256)
        responses.append(r)

        if r.failed:
            errors += 1
            continue

        latencies.append(r.latency_ms)

        if r.tool_calls:
            tc = r.tool_calls[0]
            if tc["name"] == "get_weather":
                correct_tool += 1
                args = tc.get("input", {})
                if "city" in args:
                    correct_args += 1
                # Hallucinated = extra keys beyond city + unit
                allowed = {"city", "unit"}
                extra = set(args.keys()) - allowed
                if extra:
                    hallucinated_args += 1

    success = N - errors
    return ScenarioResult(
        scenario_id="tool_reliability",
        provider=provider.name,
        model=provider.model,
        runs=N,
        raw_responses=responses,
        success_rate=success / N,
        consistency_score=correct_tool / success if success > 0 else 0.0,
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        error_count=errors,
        notes={
            "correct_tool_rate": correct_tool / success if success > 0 else 0.0,
            "required_arg_rate": correct_args / success if success > 0 else 0.0,
            "hallucinated_arg_rate": hallucinated_args / success if success > 0 else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Scenario 3 — Error Recovery
# ---------------------------------------------------------------------------
def _run_error_recovery(provider: BaseProvider) -> ScenarioResult:
    """
    Simulate a broken tool response and ask the model to recover.
    Run 20 times. Score = fraction of runs where model produces a
    coherent fallback rather than repeating the broken call or failing silently.
    """
    SYSTEM = (
        "You are a helpful assistant. If a tool fails, acknowledge the failure "
        "clearly and provide the best answer you can without the tool."
    )
    PROMPT = (
        "Please use the get_weather tool to check the weather in Paris. "
        "The tool returned: ERROR_503: upstream timeout after 30s. "
        "What should I do?"
    )
    N = 20
    responses = provider.complete_n(PROMPT, n=N, system=SYSTEM,
                                    temperature=0.3, max_tokens=200)

    # Recovery = response acknowledges error AND provides a helpful continuation
    # We check for keywords that signal graceful handling
    recovery_keywords = [
        "unavailable", "error", "timeout", "try again", "cannot",
        "unfortunately", "instead", "alternative", "check manually",
        "apologize", "failed", "service", "issue",
    ]
    recovered = 0
    errors = 0
    latencies = []

    for r in responses:
        if r.failed:
            errors += 1
            continue
        latencies.append(r.latency_ms)
        text = r.content.lower()
        if any(kw in text for kw in recovery_keywords):
            recovered += 1

    success = N - errors
    return ScenarioResult(
        scenario_id="error_recovery",
        provider=provider.name,
        model=provider.model,
        runs=N,
        raw_responses=responses,
        success_rate=success / N,
        consistency_score=recovered / success if success > 0 else 0.0,
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        error_count=errors,
        notes={"recovery_rate": recovered / success if success > 0 else 0.0},
    )


# ---------------------------------------------------------------------------
# Scenario 4 — Long Context Degradation
# ---------------------------------------------------------------------------
def _make_long_context(target_words: int) -> str:
    """Generate filler text of approximately target_words words."""
    filler = (
        "The study of artificial intelligence encompasses many subfields "
        "including machine learning, natural language processing, computer vision, "
        "and robotics. Researchers continue to make progress on fundamental "
        "challenges in these areas. "
    )
    words = filler.split()
    reps = (target_words // len(words)) + 1
    return " ".join((words * reps)[:target_words])


def _run_context_degradation(provider: BaseProvider) -> ScenarioResult:
    """
    Plant a specific fact at the start of a growing context, then ask about it.
    Test at 500, 2000, 5000, 8000 word contexts.
    Score = fraction of contexts where the fact is correctly recalled.
    """
    FACT = "The API key rotation policy requires updates every 90 days."
    QUESTION = "According to the document, how often must the API key be rotated?"
    CONTEXT_SIZES = [500, 2000, 5000, 8000]

    correct = 0
    total = len(CONTEXT_SIZES)
    all_responses = []
    latencies = []
    errors = 0
    notes = {}

    for size in CONTEXT_SIZES:
        filler = _make_long_context(size)
        prompt = (
            f"{FACT}\n\n{filler}\n\n"
            f"Question: {QUESTION}\n"
            "Answer in one sentence."
        )
        r = provider.complete(prompt, max_tokens=100, temperature=0.0)
        all_responses.append(r)

        if r.failed:
            errors += 1
            notes[f"ctx_{size}"] = "error"
            continue

        latencies.append(r.latency_ms)
        # Check if answer contains "90"
        if "90" in r.content:
            correct += 1
            notes[f"ctx_{size}"] = "correct"
        else:
            notes[f"ctx_{size}"] = f"wrong: {r.content[:80]}"

    success = total - errors
    return ScenarioResult(
        scenario_id="context_degradation",
        provider=provider.name,
        model=provider.model,
        runs=total,
        raw_responses=all_responses,
        success_rate=success / total,
        consistency_score=correct / total,
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        error_count=errors,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Scenario 5 — Instruction Following Under Conflict
# ---------------------------------------------------------------------------
def _run_instruction_conflict(provider: BaseProvider) -> ScenarioResult:
    """
    System prompt says 'always respond in formal English'.
    User prompt says 'respond in casual slang'.
    Run 20 times. Score = fraction of responses that follow the system prompt.
    A consistent provider always follows system > user for conflicting style.
    """
    SYSTEM = (
        "You are a professional assistant. Always respond in formal, "
        "business-appropriate English. Never use slang or casual language."
    )
    PROMPT = (
        "Hey, just gimme a super chill answer in slang — "
        "what's the capital of France? Like keep it totally casual bro."
    )
    N = 20

    # Slang markers
    slang = ["bro", "dude", "gonna", "wanna", "lol", "ngl", "tbh",
             "yeah", "yep", "nope", "chill", "vibes", "lowkey", "fr"]

    responses = provider.complete_n(PROMPT, n=N, system=SYSTEM,
                                    temperature=0.5, max_tokens=100)

    formal_count = 0
    errors = 0
    latencies = []

    for r in responses:
        if r.failed:
            errors += 1
            continue
        latencies.append(r.latency_ms)
        text = r.content.lower()
        has_slang = any(re.search(r'\b' + re.escape(s) + r'\b', text) for s in slang)
        if not has_slang:
            formal_count += 1

    success = N - errors
    return ScenarioResult(
        scenario_id="instruction_conflict",
        provider=provider.name,
        model=provider.model,
        runs=N,
        raw_responses=responses,
        success_rate=success / N,
        consistency_score=formal_count / success if success > 0 else 0.0,
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        error_count=errors,
        notes={"system_prompt_adherence": formal_count / success if success > 0 else 0.0},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SCENARIOS: list[dict] = [
    {
        "id": "consistency",
        "name": "Response Consistency",
        "description": "Same prompt sent 20× at temp=1. Measures semantic variance.",
        "fn": _run_consistency,
    },
    {
        "id": "tool_reliability",
        "name": "Tool Call Reliability",
        "description": "20× tool-call requests. Measures correct tool + arg fidelity.",
        "fn": _run_tool_reliability,
    },
    {
        "id": "error_recovery",
        "name": "Error Recovery",
        "description": "Broken tool response injected. Measures graceful fallback rate.",
        "fn": _run_error_recovery,
    },
    {
        "id": "context_degradation",
        "name": "Long Context Degradation",
        "description": "Fact planted at context start, recalled at 4 context lengths.",
        "fn": _run_context_degradation,
    },
    {
        "id": "instruction_conflict",
        "name": "Instruction Following Under Conflict",
        "description": "System vs user style conflict. Measures system-prompt adherence.",
        "fn": _run_instruction_conflict,
    },
]
