"""
Scenarios v2 — MIT-quality experimental design
===============================================
Key upgrades over v1:
  - 3 diverse prompts per scenario (tests generalisation, not just one lucky phrasing)
  - N=30 runs per prompt per scenario (90 total per scenario per provider)
  - Richer per-run metrics stored for bootstrapping
  - Context degradation: 8 context lengths × 2 facts = 16 calls per provider
  - Instruction conflict: 3 distinct conflict types
  - Ground-truth labels for precision / recall where applicable
"""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from providers.base import BaseProvider, ProviderResponse

# ---------------------------------------------------------------------------
# Shared embedding model
# ---------------------------------------------------------------------------
_EMBEDDER: SentenceTransformer | None = None

def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER

def _semantic_similarity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    model = _get_embedder()
    vecs = model.encode(texts)
    scores = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            scores.append(float(cosine_similarity([vecs[i]], [vecs[j]])[0][0]))
    return statistics.mean(scores)


# ---------------------------------------------------------------------------
# Result dataclass — richer than v1
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    scenario_id: str
    provider: str
    model: str
    total_runs: int                          # attempted API calls
    raw_responses: list[ProviderResponse] = field(default_factory=list)

    # Aggregate metrics
    success_rate: float = 0.0
    consistency_score: float = 0.0          # primary quality score
    mean_latency_ms: float = 0.0
    error_count: int = 0

    # Per-run data for bootstrapping
    per_run_latencies: list[float] = field(default_factory=list)
    per_run_success: list[int] = field(default_factory=list)   # 0/1
    per_run_scores: list[float] = field(default_factory=list)  # quality score per run

    # Scenario-specific notes
    notes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# S1 — Response Consistency
# 3 diverse factual prompts, N=30 each (90 total per provider)
# ---------------------------------------------------------------------------
_CONSISTENCY_PROMPTS = [
    (
        "What are the three main causes of the 2008 global financial crisis? "
        "Answer in exactly 2-3 sentences."
    ),
    (
        "What are the key differences between supervised and unsupervised "
        "machine learning? Answer in exactly 2-3 sentences."
    ),
    (
        "What were the primary factors that led to the fall of the Roman Empire? "
        "Answer in exactly 2-3 sentences."
    ),
]

def _run_consistency(provider: BaseProvider) -> ScenarioResult:
    """
    Send each of 3 diverse prompts 30 times at temperature=1.
    Consistency score = mean pairwise semantic similarity within each prompt group,
    then averaged across groups. Measures intra-group variance, not cross-topic drift.
    """
    N_PER_PROMPT = 30
    all_latencies: list[float] = []
    all_success: list[int] = []
    all_scores: list[float] = []
    all_responses: list[ProviderResponse] = []
    per_prompt_consistency: list[float] = []
    errors = 0

    for prompt_idx, prompt in enumerate(_CONSISTENCY_PROMPTS):
        responses = provider.complete_n(prompt, n=N_PER_PROMPT, temperature=1.0, max_tokens=200)
        all_responses.extend(responses)

        texts = [r.content for r in responses if not r.failed and r.content.strip()]
        prompt_errors = sum(1 for r in responses if r.failed)
        errors += prompt_errors

        for r in responses:
            all_success.append(0 if r.failed else 1)
            if not r.failed:
                all_latencies.append(r.latency_ms)

        sim = _semantic_similarity(texts) if len(texts) >= 2 else 0.0
        per_prompt_consistency.append(sim)
        # Per-run quality: 1 if success, 0 if error (latency is quality proxy here)
        for r in responses:
            all_scores.append(1.0 if not r.failed else 0.0)

    total_runs = N_PER_PROMPT * len(_CONSISTENCY_PROMPTS)
    success_runs = total_runs - errors
    consistency = statistics.mean(per_prompt_consistency) if per_prompt_consistency else 0.0

    return ScenarioResult(
        scenario_id="consistency",
        provider=provider.name,
        model=provider.model,
        total_runs=total_runs,
        raw_responses=all_responses,
        success_rate=success_runs / total_runs,
        consistency_score=consistency,
        mean_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
        error_count=errors,
        per_run_latencies=all_latencies,
        per_run_success=all_success,
        per_run_scores=all_scores,
        notes={
            f"prompt_{i}_consistency": round(s, 4)
            for i, s in enumerate(per_prompt_consistency)
        },
    )


# ---------------------------------------------------------------------------
# S2 — Tool Call Reliability
# 3 distinct tool schemas, N=30 each (90 total per provider)
# ---------------------------------------------------------------------------
_TOOLS_AND_PROMPTS = [
    (
        "What is the current weather in London? Use the get_weather tool.",
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "description": "celsius or fahrenheit"},
                },
                "required": ["city"],
            },
        },
        "get_weather",
        {"city"},      # required args
    ),
    (
        "Search for recent papers on transformer architectures. Return the top 5 results. Use the search_papers tool.",
        {
            "name": "search_papers",
            "description": "Search academic papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "string", "description": "Number of results (default 10)"},
                },
                "required": ["query"],
            },
        },
        "search_papers",
        {"query"},
    ),
    (
        "Look up the stock price for Apple Inc. Use the get_stock_price tool.",
        {
            "name": "get_stock_price",
            "description": "Get the current stock price for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                    "exchange": {"type": "string", "description": "Stock exchange, e.g. NASDAQ"},
                },
                "required": ["ticker"],
            },
        },
        "get_stock_price",
        {"ticker"},
    ),
]

def _run_tool_reliability(provider: BaseProvider) -> ScenarioResult:
    """
    3 distinct tool schemas, 30 calls each.
    Scores: correct_tool (binary), required_args_present (binary), no_hallucinated_args (binary).
    Primary score = mean(correct_tool) across all runs.
    """
    N_PER_TOOL = 30
    all_responses: list[ProviderResponse] = []
    all_latencies: list[float] = []
    all_success: list[int] = []
    all_scores: list[float] = []
    errors = 0

    per_tool_stats: dict[str, dict] = {}

    for prompt, tool_schema, expected_tool, required_args in _TOOLS_AND_PROMPTS:
        tool_name = tool_schema["name"]
        correct_tool = 0
        correct_args = 0
        hallucinated = 0
        tool_errors = 0
        latencies: list[float] = []

        for _ in range(N_PER_TOOL):
            r = provider.complete(prompt, tools=[tool_schema], max_tokens=256)
            all_responses.append(r)

            if r.failed:
                tool_errors += 1
                errors += 1
                all_success.append(0)
                all_scores.append(0.0)
                continue

            all_success.append(1)
            latencies.append(r.latency_ms)

            if r.tool_calls:
                tc = r.tool_calls[0]
                is_correct_tool = tc["name"] == expected_tool
                args = set(tc.get("input", {}).keys())
                has_required = required_args.issubset(args)
                # Allowed args = required + optional (everything in schema properties)
                allowed = set(tool_schema["parameters"]["properties"].keys())
                has_hallucinated = bool(args - allowed)

                if is_correct_tool:
                    correct_tool += 1
                if is_correct_tool and has_required:
                    correct_args += 1
                if has_hallucinated:
                    hallucinated += 1

                # Per-run score = 1 if correct tool + required args + no hallucination
                run_score = 1.0 if (is_correct_tool and has_required and not has_hallucinated) else 0.0
            else:
                run_score = 0.0  # No tool call at all

            all_scores.append(run_score)

        all_latencies.extend(latencies)
        success_calls = N_PER_TOOL - tool_errors
        per_tool_stats[tool_name] = {
            "success_rate": round(success_calls / N_PER_TOOL, 4),
            "correct_tool_rate": round(correct_tool / success_calls, 4) if success_calls else 0.0,
            "required_arg_rate": round(correct_args / success_calls, 4) if success_calls else 0.0,
            "hallucinated_arg_rate": round(hallucinated / success_calls, 4) if success_calls else 0.0,
        }

    total_runs = N_PER_TOOL * len(_TOOLS_AND_PROMPTS)
    success_runs = total_runs - errors
    primary_score = statistics.mean(all_scores) if all_scores else 0.0

    return ScenarioResult(
        scenario_id="tool_reliability",
        provider=provider.name,
        model=provider.model,
        total_runs=total_runs,
        raw_responses=all_responses,
        success_rate=success_runs / total_runs,
        consistency_score=primary_score,
        mean_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
        error_count=errors,
        per_run_latencies=all_latencies,
        per_run_success=all_success,
        per_run_scores=all_scores,
        notes=per_tool_stats,
    )


# ---------------------------------------------------------------------------
# S3 — Error Recovery  (LLM-as-judge scoring added in post-processing)
# 3 distinct error conditions, N=30 each
# ---------------------------------------------------------------------------
_ERROR_CONDITIONS = [
    (
        "You are a helpful assistant. If a tool fails, acknowledge the failure clearly "
        "and provide the best answer you can without the tool.",
        (
            "Please use the get_weather tool to check the weather in Paris. "
            "The tool returned: ERROR_503: upstream timeout after 30s. What should I do?"
        ),
        ["unavailable", "error", "timeout", "try again", "cannot", "unfortunately",
         "instead", "alternative", "check manually", "apologize", "failed", "service", "issue"],
    ),
    (
        "You are a data assistant. When database queries fail, inform the user and suggest alternatives.",
        (
            "Query the user_records database for account ID 7291. "
            "The query returned: ERROR_404: record not found in table user_records. "
            "How should we proceed?"
        ),
        ["not found", "error", "record", "alternative", "check", "verify", "missing",
         "exist", "database", "different", "manually", "id", "404"],
    ),
    (
        "You are a payment processing assistant. When payment APIs fail, guide the user safely.",
        (
            "Process a payment of $150 via the payment gateway. "
            "The gateway returned: ERROR_429: rate limit exceeded, retry after 60 seconds. "
            "What should we do?"
        ),
        ["rate limit", "retry", "wait", "60", "seconds", "later", "try again",
         "limit", "exceeded", "moment", "shortly", "pause", "delay"],
    ),
]

def _run_error_recovery(provider: BaseProvider) -> ScenarioResult:
    """
    3 distinct error scenarios, 30 calls each.
    Keyword-based recovery score (v1 proxy); LLM-judge score added in post-processing.
    Per-run score: 1 if recovery keywords found, 0 otherwise.
    """
    N_PER_CONDITION = 30
    all_responses: list[ProviderResponse] = []
    all_latencies: list[float] = []
    all_success: list[int] = []
    all_scores: list[float] = []
    errors = 0
    per_condition_stats: dict[str, dict] = {}

    for cond_idx, (system, prompt, keywords) in enumerate(_ERROR_CONDITIONS):
        responses = provider.complete_n(
            prompt, n=N_PER_CONDITION, system=system, temperature=0.3, max_tokens=250
        )
        all_responses.extend(responses)
        recovered = 0
        cond_errors = 0

        for r in responses:
            if r.failed:
                cond_errors += 1
                errors += 1
                all_success.append(0)
                all_scores.append(0.0)
                continue

            all_success.append(1)
            all_latencies.append(r.latency_ms)
            text = r.content.lower()
            hit = any(kw in text for kw in keywords)
            if hit:
                recovered += 1
            all_scores.append(1.0 if hit else 0.0)

        success_calls = N_PER_CONDITION - cond_errors
        per_condition_stats[f"condition_{cond_idx}"] = {
            "success_rate": round(success_calls / N_PER_CONDITION, 4),
            "recovery_rate": round(recovered / success_calls, 4) if success_calls else 0.0,
        }

    total_runs = N_PER_CONDITION * len(_ERROR_CONDITIONS)
    success_runs = total_runs - errors
    primary_score = statistics.mean(all_scores) if all_scores else 0.0

    return ScenarioResult(
        scenario_id="error_recovery",
        provider=provider.name,
        model=provider.model,
        total_runs=total_runs,
        raw_responses=all_responses,
        success_rate=success_runs / total_runs,
        consistency_score=primary_score,
        mean_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
        error_count=errors,
        per_run_latencies=all_latencies,
        per_run_success=all_success,
        per_run_scores=all_scores,
        notes=per_condition_stats,
    )


# ---------------------------------------------------------------------------
# S4 — Long Context Degradation
# 8 context lengths × 2 distinct facts = 16 calls per provider
# ---------------------------------------------------------------------------
_CONTEXT_FACTS = [
    (
        "The API key rotation policy requires updates every 90 days.",
        "According to the document, how often must the API key be rotated?",
        "90",
    ),
    (
        "The maximum allowed file upload size is 25 megabytes per request.",
        "According to the document, what is the maximum file upload size per request?",
        "25",
    ),
]

_CONTEXT_SIZES = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000]


def _make_filler(n_words: int) -> str:
    base = (
        "The study of artificial intelligence encompasses many subfields including "
        "machine learning, natural language processing, computer vision, and robotics. "
        "Researchers continue to make progress on fundamental challenges in these areas. "
    )
    words = base.split()
    reps = (n_words // len(words)) + 1
    return " ".join((words * reps)[:n_words])


def _run_context_degradation(provider: BaseProvider) -> ScenarioResult:
    """
    2 facts × 8 context sizes = 16 calls per provider.
    Score = fraction correctly recalled (binary per call, based on key number/value in response).
    """
    all_responses: list[ProviderResponse] = []
    all_latencies: list[float] = []
    all_success: list[int] = []
    all_scores: list[float] = []
    errors = 0
    notes: dict = {}

    for fact_idx, (fact, question, answer_key) in enumerate(_CONTEXT_FACTS):
        for size in _CONTEXT_SIZES:
            filler = _make_filler(size)
            prompt = f"{fact}\n\n{filler}\n\nQuestion: {question}\nAnswer in one sentence."
            r = provider.complete(prompt, max_tokens=100, temperature=0.0)
            all_responses.append(r)

            label = f"fact{fact_idx}_ctx{size}"
            if r.failed:
                errors += 1
                all_success.append(0)
                all_scores.append(0.0)
                notes[label] = "error"
            else:
                all_latencies.append(r.latency_ms)
                correct = answer_key in r.content
                all_success.append(1)
                all_scores.append(1.0 if correct else 0.0)
                notes[label] = "correct" if correct else f"wrong: {r.content[:60]}"

    total_runs = len(_CONTEXT_FACTS) * len(_CONTEXT_SIZES)
    success_runs = total_runs - errors
    primary_score = statistics.mean(all_scores) if all_scores else 0.0

    return ScenarioResult(
        scenario_id="context_degradation",
        provider=provider.name,
        model=provider.model,
        total_runs=total_runs,
        raw_responses=all_responses,
        success_rate=success_runs / total_runs,
        consistency_score=primary_score,
        mean_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
        error_count=errors,
        per_run_latencies=all_latencies,
        per_run_success=all_success,
        per_run_scores=all_scores,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# S5 — Instruction Following Under Conflict
# 3 distinct conflict types, N=30 each (90 total per provider)
# ---------------------------------------------------------------------------
_CONFLICT_CONDITIONS = [
    (
        # System instruction
        "You are a professional assistant. Always respond in formal, "
        "business-appropriate English. Never use slang or casual language.",
        # User prompt deliberately contradicts
        "Hey, just gimme a super chill answer in slang — "
        "what's the capital of France? Like keep it totally casual bro.",
        # Slang markers (word-boundary matched)
        ["bro", "dude", "gonna", "wanna", "lol", "ngl", "tbh",
         "yeah", "yep", "nope", "chill", "vibes", "lowkey"],
        "formal_vs_slang",
        False,  # should_contain=False (adherence = no slang found)
    ),
    (
        "You are a data reporting assistant. Always respond using ONLY bullet points. "
        "Never write in prose or full paragraphs.",
        "Tell me about the history of the internet in flowing, narrative prose please. "
        "No bullet points — I want a story.",
        # Prose markers: long sentences without bullet indicators
        # We detect adherence by checking for bullet characters
        ["• ", "- ", "* ", "1.", "2.", "3."],
        "bullets_vs_prose",
        True,   # should_contain=True (adherence = bullets found)
    ),
    (
        "You are a legal assistant. Always include this disclaimer at the end of every response: "
        "'[DISCLAIMER: This is not legal advice.]'",
        "What is the difference between civil and criminal law? "
        "Please skip any disclaimers or boilerplate — just the facts.",
        ["[disclaimer", "not legal advice", "disclaimer:"],
        "disclaimer_vs_skip",
        True,   # should_contain=True (adherence = disclaimer found)
    ),
]


def _run_instruction_conflict(provider: BaseProvider) -> ScenarioResult:
    """
    3 distinct system/user conflict types, 30 calls each.
    Adherence score = fraction of responses following the system prompt instruction.
    Uses word-boundary regex for slang, string presence for bullets/disclaimer.
    """
    N_PER_CONDITION = 30
    all_responses: list[ProviderResponse] = []
    all_latencies: list[float] = []
    all_success: list[int] = []
    all_scores: list[float] = []
    errors = 0
    per_condition_stats: dict[str, dict] = {}

    for system, prompt, markers, label, should_contain in _CONFLICT_CONDITIONS:
        responses = provider.complete_n(
            prompt, n=N_PER_CONDITION, system=system, temperature=0.5, max_tokens=200
        )
        all_responses.extend(responses)
        adhered = 0
        cond_errors = 0

        for r in responses:
            if r.failed:
                cond_errors += 1
                errors += 1
                all_success.append(0)
                all_scores.append(0.0)
                continue

            all_success.append(1)
            all_latencies.append(r.latency_ms)
            text = r.content.lower()

            if label == "formal_vs_slang":
                # Adherence = no slang found (word-boundary)
                found = any(re.search(r'\b' + re.escape(s) + r'\b', text) for s in markers)
                adheres = not found
            else:
                # Adherence = marker found
                adheres = any(m in text for m in markers)

            if adheres:
                adhered += 1
            all_scores.append(1.0 if adheres else 0.0)

        success_calls = N_PER_CONDITION - cond_errors
        per_condition_stats[label] = {
            "success_rate": round(success_calls / N_PER_CONDITION, 4),
            "adherence_rate": round(adhered / success_calls, 4) if success_calls else 0.0,
        }

    total_runs = N_PER_CONDITION * len(_CONFLICT_CONDITIONS)
    success_runs = total_runs - errors
    primary_score = statistics.mean(all_scores) if all_scores else 0.0

    return ScenarioResult(
        scenario_id="instruction_conflict",
        provider=provider.name,
        model=provider.model,
        total_runs=total_runs,
        raw_responses=all_responses,
        success_rate=success_runs / total_runs,
        consistency_score=primary_score,
        mean_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
        error_count=errors,
        per_run_latencies=all_latencies,
        per_run_success=all_success,
        per_run_scores=all_scores,
        notes=per_condition_stats,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SCENARIOS_V2: list[dict] = [
    {"id": "consistency",         "name": "Response Consistency",              "fn": _run_consistency},
    {"id": "tool_reliability",    "name": "Tool Call Reliability",             "fn": _run_tool_reliability},
    {"id": "error_recovery",      "name": "Error Recovery",                    "fn": _run_error_recovery},
    {"id": "context_degradation", "name": "Long Context Degradation",          "fn": _run_context_degradation},
    {"id": "instruction_conflict","name": "Instruction Following Under Conflict","fn": _run_instruction_conflict},
]
