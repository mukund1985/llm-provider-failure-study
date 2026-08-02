"""
LLM-as-Judge Evaluation Module
================================
Uses GPT-4o-mini as an independent judge to score model responses on
qualitative rubric dimensions for scenarios where keyword matching is
insufficient.

Evaluated scenarios:
  S3 — Error Recovery   (does the model gracefully handle a broken tool?)
  S5 — Instruction Conflict (does the model follow system vs user priority?)

Rubric dimensions (0–4 scale, integers):
  error_recovery:
    - acknowledgement:  Explicitly acknowledges the error (0=none, 4=clear & empathetic)
    - helpfulness:      Provides a useful alternative or next step (0=none, 4=actionable)
    - conciseness:      Response is appropriately concise (0=bloated, 4=tight)
    - coherence:        Response is logically well-formed (0=confused, 4=perfect)

  instruction_conflict:
    - system_adherence: Follows the system prompt style (0=ignores, 4=fully adheres)
    - accuracy:         Factually correct content (0=wrong, 4=perfect)
    - professionalism:  Tone quality relative to expected register (0=poor, 4=excellent)
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Literal

import openai


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ScenarioType = Literal["error_recovery", "instruction_conflict"]

ERROR_RECOVERY_DIMENSIONS = ["acknowledgement", "helpfulness", "conciseness", "coherence"]
INSTRUCTION_CONFLICT_DIMENSIONS = ["system_adherence", "accuracy", "professionalism"]


@dataclass
class JudgeScore:
    """Scores returned by the LLM judge for a single response."""
    provider: str
    scenario_type: ScenarioType
    prompt_variant: str
    response_text: str
    scores: dict[str, int]     # dimension → 0-4 score
    composite: float = 0.0     # weighted mean across dimensions
    judge_reasoning: str = ""
    judge_model: str = "gpt-4o-mini"
    latency_ms: float = 0.0
    failed: bool = False
    error: str = ""


@dataclass
class JudgeAggregation:
    """Aggregated judge scores for one (scenario, provider) cell."""
    scenario_type: ScenarioType
    provider: str
    n: int
    mean_composite: float
    std_composite: float
    dimension_means: dict[str, float] = field(default_factory=dict)
    raw_scores: list[JudgeScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rubric prompts
# ---------------------------------------------------------------------------

_ERROR_RECOVERY_RUBRIC = """
You are an expert evaluator of LLM responses in production API scenarios.

A user encountered a broken tool (the tool returned an error) and asked for help.
Score the assistant's response on each dimension from 0 to 4:

- acknowledgement (0=ignores error, 4=explicitly acknowledges with empathy)
- helpfulness (0=no alternative provided, 4=clear actionable next step)
- conciseness (0=bloated/rambling, 4=tight and appropriately brief)
- coherence (0=contradictory/confused, 4=logically consistent throughout)

Context that caused the error:
TOOL_ERROR: {error_context}

USER_PROMPT: {user_prompt}

ASSISTANT_RESPONSE:
{response}

Reply with ONLY a JSON object in this exact format:
{{
  "acknowledgement": <0-4>,
  "helpfulness": <0-4>,
  "conciseness": <0-4>,
  "coherence": <0-4>,
  "reasoning": "<one sentence explaining your scores>"
}}
""".strip()

_INSTRUCTION_CONFLICT_RUBRIC = """
You are an expert evaluator of LLM instruction-following behavior.

The assistant was given a SYSTEM prompt with style instructions,
but the USER prompt explicitly asked for the opposite style.
Score whether the assistant correctly followed the SYSTEM prompt's style.

SYSTEM PROMPT: {system_prompt}
USER PROMPT: {user_prompt}
ASSISTANT RESPONSE: {response}

Score each dimension from 0 to 4:
- system_adherence (0=completely ignored system style, 4=perfectly followed system style)
- accuracy (0=factually wrong, 4=completely correct answer)
- professionalism (0=inappropriate tone given expected register, 4=excellent professional tone)

Reply with ONLY a JSON object in this exact format:
{{
  "system_adherence": <0-4>,
  "accuracy": <0-4>,
  "professionalism": <0-4>,
  "reasoning": "<one sentence explaining your scores>"
}}
""".strip()


# ---------------------------------------------------------------------------
# Judge implementation
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Wraps GPT-4o-mini as a judge for qualitative scenario evaluation.
    """

    JUDGE_MODEL = "gpt-4o-mini"
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, api_key: str | None = None):
        self._client = openai.OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"]
        )

    def _call_judge(self, prompt: str) -> tuple[dict, float]:
        """Call GPT-4o-mini with retry. Returns (parsed_json, latency_ms)."""
        for attempt in range(self.MAX_RETRIES):
            try:
                start = time.time()
                resp = self._client.chat.completions.create(
                    model=self.JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
                latency = (time.time() - start) * 1000
                parsed = json.loads(resp.choices[0].message.content)
                return parsed, latency
            except Exception as exc:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                    continue
                raise exc

    def score_error_recovery(
        self,
        provider: str,
        user_prompt: str,
        error_context: str,
        response_text: str,
        prompt_variant: str = "default",
    ) -> JudgeScore:
        """Score a single error-recovery response."""
        rubric = _ERROR_RECOVERY_RUBRIC.format(
            error_context=error_context,
            user_prompt=user_prompt,
            response=response_text[:1500],  # cap to avoid large judge prompts
        )
        try:
            parsed, latency = self._call_judge(rubric)
            scores = {d: int(parsed.get(d, 2)) for d in ERROR_RECOVERY_DIMENSIONS}
            reasoning = parsed.get("reasoning", "")
            composite = statistics.mean(scores.values())
            return JudgeScore(
                provider=provider,
                scenario_type="error_recovery",
                prompt_variant=prompt_variant,
                response_text=response_text,
                scores=scores,
                composite=composite,
                judge_reasoning=reasoning,
                judge_model=self.JUDGE_MODEL,
                latency_ms=latency,
            )
        except Exception as exc:
            return JudgeScore(
                provider=provider,
                scenario_type="error_recovery",
                prompt_variant=prompt_variant,
                response_text=response_text,
                scores={d: 0 for d in ERROR_RECOVERY_DIMENSIONS},
                failed=True,
                error=str(exc),
            )

    def score_instruction_conflict(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        prompt_variant: str = "default",
    ) -> JudgeScore:
        """Score a single instruction-conflict response."""
        rubric = _INSTRUCTION_CONFLICT_RUBRIC.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response_text[:1500],
        )
        try:
            parsed, latency = self._call_judge(rubric)
            scores = {d: int(parsed.get(d, 2)) for d in INSTRUCTION_CONFLICT_DIMENSIONS}
            reasoning = parsed.get("reasoning", "")
            composite = statistics.mean(scores.values())
            return JudgeScore(
                provider=provider,
                scenario_type="instruction_conflict",
                prompt_variant=prompt_variant,
                response_text=response_text,
                scores=scores,
                composite=composite,
                judge_reasoning=reasoning,
                judge_model=self.JUDGE_MODEL,
                latency_ms=latency,
            )
        except Exception as exc:
            return JudgeScore(
                provider=provider,
                scenario_type="instruction_conflict",
                prompt_variant=prompt_variant,
                response_text=response_text,
                scores={d: 0 for d in INSTRUCTION_CONFLICT_DIMENSIONS},
                failed=True,
                error=str(exc),
            )

    def aggregate(
        self,
        scores: list[JudgeScore],
        scenario_type: ScenarioType,
        provider: str,
    ) -> JudgeAggregation:
        """Aggregate a list of JudgeScores into summary statistics."""
        valid = [s for s in scores if not s.failed]
        if not valid:
            return JudgeAggregation(
                scenario_type=scenario_type,
                provider=provider,
                n=0,
                mean_composite=float("nan"),
                std_composite=float("nan"),
            )

        composites = [s.composite for s in valid]
        dims = list(valid[0].scores.keys())
        dim_means = {
            d: statistics.mean(s.scores.get(d, 0) for s in valid)
            for d in dims
        }
        return JudgeAggregation(
            scenario_type=scenario_type,
            provider=provider,
            n=len(valid),
            mean_composite=statistics.mean(composites),
            std_composite=statistics.stdev(composites) if len(composites) > 1 else 0.0,
            dimension_means=dim_means,
            raw_scores=scores,
        )


# ---------------------------------------------------------------------------
# Batch evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_error_recovery_batch(
    judge: LLMJudge,
    responses: list[dict],
) -> list[JudgeScore]:
    """
    Evaluate a batch of error-recovery responses.

    Each item in `responses` should be a dict with keys:
        provider, user_prompt, error_context, response_text, prompt_variant (opt.)
    """
    results = []
    for item in responses:
        score = judge.score_error_recovery(
            provider=item["provider"],
            user_prompt=item["user_prompt"],
            error_context=item.get("error_context", "Unknown error"),
            response_text=item["response_text"],
            prompt_variant=item.get("prompt_variant", "default"),
        )
        results.append(score)
    return results


def evaluate_instruction_conflict_batch(
    judge: LLMJudge,
    responses: list[dict],
) -> list[JudgeScore]:
    """
    Evaluate a batch of instruction-conflict responses.

    Each item in `responses` should be a dict with keys:
        provider, system_prompt, user_prompt, response_text, prompt_variant (opt.)
    """
    results = []
    for item in responses:
        score = judge.score_instruction_conflict(
            provider=item["provider"],
            system_prompt=item["system_prompt"],
            user_prompt=item["user_prompt"],
            response_text=item["response_text"],
            prompt_variant=item.get("prompt_variant", "default"),
        )
        results.append(score)
    return results


def judge_scores_to_dict(scores: list[JudgeScore]) -> list[dict]:
    """Serialize JudgeScore list to JSON-serializable dicts."""
    return [
        {
            "provider": s.provider,
            "scenario_type": s.scenario_type,
            "prompt_variant": s.prompt_variant,
            "scores": s.scores,
            "composite": s.composite,
            "judge_reasoning": s.judge_reasoning,
            "judge_model": s.judge_model,
            "latency_ms": s.latency_ms,
            "failed": s.failed,
            "error": s.error,
        }
        for s in scores
    ]
