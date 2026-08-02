"""
run_judge.py
=============
LLM-as-judge evaluation over the S3/S5 re-run results (with response texts).

Maps each response (index // 30) back to its condition in scenarios_v2,
scores it with GPT-4o-mini via analysis/llm_judge.py, and saves:
  results_v2_judge/judge_scores_<timestamp>.json  (raw per-response scores)
  plus prints aggregated tables with bootstrap CIs.

Usage:
    export OPENAI_API_KEY=...
    python run_judge.py results_v2_judge/experiment_v2_results_20260802_090939.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from analysis.llm_judge import (
    LLMJudge, JudgeScore, judge_scores_to_dict,
    ERROR_RECOVERY_DIMENSIONS, INSTRUCTION_CONFLICT_DIMENSIONS,
)
from analysis.stats import bootstrap_ci
from experiments.scenarios_v2 import _ERROR_CONDITIONS, _CONFLICT_CONDITIONS

N_PER_CONDITION = 30
MAX_WORKERS = 8


def main():
    results_path = sys.argv[1]
    records = json.load(open(results_path))
    judge = LLMJudge()

    jobs = []  # (kind, provider, variant_label, system, user, error_ctx, text)
    for rec in records:
        provider = rec["provider"]
        texts = rec["per_run_texts"]
        if rec["scenario_id"] == "error_recovery":
            for i, text in enumerate(texts):
                system, prompt, _kw = _ERROR_CONDITIONS[i // N_PER_CONDITION]
                # error context is embedded in the prompt; extract the ERROR_ line
                err = [ln for ln in prompt.split(". ") if "ERROR_" in ln]
                jobs.append(("error_recovery", provider, f"condition_{i // N_PER_CONDITION}",
                             system, prompt, err[0] if err else "tool error", text))
        elif rec["scenario_id"] == "instruction_conflict":
            for i, text in enumerate(texts):
                system, prompt, _m, label, _sc = _CONFLICT_CONDITIONS[i // N_PER_CONDITION]
                jobs.append(("instruction_conflict", provider, label,
                             system, prompt, "", text))

    print(f"Total judge calls: {len(jobs)}")
    t0 = time.time()

    def score_one(job) -> JudgeScore:
        kind, provider, variant, system, user, err_ctx, text = job
        if kind == "error_recovery":
            return judge.score_error_recovery(
                provider=provider, user_prompt=user, error_context=err_ctx,
                response_text=text, prompt_variant=variant)
        return judge.score_instruction_conflict(
            provider=provider, system_prompt=system, user_prompt=user,
            response_text=text, prompt_variant=variant)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        scores = list(ex.map(score_one, jobs))

    elapsed = time.time() - t0
    failed = sum(1 for s in scores if s.failed)
    print(f"Done in {elapsed:.0f}s — {failed} failed judge calls")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"results_v2_judge/judge_scores_{ts}.json"
    with open(out, "w") as f:
        json.dump(judge_scores_to_dict(scores), f, indent=2)
    print(f"Saved → {out}")

    # ---- Aggregate ----
    for kind, dims in (("error_recovery", ERROR_RECOVERY_DIMENSIONS),
                       ("instruction_conflict", INSTRUCTION_CONFLICT_DIMENSIONS)):
        print(f"\n{'='*70}\n{kind}\n{'='*70}")
        for provider in ("claude", "openai", "gemini"):
            sel = [s for s in scores if s.scenario_type == kind
                   and s.provider == provider and not s.failed]
            if not sel:
                continue
            comps = [s.composite for s in sel]
            ci = bootstrap_ci(comps)
            dim_means = {d: statistics.mean(s.scores[d] for s in sel) for d in dims}
            dims_str = "  ".join(f"{d}={m:.2f}" for d, m in dim_means.items())
            print(f"  {provider:<8} composite={statistics.mean(comps):.3f} "
                  f"[{ci.lower:.3f},{ci.upper:.3f}] n={len(sel)}")
            print(f"           {dims_str}")
            # per-variant composites
            variants = sorted(set(s.prompt_variant for s in sel))
            for v in variants:
                vsel = [s.composite for s in sel if s.prompt_variant == v]
                print(f"           {v:<22} {statistics.mean(vsel):.3f}")


if __name__ == "__main__":
    main()
