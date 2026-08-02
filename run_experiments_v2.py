"""
run_experiments_v2.py
======================
Main entrypoint for the upgraded MIT-quality cross-provider failure mode study.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=...

    python run_experiments_v2.py
    python run_experiments_v2.py --scenarios consistency tool_reliability
    python run_experiments_v2.py --providers claude openai  # skip gemini
    python run_experiments_v2.py --dry-run  # 1 call per scenario to verify setup

Upgrade over v1:
  - 3 diverse prompts per scenario × N=30 = 90 calls per scenario per provider
  - Per-run latency / success / score arrays (for bootstrap CIs in analysis/)
  - Rate-limited Gemini calls (4s gap → 0% quota failures)
  - Context degradation: 8 lengths × 2 facts = 16 calls (vs. 4 in v1)
  - Instruction conflict: 3 distinct conflict types

Estimated runtime (all scenarios, all 3 providers):
  Claude + OpenAI: ~5 min each (no throttling)
  Gemini: ~30 min (4s/call × ~450 calls)
  Total: ~40 min
"""
from __future__ import annotations

import argparse
import os
import sys

from providers import AnthropicProvider, OpenAIProvider, GeminiProvider
from experiments.runner_v2 import ExperimentRunnerV2


def main():
    parser = argparse.ArgumentParser(
        description="Cross-provider LLM failure mode study v2 (MIT-quality)"
    )
    parser.add_argument(
        "--scenarios", nargs="*",
        choices=["consistency", "tool_reliability", "error_recovery",
                 "context_degradation", "instruction_conflict"],
        help="Which scenarios to run (default: all 5)",
    )
    parser.add_argument(
        "--providers", nargs="*",
        choices=["claude", "openai", "gemini"],
        help="Which providers to run (default: all 3)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to save results JSON (default: results/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 1 call per scenario×provider to verify setup",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — overriding N to 1 per scenario for setup verification")
        # Monkey-patch N in scenarios_v2 before importing runner
        import experiments.scenarios_v2 as sv2
        sv2._DRY_RUN = True

    # ── API key check ──────────────────────────────────────────────────────
    KEY_MAP = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    PROVIDER_CLS = {
        "claude": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }

    selected = args.providers or ["claude", "openai", "gemini"]
    providers = []
    for name in selected:
        env_key = KEY_MAP[name]
        api_key = os.environ.get(env_key)
        if not api_key:
            print(f"  ⚠  {env_key} not set — skipping {name}")
            continue
        try:
            p = PROVIDER_CLS[name]()
            providers.append(p)
            print(f"  ✓  {name} ({p.model}) ready")
        except Exception as exc:
            print(f"  ✗  {name} failed to init: {exc}")

    if not providers:
        print("\nNo providers available. Set API key environment variables and retry.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Providers : {', '.join(p.name for p in providers)}")
    print(f"Scenarios : {args.scenarios or 'all'}")
    print(f"{'='*60}")

    if not args.dry_run:
        # Estimate and warn about Gemini runtime
        if any(p.name == "gemini" for p in providers):
            n_scenarios = len(args.scenarios) if args.scenarios else 5
            est_gemini_calls = n_scenarios * 90  # approx
            est_min = est_gemini_calls * 4 / 60
            print(f"\n  Note: Gemini rate-limiting adds ~{est_min:.0f} min for "
                  f"~{est_gemini_calls} calls.")

    runner = ExperimentRunnerV2(
        providers=providers,
        output_dir=args.output_dir,
        scenario_ids=args.scenarios,
    )

    print(f"\nStarting experiment...")
    results = runner.run()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT V2 COMPLETE")
    print(f"  Scenario×provider runs : {len(results)}")
    total_calls = sum(r.total_runs for r in results)
    total_errors = sum(r.error_count for r in results)
    print(f"  Total API calls        : {total_calls}")
    print(f"  Total errors           : {total_errors} ({total_errors/total_calls:.1%})")
    print(f"  Results saved in       : {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
