"""
run_experiments.py
==================
Main entrypoint for the cross-provider failure mode study.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=AIza...

    python run_experiments.py
    python run_experiments.py --scenarios consistency tool_reliability
    python run_experiments.py --providers claude openai
"""
from __future__ import annotations

import argparse
import os

from providers import AnthropicProvider, OpenAIProvider, GeminiProvider
from experiments import ExperimentRunner, ResultsReporter


def main():
    parser = argparse.ArgumentParser(description="Cross-provider LLM failure mode study")
    parser.add_argument(
        "--scenarios", nargs="*",
        choices=["consistency", "tool_reliability", "error_recovery",
                 "context_degradation", "instruction_conflict"],
        help="Which scenarios to run (default: all)",
    )
    parser.add_argument(
        "--providers", nargs="*",
        choices=["claude", "openai", "gemini"],
        help="Which providers to run (default: all)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to save results JSON (default: results/)",
    )
    args = parser.parse_args()

    # Build provider list
    all_providers = {
        "claude": lambda: AnthropicProvider(),
        "openai": lambda: OpenAIProvider(),
        "gemini": lambda: GeminiProvider(),
    }

    selected = args.providers or ["claude", "openai", "gemini"]
    providers = []
    for name in selected:
        key_map = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        env_key = key_map[name]
        if not os.environ.get(env_key):
            print(f"WARNING: {env_key} not set — skipping {name}")
            continue
        try:
            providers.append(all_providers[name]())
            print(f"✓ {name} provider initialised")
        except Exception as exc:
            print(f"✗ {name} provider failed to initialise: {exc}")

    if not providers:
        print("No providers available. Set API key environment variables and retry.")
        return

    runner = ExperimentRunner(
        providers=providers,
        output_dir=args.output_dir,
        scenario_ids=args.scenarios,
    )

    print(f"\nRunning {len(runner.scenarios)} scenario(s) across {len(providers)} provider(s)...")
    results = runner.run()

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"  Total scenario×provider runs: {len(results)}")
    print(f"  Results saved in: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
