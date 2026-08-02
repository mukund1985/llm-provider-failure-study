"""
ExperimentRunner v2
====================
Runs all upgraded scenarios (scenarios_v2) with MIT-level statistical rigor.
Saves per-run arrays alongside aggregate metrics for bootstrapping.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from providers.base import BaseProvider
from .scenarios_v2 import SCENARIOS_V2, ScenarioResult


class ExperimentRunnerV2:
    def __init__(
        self,
        providers: Sequence[BaseProvider],
        output_dir: str | Path = "results",
        scenario_ids: list[str] | None = None,
    ):
        self.providers = list(providers)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if scenario_ids:
            self.scenarios = [s for s in SCENARIOS_V2 if s["id"] in scenario_ids]
        else:
            self.scenarios = SCENARIOS_V2

    def run(self) -> list[ScenarioResult]:
        all_results: list[ScenarioResult] = []
        total = len(self.scenarios) * len(self.providers)
        done = 0

        for scenario in self.scenarios:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario['name']}  [{scenario['id']}]")
            print(f"{'='*60}")

            for provider in self.providers:
                done += 1
                print(f"\n  [{done}/{total}] {provider.name} / {provider.model}", flush=True)
                t0 = time.perf_counter()
                try:
                    result = scenario["fn"](provider)
                    elapsed = time.perf_counter() - t0
                    self._print_result(result, elapsed)
                    all_results.append(result)
                except Exception as exc:
                    import traceback
                    elapsed = time.perf_counter() - t0
                    print(f"  FAILED after {elapsed:.1f}s: {exc}")
                    traceback.print_exc()

        self._save(all_results)
        return all_results

    def _print_result(self, r: ScenarioResult, elapsed: float) -> None:
        print(f"    Done in {elapsed:.1f}s")
        print(f"    success_rate  : {r.success_rate:.2%}  ({r.total_runs - r.error_count}/{r.total_runs} calls)")
        print(f"    quality_score : {r.consistency_score:.3f}")
        print(f"    mean_latency  : {r.mean_latency_ms:.0f} ms")
        if r.notes:
            for k, v in sorted(r.notes.items()):
                if isinstance(v, float):
                    print(f"    {k:<24}: {v:.3f}")
                elif isinstance(v, str) and len(v) < 80:
                    print(f"    {k:<24}: {v}")

    def _save(self, results: list[ScenarioResult]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"experiment_v2_results_{timestamp}.json"

        serialised = []
        for r in results:
            serialised.append({
                "schema_version": 2,
                "scenario_id": r.scenario_id,
                "provider": r.provider,
                "model": r.model,
                "total_runs": r.total_runs,
                "success_rate": r.success_rate,
                "consistency_score": r.consistency_score,
                "mean_latency_ms": r.mean_latency_ms,
                "error_count": r.error_count,
                # Per-run arrays — essential for bootstrap CIs
                "per_run_latencies": r.per_run_latencies,
                "per_run_success": r.per_run_success,
                "per_run_scores": r.per_run_scores,
                # Raw response texts — required for LLM-judge evaluation (S3/S5).
                # Ordered identically to the calls (condition blocks of N=30).
                "per_run_texts": [resp.content for resp in r.raw_responses],
                "notes": r.notes,
            })

        out_path.write_text(json.dumps(serialised, indent=2))
        print(f"\nResults saved → {out_path}")
        return out_path
