"""
ExperimentRunner
================
Runs all (or selected) scenarios across all (or selected) providers.
Saves raw results to JSON and prints a summary table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import structlog

from providers.base import BaseProvider
from .scenarios import SCENARIOS, ScenarioResult

log = structlog.get_logger()


class ExperimentRunner:
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
            self.scenarios = [s for s in SCENARIOS if s["id"] in scenario_ids]
        else:
            self.scenarios = SCENARIOS

    def run(self) -> list[ScenarioResult]:
        all_results: list[ScenarioResult] = []

        for scenario in self.scenarios:
            log.info("scenario_start", scenario=scenario["id"])
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario['name']}")
            print(f"  {scenario['description']}")
            print(f"{'='*60}")

            for provider in self.providers:
                print(f"  Running on [{provider.name} / {provider.model}] ...", end=" ", flush=True)
                t0 = time.perf_counter()
                try:
                    result = scenario["fn"](provider)
                    elapsed = time.perf_counter() - t0
                    print(f"done ({elapsed:.1f}s)")
                    self._print_result(result)
                    all_results.append(result)
                    log.info(
                        "scenario_done",
                        scenario=scenario["id"],
                        provider=provider.name,
                        success_rate=result.success_rate,
                        consistency_score=result.consistency_score,
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    print(f"FAILED ({elapsed:.1f}s): {exc}")
                    log.error("scenario_error", scenario=scenario["id"],
                              provider=provider.name, error=str(exc))

        self._save(all_results)
        return all_results

    def _print_result(self, r: ScenarioResult) -> None:
        print(f"    success_rate     : {r.success_rate:.2%}")
        print(f"    consistency/score: {r.consistency_score:.3f}")
        print(f"    mean_latency_ms  : {r.mean_latency_ms:.0f} ms")
        print(f"    errors           : {r.error_count}/{r.runs}")
        if r.notes:
            for k, v in r.notes.items():
                if isinstance(v, float):
                    print(f"    {k:<22}: {v:.3f}")
                else:
                    print(f"    {k:<22}: {v}")

    def _save(self, results: list[ScenarioResult]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"experiment_results_{timestamp}.json"

        serialised = []
        for r in results:
            serialised.append({
                "scenario_id": r.scenario_id,
                "provider": r.provider,
                "model": r.model,
                "runs": r.runs,
                "success_rate": r.success_rate,
                "consistency_score": r.consistency_score,
                "mean_latency_ms": r.mean_latency_ms,
                "error_count": r.error_count,
                "notes": r.notes,
            })

        out_path.write_text(json.dumps(serialised, indent=2))
        print(f"\nResults saved → {out_path}")
        log.info("results_saved", path=str(out_path))
