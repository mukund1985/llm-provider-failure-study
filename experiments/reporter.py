"""
ResultsReporter
===============
Loads saved experiment results and generates paper-ready summary tables.
"""
from __future__ import annotations

import json
from pathlib import Path


class ResultsReporter:
    def __init__(self, results_path: str | Path):
        self.data = json.loads(Path(results_path).read_text())

    def summary_table(self) -> str:
        """
        Returns a markdown table suitable for copy-paste into the paper.
        Rows = scenarios, columns = providers, cell = key metric.
        """
        scenarios = list({r["scenario_id"] for r in self.data})
        providers = list({r["provider"] for r in self.data})
        scenarios.sort()
        providers.sort()

        # Build lookup
        lookup: dict[tuple, dict] = {}
        for r in self.data:
            lookup[(r["scenario_id"], r["provider"])] = r

        # Header
        col_w = 14
        header = f"{'Scenario':<30}" + "".join(f"{p:>{col_w}}" for p in providers)
        sep = "-" * len(header)
        lines = [sep, header, sep]

        scenario_labels = {
            "consistency": "Response Consistency",
            "tool_reliability": "Tool Call Reliability",
            "error_recovery": "Error Recovery",
            "context_degradation": "Context Degradation",
            "instruction_conflict": "Instruction Conflict",
        }

        for sid in scenarios:
            row = f"{scenario_labels.get(sid, sid):<30}"
            for p in providers:
                r = lookup.get((sid, p))
                if r:
                    val = r["consistency_score"]
                    row += f"{val:>{col_w}.3f}"
                else:
                    row += f"{'N/A':>{col_w}}"
            lines.append(row)

        lines.append(sep)
        lines.append("(Values are primary metric per scenario — see paper for metric definitions)")
        return "\n".join(lines)

    def latency_table(self) -> str:
        """Mean latency per provider per scenario."""
        scenarios = sorted({r["scenario_id"] for r in self.data})
        providers = sorted({r["provider"] for r in self.data})
        lookup = {(r["scenario_id"], r["provider"]): r for r in self.data}

        col_w = 14
        header = f"{'Scenario':<30}" + "".join(f"{p:>{col_w}}" for p in providers)
        sep = "-" * len(header)
        lines = [sep, header + "   (mean latency ms)", sep]

        for sid in scenarios:
            row = f"{sid:<30}"
            for p in providers:
                r = lookup.get((sid, p))
                val = f"{r['mean_latency_ms']:.0f}" if r else "N/A"
                row += f"{val:>{col_w}}"
            lines.append(row)

        lines.append(sep)
        return "\n".join(lines)

    def print_all(self) -> None:
        print("\n=== PRIMARY METRIC TABLE ===")
        print(self.summary_table())
        print("\n=== LATENCY TABLE ===")
        print(self.latency_table())
