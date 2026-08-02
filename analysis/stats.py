"""
Statistical Analysis Module
============================
Bootstrap confidence intervals, hypothesis tests, and effect sizes
for cross-provider LLM reliability experiments.

Usage:
    from analysis.stats import bootstrap_ci, pairwise_tests, summarize_stats
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """95% bootstrap confidence interval for a scalar metric."""
    estimate: float        # point estimate on original data
    lower: float           # 2.5th percentile of bootstrap distribution
    upper: float           # 97.5th percentile of bootstrap distribution
    n_bootstrap: int = 10_000

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.lower:.3f}, {self.upper:.3f}]"

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass
class HypothesisTest:
    """Result of a two-sample hypothesis test."""
    test_name: str
    provider_a: str
    provider_b: str
    metric: str
    statistic: float
    p_value: float
    significant: bool          # after Bonferroni correction
    effect_size: float = 0.0   # Cohen's d or odds ratio depending on test
    effect_label: str = ""     # "small" / "medium" / "large"


@dataclass
class ScenarioStats:
    """Full statistical summary for one (scenario, provider) cell."""
    scenario_id: str
    provider: str
    n: int
    success_rate_ci: BootstrapCI
    score_ci: BootstrapCI
    latency_ci: BootstrapCI
    raw_successes: list[float] = field(default_factory=list)
    raw_scores: list[float] = field(default_factory=list)
    raw_latencies: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: list[float],
    stat_fn: Callable[[list[float]], float] = statistics.mean,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """
    Compute a bootstrap confidence interval for any scalar statistic.

    Args:
        data:        Raw observations (must have len >= 1).
        stat_fn:     Function mapping list → scalar (default: mean).
        n_bootstrap: Number of bootstrap resamples.
        confidence:  Desired confidence level (default 0.95 → 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        BootstrapCI with point estimate and [lower, upper] bounds.
    """
    if not data:
        return BootstrapCI(estimate=float("nan"), lower=float("nan"), upper=float("nan"), n_bootstrap=0)

    rng = random.Random(seed)
    point = stat_fn(data)
    n = len(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(data) for _ in range(n)]
        boot_stats.append(stat_fn(sample))

    boot_stats.sort()
    alpha = 1.0 - confidence
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, lo_idx)
    hi_idx = min(n_bootstrap - 1, hi_idx)

    return BootstrapCI(
        estimate=point,
        lower=boot_stats[lo_idx],
        upper=boot_stats[hi_idx],
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Pooled Cohen's d. Returns 0.0 if either group has fewer than 2 samples."""
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)
    n_a, n_b = len(group_a), len(group_b)
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return 0.0
    return abs(mean_a - mean_b) / pooled_sd


def _effect_label(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Hypothesis tests (pure Python, no scipy)
# ---------------------------------------------------------------------------

def _factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _hypergeometric_prob(N: int, K: int, n: int, k: int) -> float:
    """P(X = k) under Hypergeometric(N, K, n)."""
    num = _comb(K, k) * _comb(N - K, n - k)
    den = _comb(N, n)
    if den == 0:
        return 0.0
    return num / den


def fishers_exact_test(
    a_success: int, a_fail: int, b_success: int, b_fail: int
) -> tuple[float, float]:
    """
    Two-tailed Fisher's exact test for a 2×2 contingency table.

    Table:
                success   fail
        group_a   a_s      a_f
        group_b   b_s      b_f

    Returns (odds_ratio, p_value).
    """
    # Odds ratio
    if a_fail == 0 or b_success == 0:
        odds_ratio = float("inf")
    else:
        odds_ratio = (a_success * b_fail) / (a_fail * b_success)

    # Two-tailed p-value via hypergeometric distribution
    N = a_success + a_fail + b_success + b_fail
    K = a_success + b_success      # total successes
    n = a_success + a_fail         # group_a total
    k_obs = a_success

    p_obs = _hypergeometric_prob(N, K, n, k_obs)

    p_value = 0.0
    k_min = max(0, n + K - N)
    k_max = min(n, K)
    for k in range(k_min, k_max + 1):
        p_k = _hypergeometric_prob(N, K, n, k)
        if p_k <= p_obs + 1e-10:
            p_value += p_k

    return odds_ratio, min(p_value, 1.0)


def mann_whitney_u(group_a: list[float], group_b: list[float]) -> tuple[float, float]:
    """
    Mann-Whitney U test (two-tailed). Returns (U, approximate p_value).
    Uses normal approximation, valid when both groups have n > 10.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0

    # Count U
    U = 0
    for a in group_a:
        for b in group_b:
            if a > b:
                U += 1
            elif a == b:
                U += 0.5

    # Normal approximation
    mu_U = n_a * n_b / 2
    sigma_U = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    if sigma_U == 0:
        return U, 1.0

    z = (U - mu_U) / sigma_U
    # Two-tailed p from standard normal via erf
    p_value = 2 * (1 - _standard_normal_cdf(abs(z)))
    return U, min(p_value, 1.0)


def _standard_normal_cdf(z: float) -> float:
    """CDF of the standard normal distribution using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Apply Bonferroni correction.
    Returns a boolean mask: True where the corrected p-value < alpha.
    """
    m = len(p_values)
    if m == 0:
        return []
    threshold = alpha / m
    return [p < threshold for p in p_values]


# ---------------------------------------------------------------------------
# Pairwise tests across providers
# ---------------------------------------------------------------------------

PROVIDER_PAIRS = [
    ("claude", "openai"),
    ("claude", "gemini"),
    ("openai", "gemini"),
]


def pairwise_tests(
    results_by_provider: dict[str, dict],
    scenario_id: str,
    alpha: float = 0.05,
) -> list[HypothesisTest]:
    """
    Run all pairwise hypothesis tests for a scenario.

    Args:
        results_by_provider: {provider_name: ScenarioResult or dict with
                               per_run_success, per_run_scores, per_run_latencies}
        scenario_id:         For labeling output.
        alpha:               Family-wise error rate before Bonferroni.

    Returns:
        List of HypothesisTest objects (significant field already corrected).
    """
    tests: list[HypothesisTest] = []

    for pa, pb in PROVIDER_PAIRS:
        if pa not in results_by_provider or pb not in results_by_provider:
            continue

        res_a = results_by_provider[pa]
        res_b = results_by_provider[pb]

        # Extract per-run arrays
        def _get(res, key, fallback_rate):
            if isinstance(res, dict):
                arr = res.get(key, [])
            else:
                arr = getattr(res, key, [])
            return arr if arr else [fallback_rate]

        successes_a = _get(res_a, "per_run_success", res_a.get("success_rate", 0) if isinstance(res_a, dict) else res_a.success_rate)
        successes_b = _get(res_b, "per_run_success", res_b.get("success_rate", 0) if isinstance(res_b, dict) else res_b.success_rate)
        scores_a = _get(res_a, "per_run_scores", res_a.get("consistency_score", 0) if isinstance(res_a, dict) else res_a.consistency_score)
        scores_b = _get(res_b, "per_run_scores", res_b.get("consistency_score", 0) if isinstance(res_b, dict) else res_b.consistency_score)
        latencies_a = _get(res_a, "per_run_latencies", [res_a.get("mean_latency_ms", 0)] if isinstance(res_a, dict) else [res_a.mean_latency_ms])
        latencies_b = _get(res_b, "per_run_latencies", [res_b.get("mean_latency_ms", 0)] if isinstance(res_b, dict) else [res_b.mean_latency_ms])

        # 1. Fisher's exact on success/fail counts
        a_s = sum(1 for x in successes_a if x > 0.5)
        a_f = len(successes_a) - a_s
        b_s = sum(1 for x in successes_b if x > 0.5)
        b_f = len(successes_b) - b_s
        odds, p_fisher = fishers_exact_test(a_s, a_f, b_s, b_f)
        tests.append(HypothesisTest(
            test_name="fishers_exact",
            provider_a=pa,
            provider_b=pb,
            metric="success_rate",
            statistic=odds,
            p_value=p_fisher,
            significant=False,  # set after Bonferroni
            effect_size=odds,
            effect_label="odds_ratio",
        ))

        # 2. Mann-Whitney U on quality scores
        U_score, p_mw_score = mann_whitney_u(scores_a, scores_b)
        d_score = cohens_d(scores_a, scores_b)
        tests.append(HypothesisTest(
            test_name="mann_whitney_u",
            provider_a=pa,
            provider_b=pb,
            metric="quality_score",
            statistic=U_score,
            p_value=p_mw_score,
            significant=False,
            effect_size=d_score,
            effect_label=_effect_label(d_score),
        ))

        # 3. Mann-Whitney U on latency
        U_lat, p_mw_lat = mann_whitney_u(latencies_a, latencies_b)
        d_lat = cohens_d(latencies_a, latencies_b)
        tests.append(HypothesisTest(
            test_name="mann_whitney_u",
            provider_a=pa,
            provider_b=pb,
            metric="latency_ms",
            statistic=U_lat,
            p_value=p_mw_lat,
            significant=False,
            effect_size=d_lat,
            effect_label=_effect_label(d_lat),
        ))

    # Apply Bonferroni correction across all tests in this scenario
    p_values = [t.p_value for t in tests]
    sig_mask = bonferroni_correction(p_values, alpha=alpha)
    for test, sig in zip(tests, sig_mask):
        test.significant = sig

    return tests


# ---------------------------------------------------------------------------
# High-level summary builder
# ---------------------------------------------------------------------------

def compute_scenario_stats(
    scenario_id: str,
    provider: str,
    per_run_success: list[float],
    per_run_scores: list[float],
    per_run_latencies: list[float],
    n_bootstrap: int = 10_000,
) -> ScenarioStats:
    """
    Compute bootstrap CIs for all three key metrics.

    All input lists should be per-run values (one float per API call or run).
    Binary success/fail should be encoded as 1.0 / 0.0.
    """
    return ScenarioStats(
        scenario_id=scenario_id,
        provider=provider,
        n=len(per_run_success),
        success_rate_ci=bootstrap_ci(per_run_success, n_bootstrap=n_bootstrap),
        score_ci=bootstrap_ci(per_run_scores, n_bootstrap=n_bootstrap),
        latency_ci=bootstrap_ci(per_run_latencies, n_bootstrap=n_bootstrap),
        raw_successes=per_run_success,
        raw_scores=per_run_scores,
        raw_latencies=per_run_latencies,
    )


def format_stats_table(stats_list: list[ScenarioStats]) -> str:
    """
    Render a markdown table of bootstrap CIs for all (scenario, provider) combos.
    """
    header = (
        "| Scenario | Provider | N | Success Rate 95% CI | "
        "Quality Score 95% CI | Latency 95% CI (ms) |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for s in stats_list:
        rows.append(
            f"| {s.scenario_id} | {s.provider} | {s.n} "
            f"| {s.success_rate_ci} "
            f"| {s.score_ci} "
            f"| {s.latency_ci} |"
        )
    return header + "\n".join(rows)


def format_test_table(tests: list[HypothesisTest]) -> str:
    """Render a markdown table of hypothesis test results."""
    header = (
        "| Provider A | Provider B | Metric | Test | Statistic | p-value | Bonferroni sig. | Effect |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for t in tests:
        sig = "✓" if t.significant else "✗"
        rows.append(
            f"| {t.provider_a} | {t.provider_b} | {t.metric} | {t.test_name} "
            f"| {t.statistic:.3f} | {t.p_value:.4f} | {sig} "
            f"| {t.effect_size:.3f} ({t.effect_label}) |"
        )
    return header + "\n".join(rows)
