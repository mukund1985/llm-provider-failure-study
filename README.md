# Production Failure Modes in Large Language Model APIs: An Empirical Cross-Provider Study

**Mukund Pandey** — Independent Research

Submitted to arXiv on 3 August 2026 (cs.AI, cross-listed cs.SE). Paper: [`paper.pdf`](paper.pdf) / [`paper.tex`](paper.tex)

## What this is

An empirical study measuring five production-critical failure modes across three commercial LLM API providers — Anthropic Claude (claude-haiku-4-5), OpenAI GPT-4o-mini, and Google Gemini (gemini-3.1-flash-lite) — using 1,668 real API calls across two experimental sessions, plus 540 LLM-as-judge evaluation calls.

## Key findings

1. **All three providers achieve 100% API success rates** across every scenario when Gemini's free-tier rate limits are respected (4-second inter-call gap). The reliability gap in our earlier v1 run was a call-rate artifact, not a model or infrastructure problem.
2. **Latency is the dominant statistically significant differentiator.** Claude is consistently ~300–500 ms slower than OpenAI (medium effect sizes, Bonferroni-corrected p < 0.001); Gemini is dramatically faster on error recovery (Cohen's d = 4.061 vs Claude).
3. **System prompts do not reliably enforce mandatory content.** When a system prompt requires a legal disclaimer and the user explicitly asks to skip it, Claude and Gemini comply with the user 100% of the time and OpenAI 67% of the time — replicated exactly across two experimental dates. Operators should enforce compliance content in post-processing, not prompts.
4. Tool calling, error recovery, and long-context fact retrieval (up to 16,000 words) are perfect for all three providers at this tier.

## Methodology

- N=90 per scenario per provider (3 prompt variants × 30 repetitions)
- Bootstrap 95% confidence intervals (10,000 resamples)
- Fisher's exact / Mann-Whitney U pairwise tests with Bonferroni correction
- LLM-as-judge rubric scoring (GPT-4o-mini, 0–4 scale) for error recovery and instruction conflict
- Two-date replication of the instruction-conflict and error-recovery scenarios

## Repository layout

| Path | Contents |
|------|----------|
| `paper.tex` / `paper.pdf` | The paper (LaTeX source and compiled PDF) |
| `run_experiments_v2.py` | Main experiment entrypoint |
| `experiments/scenarios_v2.py` | The five failure-mode scenarios |
| `experiments/runner_v2.py` | Experiment runner with per-run array logging |
| `providers/` | Unified provider wrappers (Anthropic, OpenAI, Gemini) with rate limiting |
| `analysis/stats.py` | Bootstrap CIs, hypothesis tests, effect sizes (pure Python) |
| `analysis/llm_judge.py` | LLM-as-judge evaluation module |
| `run_judge.py` | Judge evaluation entrypoint |
| `results_v2/` | Session 1 results (2026-07-31, all five scenarios) |
| `results_v2_judge/` | Session 2 results (2026-08-02, S3+S5 replication with response texts) and judge scores |
| `experiment_v2.log` / `experiment_v2_judge.log` | Full experiment logs |

## Reproducing

```bash
pip install anthropic openai google-genai sentence-transformers scikit-learn --break-system-packages

export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...

python run_experiments_v2.py                 # full run, ~40 min (Gemini rate limit dominates)
python run_judge.py results_v2_judge/<results-file>.json   # judge evaluation
```

All statistics in the paper are computed from the per-run arrays in the results JSON files; no external data is required.
