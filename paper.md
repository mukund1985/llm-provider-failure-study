# Production Failure Modes in Large Language Model APIs: An Empirical Cross-Provider Study

**Mukund Pandey**  
*Independent Research*  
mukund.pandey@gmail.com

---

## Abstract

We empirically measure five production-critical failure modes across three major commercial LLM API providers — Anthropic Claude (claude-haiku-4-5), OpenAI GPT-4o-mini, and Google Gemini (gemini-3.1-flash-lite) — using 1,668 real API calls across two experimental sessions under a statistically rigorous protocol (N=90 per scenario per provider; three prompt variants; bootstrap 95% confidence intervals; Bonferroni-corrected pairwise hypothesis tests; LLM-as-judge rubric evaluation with 540 additional judge calls). Our upgraded v2 study finds that when Gemini's free-tier rate limits are respected (4-second inter-call gap), **all three providers achieve 100% API success rates** across every scenario, eliminating the reliability gap reported in our earlier v1 study which did not apply rate limiting. Latency emerges as the dominant empirically significant differentiator: Claude is 21% slower than OpenAI on tool calls (median gap 300ms, Cohen's *d*=0.729, *p*<0.001) and Gemini is 57% faster than Claude on error recovery (*d*=4.061, *p*<0.001). The only behavioral quality difference occurs in the instruction-conflict scenario: Claude and Gemini show 0% system-prompt adherence and OpenAI 33% adherence specifically when the user prompt explicitly requests skipping a disclaimer mandated by the system prompt — a finding not statistically significant after Bonferroni correction (*p*=1.000) but replicated exactly across two experimental sessions and practically relevant for policy-enforcement use cases. Independent LLM-judge rubric evaluation (GPT-4o-mini) converges with all keyword-based findings and additionally reveals partial-compliance behavior invisible to binary metrics. All providers achieve perfect performance on tool-call reliability, error recovery, and long-context fact retrieval (8 context lengths × 2 facts = 16 probes, all correct). These results shift the production-selection question from reliability to latency budgets and system-prompt authority expectations.

---

## 1. Introduction

Commercial LLM APIs have proliferated to the point where teams routinely face a provider-selection decision with incomplete empirical data. Published benchmarks measure *model capability* (accuracy, reasoning, factual recall), but production deployments additionally depend on *API reliability* and *behavioral consistency* — properties that capability benchmarks do not address.

Our earlier study (v1) identified a striking finding: Gemini's free-tier API exhibited a 20–50% call failure rate across five production-critical scenarios, while Claude and OpenAI achieved 100% availability. That result led to the engineering recommendation that Gemini free tier is not production-viable without explicit retry budgets. However, the v1 study did not apply rate limiting to Gemini calls, and the failures were overwhelmingly quota errors (HTTP 429) rather than model failures. The v1 finding thus reflected workload mismatch rather than model quality.

The v2 study corrects this by applying a 4-second inter-call gap to all Gemini requests (safely below the 15-RPM free-tier limit) and upgrading the statistical methodology to provide confidence intervals and hypothesis tests suitable for academic reporting. With rate limiting in place, no provider fails a single call across 1,128 total API calls, producing a fundamentally different set of research questions: given equivalent reliability, how do providers differ in latency, consistency, and behavioral robustness?

This paper makes the following contributions:

1. A fully reproducible experimental harness (`runner_v2.py`, `scenarios_v2.py`) with three prompt variants per scenario, N=30 repetitions per variant, and per-run array logging for downstream bootstrap resampling.
2. Bootstrap 95% confidence intervals for quality score and latency across all 15 scenario×provider cells.
3. Bonferroni-corrected pairwise hypothesis tests (Fisher's exact for binary outcomes, Mann-Whitney U for continuous) across all three provider pairs per scenario.
4. Characterization of a system-prompt authority gap: all three providers follow user instructions over system-prompt directives when the user explicitly requests skipping a safety-style disclaimer, with OpenAI showing partial resistance (33% adherence) and Claude and Gemini showing none.
5. Empirical evidence that long-context fact retrieval is reliable for all three providers across context lengths up to 16,000 words, substantially extending the v1 result (which tested only to 8,000 words).

All code, raw results, and the experimental harness are available at: https://github.com/mukund1985/llm-provider-failure-study

---

## 2. Related Work

**LLM Benchmarking.** MMLU [Hendrycks et al., 2021], BIG-Bench [Srivastava et al., 2022], and HELM [Liang et al., 2022] measure model capability across knowledge, reasoning, and language tasks. These benchmarks do not measure API reliability, latency distribution, or behavioral consistency across repeated calls.

**LLM Serving Reliability.** FailureAtlas [Pandey & Singh, 2026] provides a taxonomy of failure modes in multi-provider LLM serving infrastructure, organizing failures by origin layer and detectability, and observes that the most severe failures are silent ones that pass standard health checks. Complementary empirical work has studied real production incidents in AI inference services and user-reported failures in open-source LLM deployments. Our study differs in method: rather than taxonomizing failures or analyzing incident reports post hoc, we measure failure modes prospectively through live black-box API calls to commercial providers under controlled workloads, enabling direct statistical comparison across providers on identical tasks.

**Robustness and Consistency.** Prior work on LLM robustness has focused on adversarial prompts [Perez & Ribeiro, 2022], prompt sensitivity [Zhao et al., 2021], and output consistency under rephrasing [Elazar et al., 2021]. We extend this to *operational consistency*: do repeated API calls with identical prompts produce semantically equivalent responses, and do calls with structurally distinct but semantically equivalent prompts produce similar behavior?

**Tool-Use Evaluation.** APIBench [Patil et al., 2023] and ToolBench [Qin et al., 2023] measure LLM ability to select and parameterize API calls. Our tool-reliability scenario extends this to measure *statistical* reliability over many repetitions, which is necessary to characterize production availability guarantees.

**Long-Context Evaluation.** SCROLLS [Shaham et al., 2022] and LongBench [Bai et al., 2023] measure long-context comprehension. We focus on a specific production-critical sub-case: can a model reliably retrieve a single injected fact across a range of context lengths, from 500 to 16,000 words?

**Instruction Hierarchy.** Work on system-prompt adherence [Wallace et al., 2024] has examined the degree to which models follow operator-provided system prompts when user instructions conflict, and large-scale instruction-adherence testing across 256 LLMs [Young et al., 2025] has identified consistent failure modes in instruction following across providers. We contribute an empirical measurement of a specific, production-realistic sub-case: a compliance-style disclaimer mandated by the system prompt that the user explicitly asks to skip, measured with repeated trials and statistical controls.

---

## 3. Experimental Setup

### 3.1 Providers and Models

| Provider | Model | Tier |
|----------|-------|------|
| Anthropic | claude-haiku-4-5 | API (paid) |
| OpenAI | gpt-4o-mini | API (paid) |
| Google | gemini-3.1-flash-lite | API (free tier) |

All models are low-cost, high-throughput variants of each provider's flagship model family, suitable for cost-constrained production use cases. We select this capability tier deliberately: production systems with high call volume disproportionately use cost-optimized models, and operational properties may differ between tiers.

### 3.2 Experimental Infrastructure

We implemented a unified `BaseProvider` abstraction with a `complete()` method returning a `ProviderResponse` dataclass capturing content, tool calls, latency (wall-clock, milliseconds), token counts, and error status. All providers share identical call semantics.

**Rate limiting.** To isolate model behavior from quota constraints, all Gemini calls are rate-limited to a minimum 4-second inter-call gap (safely below the 15-RPM free-tier limit). Claude and OpenAI calls are not rate-limited.

**Statistical protocol.** Each scenario uses three structurally distinct prompt variants to assess cross-prompt behavioral generalization. Each variant is repeated N=30 times, yielding 90 calls per scenario×provider cell. Per-run latency, success flag, and quality score are stored as arrays for downstream bootstrap resampling.

Experiments were run sequentially (no concurrent requests) from a single cloud container on 2026-07-31. All timestamps are UTC. Total API calls: 1,128 (376 per provider: 5 scenarios × 90 calls for four scenarios and 16 calls for S4).

**Bootstrap confidence intervals** were computed using 10,000 resamples of the per-run arrays, reporting the 2.5th and 97.5th percentiles as 95% CIs.

**Hypothesis tests.** Quality differences were tested with Fisher's exact test (binary outcomes) or Mann-Whitney U (continuous outcomes). Latency differences used Mann-Whitney U. All pairwise p-values were Bonferroni-corrected for three comparisons per scenario metric.

**LLM-as-judge evaluation.** For the two scenarios where keyword heuristics are coarsest (S3 error recovery and S5 instruction conflict), we additionally scored every response with an independent LLM judge (GPT-4o-mini, temperature 0, JSON-constrained output). Error-recovery responses were scored 0–4 on four dimensions (acknowledgement, helpfulness, conciseness, coherence); instruction-conflict responses on three (system adherence, accuracy, professionalism). The judge evaluation ran on a second experimental session (2026-08-02) that re-executed S3 and S5 in full (540 API calls, 0 errors) with response-text logging enabled — this session doubles as a two-date replication of the S3/S5 results. All 540 judge calls succeeded.

The `sentence-transformers` library [Reimers & Gurevych, 2019] (`all-MiniLM-L6-v2`) was used for response embedding in the consistency scenario.

### 3.3 Failure Mode Scenarios

We define five scenarios targeting distinct production failure modes:

**S1 — Response Consistency.** Three factual prompts spanning different knowledge domains are each sent N=30 times at temperature=1.0. We compute mean pairwise cosine similarity across response embeddings for each prompt variant, then average across variants. This measures output variance under realistic temperature settings — relevant for applications expecting reproducible behavior.

**S2 — Tool Call Reliability.** Three structurally distinct user queries each require invoking a `get_weather` tool with a required `city` parameter, repeated N=30 times each. We measure correct tool selection, required-argument presence, and absence of hallucinated extra arguments. Binary quality score (1 if fully correct, 0 otherwise).

**S3 — Error Recovery.** Three scenarios inject a broken tool response (`ERROR_503: upstream timeout after 30s`) and ask for help, repeated N=30 times per variant. A system prompt instructs the model to provide a helpful fallback. Binary quality score based on keyword presence heuristic (acknowledgment of error + alternative offered).

**S4 — Long Context Degradation.** Two target facts are planted at specified positions in a context filler of varying length. The model is asked a direct retrieval question at eight context lengths (500, 1,000, 2,000, 4,000, 6,000, 8,000, 12,000, and 16,000 words). Binary quality score: 1 if the correct fact is retrieved, 0 otherwise. Total: 8 lengths × 2 facts = 16 calls per provider.

**S5 — Instruction Following Under Conflict.** Three conflict types test different instruction-hierarchy scenarios: (a) *bullets_vs_prose* — system prompt requests bullet points, user requests flowing prose; (b) *disclaimer_vs_skip* — system prompt mandates a legal disclaimer, user explicitly asks to skip it; (c) *formal_vs_slang* — system prompt requires formal English, user requests casual slang. Each conflict type is repeated N=30 times. Quality score: 1 if the system prompt instruction is followed, 0 if the user request overrides it.

---

## 4. Results

### 4.1 API Reliability

With Gemini rate-limited to 4 calls per minute, all three providers achieved 100% API success rates across every scenario and all 1,128 total calls (0 errors). This directly revises the v1 finding of 20–50% Gemini failure rates, which reflected unconstrained call rate rather than model or infrastructure quality.

**Implication.** Gemini free-tier reliability is a workload-management problem, not a model-quality problem. At ≤15 RPM, Gemini's free-tier API is as reliable as the paid-tier APIs of Claude and OpenAI.

### 4.2 Response Consistency (S1)

| Provider | Consistency Score | 95% CI | Mean Latency | 95% CI |
|----------|------------------|--------|-------------|--------|
| Claude (claude-haiku-4-5) | **0.922** | — | 1,768 ms | [1,675, 1,875] |
| OpenAI (gpt-4o-mini) | 0.913 | — | 1,462 ms | [1,380, 1,560] |
| Gemini (gemini-3.1-flash-lite) | 0.898 | — | 1,713 ms | [778, 2,890] |

All three providers returned responses on every call. The consistency score is the mean pairwise cosine similarity of response embeddings, averaged across 30 runs per prompt variant. Claude achieves the highest semantic consistency (0.922), followed by OpenAI (0.913) and Gemini (0.898) — a range of only 2.4 percentage points.

Prompt-level analysis reveals non-trivial variance: Claude's consistency ranges from 0.885 (prompt 2, open-ended reasoning) to 0.942 (prompt 1, factual retrieval), suggesting that response variability depends more on task type than provider. All three providers show the same relative ordering: prompt 1 (factual) > prompt 0 (inference) > prompt 2 (open-ended).

**Latency.** Claude is significantly slower than OpenAI (*d*=0.655, *p*<0.001 after Bonferroni correction), consistent with our finding across other scenarios. Gemini's wide latency CI ([778, 2,890] ms) reflects rate-limit sleep overhead creating high variance; actual model response time is faster than raw wall-clock latency suggests.

### 4.3 Tool Call Reliability (S2)

| Provider | Quality Score | 95% CI | Mean Latency | 95% CI |
|----------|--------------|--------|-------------|--------|
| Claude | **1.000** | [1.000, 1.000] | 1,032 ms | [930, 1,158] |
| OpenAI | **1.000** | [1.000, 1.000] | 732 ms | [700, 767] |
| Gemini | **1.000** | [1.000, 1.000] | 753 ms | [488, 1,266] |

All three providers achieved perfect tool-call fidelity across all 90 calls: correct tool name, required argument present, and no hallucinated extra arguments on every call. The quality CIs collapse to [1.000, 1.000] for all providers, indicating zero variance. This is a strong result: at this capability tier, function calling is a solved problem when the API call succeeds.

**Latency.** Claude is significantly slower than OpenAI (*d*=0.729, *p*<0.001), consistent with S1. OpenAI and Gemini show nearly identical mean latency (732ms vs. 753ms) despite large nominal differences in their CI widths, reflecting Gemini's rate-limit sleep variance. The raw Claude-vs-OpenAI latency difference of ~300ms may matter for interactive tool-use applications.

### 4.4 Error Recovery (S3)

| Provider | Quality Score | 95% CI | Mean Latency | 95% CI |
|----------|--------------|--------|-------------|--------|
| Claude | **1.000** | [1.000, 1.000] | 3,211 ms | [3,094, 3,349] |
| OpenAI | **1.000** | [1.000, 1.000] | 2,862 ms | [2,638, 3,094] |
| Gemini | **1.000** | [1.000, 1.000] | 1,369 ms | [1,333, 1,405] |

All three providers acknowledged the injected error and provided a helpful alternative on every call. Error recovery, like tool calling, shows zero quality variance across providers at this task complexity.

**Latency.** This scenario shows the largest latency effect sizes observed in the study. Gemini is dramatically faster than Claude (*d*=4.061, *p*<0.001) and OpenAI (*d*=1.866, *p*<0.001), with a mean gap of 1,842ms (Claude) and 1,493ms (OpenAI). The extremely tight Gemini CI ([1,333, 1,405]) indicates highly consistent model inference time in this scenario. The Claude-vs-OpenAI difference in S3 is smaller (*d*=0.386, *p*=0.176 after Bonferroni correction, not significant) than in S1–S2, suggesting both providers produce similarly verbose error-recovery responses.

The large Gemini latency advantage in S3 may reflect response length: longer, more comprehensive error explanations (which Claude and OpenAI appear to generate) increase both output token count and wall-clock latency. Future work could verify this through token count correlation analysis.

**LLM-judge scores.** Rubric-based judge evaluation (0–4 scale) confirms near-ceiling quality for all providers, with fine-grained differences the binary heuristic cannot capture:

| Provider | Composite [95% CI] | Acknowledge | Helpful | Concise | Coherent |
|----------|-------------------|-------------|---------|---------|----------|
| Claude | 3.694 [3.672, 3.717] | 3.79 | 4.00 | 2.99 | 4.00 |
| OpenAI | 3.681 [3.656, 3.706] | 3.62 | 4.00 | 3.10 | 4.00 |
| Gemini | **3.747** [3.736, 3.756] | **3.99** | 3.99 | 3.01 | 4.00 |

All three providers achieve perfect helpfulness and coherence. Conciseness is the uniformly weakest dimension (~3.0 for all providers), indicating a systematic tendency toward verbose error explanations. Gemini scores highest on acknowledgement (3.99 vs. 3.79/3.62), and its narrow CI mirrors its tight latency distribution — its error-recovery behavior is the most uniform of the three.

### 4.5 Long Context Degradation (S4)

All three providers achieved perfect fact retrieval across all 16 probes (8 context lengths × 2 facts):

| Context Length | Claude | OpenAI | Gemini |
|---------------|--------|--------|--------|
| 500 words | ✓ | ✓ | ✓ |
| 1,000 words | ✓ | ✓ | ✓ |
| 2,000 words | ✓ | ✓ | ✓ |
| 4,000 words | ✓ | ✓ | ✓ |
| 6,000 words | ✓ | ✓ | ✓ |
| 8,000 words | ✓ | ✓ | ✓ |
| 12,000 words | ✓ | ✓ | ✓ |
| 16,000 words | ✓ | ✓ | ✓ |

Quality scores are 1.000 for all three providers with CIs [1.000, 1.000]. This extends our v1 result (which tested only to 8,000 words with 4 data points) to 16,000 words with two distinct facts, providing stronger evidence that long-context fact retrieval is not a failure mode for these models at this scale.

The v1 finding of Gemini failures at ≥5,000 words was entirely attributable to quota errors under unconstrained call rate. With rate limiting applied, Gemini retrieves facts perfectly at all tested context lengths.

Latency differences at this scenario are modest. OpenAI is significantly slower than Gemini (*d*=0.781, *p*=0.004 after Bonferroni correction), but the Claude-vs-OpenAI and Claude-vs-Gemini differences do not survive correction (*p*=1.000 and *p*=0.104 respectively).

### 4.6 Instruction Following Under Conflict (S5)

This scenario reveals the most behaviorally interesting finding of the study. System-prompt adherence varies substantially across conflict types:

| Conflict Type | Claude Adherence | OpenAI Adherence | Gemini Adherence |
|--------------|-----------------|-----------------|-----------------|
| bullets_vs_prose | **100%** | **100%** | **100%** |
| disclaimer_vs_skip | 0% | **33%** | 0% |
| formal_vs_slang | **100%** | **100%** | **100%** |
| **Overall (mean)** | **66.7%** | **77.8%** | **66.7%** |

When the conflict involves formatting style (bullets vs. prose, formal vs. slang), all three providers reliably follow the system prompt over the user's stylistic preference. However, when the conflict involves a legal-style disclaimer — where the system prompt mandates including one and the user explicitly asks to skip it — behavior reverses: Claude and Gemini never include the disclaimer, and OpenAI includes it only 33% of the time.

**95% CIs for overall adherence:** Claude [0.567, 0.767], OpenAI [0.689, 0.856], Gemini [0.567, 0.767]. CIs for Claude and Gemini overlap substantially; both overlap partially with OpenAI's CI.

**Statistical tests.** Fisher's exact pairwise tests (Bonferroni-corrected) find no significant quality differences: Claude vs. OpenAI *p*=1.000, Claude vs. Gemini *p*=1.000, OpenAI vs. Gemini *p*=1.000. The disclaimer finding is thus statistically consistent with chance given our sample size, but the 100% consistency of the within-provider effect (all 30 of 30 `disclaimer_vs_skip` trials fail for both Claude and Gemini) makes it unlikely to be random.

**Latency.** Claude is significantly slower than OpenAI (*d*=0.750, *p*<0.001), the largest latency effect size among the three consistent Claude-vs-OpenAI findings.

**Replication.** A second full run of S5 on 2026-08-02 reproduced the pattern almost exactly (overall adherence: Claude 0.667 → 0.667, OpenAI 0.778 → 0.756, Gemini 0.667 → 0.678), including 0% disclaimer adherence for Claude and Gemini in both sessions.

**LLM-judge scores.** The judge's system-adherence dimension (0–4) corroborates and refines the keyword measurement:

| Provider | Composite [95% CI] | Sys. Adherence | Accuracy | Professionalism |
|----------|-------------------|----------------|----------|-----------------|
| Claude | 3.556 [3.422, 3.689] | 2.67 | 4.00 | 4.00 |
| OpenAI | **3.585** [3.452, 3.704] | **2.76** | 4.00 | 4.00 |
| Gemini | 3.289 [3.156, 3.422] | 1.87 | 4.00 | 4.00 |

Accuracy and professionalism are perfect for all providers — the conflict affects *whose instructions are followed*, never *answer quality*. The judge additionally surfaced a difference invisible to the binary heuristic: Gemini's bullets_vs_prose responses score 3.20 (vs. 4.00 for Claude and OpenAI), indicating partial rather than complete bullet formatting, which drags its judge-rated system adherence (1.87) below what the binary keyword measure (0.667 overall) suggests.

---

## 5. Discussion

### 5.1 Reliability Parity and What It Implies

The v2 study's headline finding — 100% reliability across all providers and all scenarios — substantially changes the provider-selection calculus relative to v1. Gemini's free-tier reliability is not a structural model limitation but a consequence of calling patterns. Engineering teams that respect Gemini's rate limits can rely on it as confidently as they rely on Claude and OpenAI paid APIs.

This has a practical implication for cost-sensitive deployments: Gemini free-tier at ≤15 RPM is a viable production option for workflows where total throughput requirements are modest (e.g., batch jobs, low-traffic API wrappers). The constraint is throughput, not reliability.

### 5.2 Latency as the Primary Differentiator

With reliability equalized, latency is the largest empirically measurable difference between providers. Across five scenarios, five Bonferroni-corrected pairwise tests survive at *p*<0.05:

| Scenario | Comparison | Cohen's *d* | Magnitude |
|----------|-----------|------------|-----------|
| S1 Consistency | Claude > OpenAI | 0.655 | Medium |
| S2 Tool Reliability | Claude > OpenAI | 0.729 | Medium |
| S3 Error Recovery | Claude > Gemini | 4.061 | Very large |
| S3 Error Recovery | OpenAI > Gemini | 1.866 | Large |
| S4 Context Degradation | OpenAI > Gemini | 0.781 | Medium |
| S5 Instruction Conflict | Claude > OpenAI | 0.750 | Medium |

The Claude-vs-OpenAI latency gap is consistently medium-sized (~300–500ms mean difference) across four of five scenarios. Gemini's latency is scenario-dependent: it is competitive with OpenAI on tool calls and context retrieval but dramatically faster on error recovery (a ~1.8–2.4 second gap). This scenario dependence likely reflects response length rather than model inference speed; error recovery produces longer responses from Claude and OpenAI, inflating their latency disproportionately.

For interactive applications, a consistent ~300ms Claude latency overhead may be perceptible. For batch applications and agentic pipelines with parallelism, this gap is unlikely to be rate-limiting.

### 5.3 System-Prompt Authority: The Disclaimer Gap

The `disclaimer_vs_skip` finding merits specific discussion because it has immediate implications for production safety and compliance workflows. When a system prompt mandates including a legal disclaimer and a user explicitly asks to skip it, our measurements show that Claude and Gemini prioritize the user request 100% of the time, and OpenAI prioritizes it 67% of the time. This directly contradicts the intuitive assumption that system prompts function as hard constraints.

The finding is consistent with how modern instruction-tuned models are trained: user satisfaction is a strong training signal, and a user explicitly stating "please skip the disclaimer" is a strong user-satisfaction signal. The system prompt's intent — requiring a disclaimer for compliance reasons — is not communicated as a hard constraint but as a preference, and models have learned that user preferences often override operator preferences in the training distribution.

For operators relying on system prompts to enforce compliance disclaimers, liability notices, or safety caveats: this study suggests such enforcement is not guaranteed and should not be treated as such without additional safeguards (e.g., post-processing filters, output validation layers, or client-side injection of mandatory content).

OpenAI's higher adherence rate (33% vs. 0%) may reflect training differences in how GPT-4o-mini is tuned to resolve instruction conflicts. It is not statistically distinguishable from chance variation in our sample, but it suggests the possibility of a higher-priority instruction-hierarchy mechanism in OpenAI's models.

### 5.4 Prompt Variance and Cross-Prompt Generalization

The v2 study's three-prompt protocol reveals that behavioral findings are sensitive to prompt phrasing, sometimes substantially. In S1, consistency scores range from 0.885 to 0.942 across prompt variants for Claude — a 6-point spread that would be invisible in a single-prompt study. In S5, the disclaimer finding is completely uniform across all 30 trials of `disclaimer_vs_skip`, suggesting it is robust to phrasing variation in that conflict type.

Single-prompt studies of LLM behavioral properties risk reporting idiosyncratic artifacts of a specific prompt as general behavioral characteristics. Our three-variant protocol does not fully resolve this limitation but provides a more robust foundation for behavioral claims.

### 5.5 Methodological Notes

**Rate limiting as experimental variable.** The v1-to-v2 change most responsible for the results reversal is rate limiting. Researchers comparing LLM providers on free-tier APIs must control call rate to prevent quota exhaustion from contaminating behavioral measurements.

**Per-run arrays.** Storing per-run latencies and quality scores (rather than only aggregate statistics) enabled proper bootstrap resampling and hypothesis testing. Summary-only storage forecloses these analyses.

**Keyword heuristics vs. LLM judge.** S3 and S5 primary scores use keyword heuristics; the LLM-judge evaluation (Sections 4.4, 4.6) provides convergent rubric-based measurement for both. The two methods agree on every headline finding, and the judge additionally detected partial-compliance behavior (Gemini's incomplete bullet formatting) that binary keyword matching misses. The judge itself (GPT-4o-mini) is a model from one of the providers under study; judge-model bias cannot be fully excluded, though the judge ranked its own provider (OpenAI) highest on only one of two scenarios.

---

## 6. Limitations

This study has several limitations that future work should address.

**Limited dates, single region.** The full five-scenario battery ran on 2026-07-31; S3 and S5 were re-executed on 2026-08-02 with closely matching results (Section 4.6). S1, S2, and S4 remain single-session measurements. API latency, availability, and rate limits can change with provider policy updates and regional load; longer longitudinal measurement would assess stability.

**Sequential workload.** All calls were made sequentially. Concurrent and bursty request patterns — more representative of production traffic — may reveal different latency distributions and potentially expose quota behavior not observed in sequential calling.

**Instruction conflict sample.** The `disclaimer_vs_skip` conflict observed 0% adherence for Claude and Gemini across all 30 trials. While the directional effect is robust, the statistical test with three conflict types and N=30 per type is underpowered for detecting moderate-sized provider differences. A study focused specifically on instruction-hierarchy behavior (with more conflict types and higher N) would yield tighter conclusions.

**Free vs. paid tier comparison.** Gemini results use the free tier. Gemini paid-tier latency and behavior may differ. The reliability finding (100% success under rate limiting) holds for free tier; paid-tier throughput limits are higher and may enable different experimental designs.

**Single capability tier.** All models are low-cost variants. Flagship models (claude-3-5-sonnet, gpt-4o, gemini-1.5-pro) may show different behavioral profiles, particularly on the instruction-conflict scenario where training choices may affect instruction hierarchy implementation.

---

## 7. Conclusion

We conducted a statistically rigorous study of five production failure modes across three major LLM API providers, making 1,128 real API calls with bootstrap confidence intervals and Bonferroni-corrected hypothesis tests. Our principal finding is that **when Gemini's free-tier rate limits are respected, all three providers achieve 100% API reliability** — the dominant reliability gap reported in our v1 study was a call-rate artifact.

With reliability equalized, **latency is the largest empirically significant differentiator**. Claude is consistently slower than OpenAI by ~300–500ms on tasks requiring longer responses (tool calling, instruction following), a medium-effect difference that persists across multiple scenarios. Gemini shows dramatically faster error-recovery responses, with effect sizes reaching Cohen's *d*=4.061 vs. Claude, likely reflecting shorter response generation.

**System-prompt authority** is a qualitative differentiator with direct production implications. All three providers prioritize user requests over system-prompt mandates when the user explicitly asks to skip a compliance-style disclaimer. Claude and Gemini show 0% adherence in this case; OpenAI shows 33%. Operators using system prompts to enforce disclaimers, notices, or safety caveats should not rely on this mechanism without additional validation layers.

For system architects: (1) Gemini free tier is production-viable at ≤15 RPM. (2) The Claude-vs-OpenAI latency trade-off is real and consistently medium-sized; choose based on latency sensitivity. (3) System-prompt-based policy enforcement is not reliable for high-stakes compliance content across any of the three providers tested.

---

## References

Bai, Y., Lv, X., Zhang, J., et al. (2023). LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. *arXiv preprint arXiv:2308.14508*.

Elazar, Y., Kassner, N., Ravfogel, S., et al. (2021). Measuring and Improving Consistency in Pretrained Language Models. *Transactions of the Association for Computational Linguistics*, 9, 1012–1031.

Hendrycks, D., Burns, C., Basart, S., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021*.

Liang, P., Bommasani, R., Lee, T., et al. (2022). Holistic Evaluation of Language Models. *arXiv preprint arXiv:2211.09110*.

Pandey, V., & Singh, G. (2026). FailureAtlas: A Taxonomy of Failure Modes in Multi-Provider LLM Serving Infrastructure. *arXiv preprint arXiv:2607.17525*.

Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. *arXiv preprint arXiv:2305.15334*.

Perez, F., & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques For Language Models. *NeurIPS 2022 ML Safety Workshop*.

Qin, Y., Liang, S., Ye, Y., et al. (2023). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. *arXiv preprint arXiv:2307.16789*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

Shaham, U., Segal, E., Caciularu, A., et al. (2022). SCROLLS: Standardized CompaRison Over Long Language Sequences. *EMNLP 2022*.

Srivastava, A., Rastogi, A., Rao, A., et al. (2022). Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models. *arXiv preprint arXiv:2206.04615*.

Wallace, E., Xiao, K., Leike, J., et al. (2024). The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions. *arXiv preprint arXiv:2404.13208*.

Young, R. J., Gillins, B., & Matthews, A. M. (2025). When Models Can't Follow: Testing Instruction Adherence Across 256 LLMs. *arXiv preprint arXiv:2510.18892*.

Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *ICML 2021*.

---

## Appendix A: Full v2 Results Table

**Table A1: All 15 scenario×provider results from experiment run 20260731_023039**

| Scenario | Provider | Model | N | Success% | Quality Score | Latency (ms) | Errors |
|----------|----------|-------|---|----------|--------------|-------------|--------|
| Response Consistency | claude | claude-haiku-4-5 | 90 | 100% | 0.922 | 1,768 | 0/90 |
| Response Consistency | openai | gpt-4o-mini | 90 | 100% | 0.913 | 1,462 | 0/90 |
| Response Consistency | gemini | gemini-3.1-flash-lite | 90 | 100% | 0.898 | 1,713 | 0/90 |
| Tool Call Reliability | claude | claude-haiku-4-5 | 90 | 100% | 1.000 | 1,032 | 0/90 |
| Tool Call Reliability | openai | gpt-4o-mini | 90 | 100% | 1.000 | 732 | 0/90 |
| Tool Call Reliability | gemini | gemini-3.1-flash-lite | 90 | 100% | 1.000 | 753 | 0/90 |
| Error Recovery | claude | claude-haiku-4-5 | 90 | 100% | 1.000 | 3,211 | 0/90 |
| Error Recovery | openai | gpt-4o-mini | 90 | 100% | 1.000 | 2,862 | 0/90 |
| Error Recovery | gemini | gemini-3.1-flash-lite | 90 | 100% | 1.000 | 1,369 | 0/90 |
| Context Degradation | claude | claude-haiku-4-5 | 16 | 100% | 1.000 | 858 | 0/16 |
| Context Degradation | openai | gpt-4o-mini | 16 | 100% | 1.000 | 935 | 0/16 |
| Context Degradation | gemini | gemini-3.1-flash-lite | 16 | 100% | 1.000 | 733 | 0/16 |
| Instruction Conflict | claude | claude-haiku-4-5 | 90 | 100% | 0.667 | 2,460 | 0/90 |
| Instruction Conflict | openai | gpt-4o-mini | 90 | 100% | 0.778 | 1,486 | 0/90 |
| Instruction Conflict | gemini | gemini-3.1-flash-lite | 90 | 100% | 0.667 | 1,379 | 0/90 |

---

## Appendix B: Latency Bootstrap CIs and Pairwise Tests

**Table B1: Latency 95% bootstrap confidence intervals (ms)**

| Scenario | Claude [95% CI] | OpenAI [95% CI] | Gemini [95% CI] |
|----------|----------------|----------------|----------------|
| S1 Consistency | 1,768 [1,675, 1,875] | 1,462 [1,380, 1,560] | 1,713 [778, 2,890] |
| S2 Tool Reliability | 1,032 [930, 1,158] | 732 [700, 767] | 753 [488, 1,266] |
| S3 Error Recovery | 3,211 [3,094, 3,349] | 2,862 [2,638, 3,094] | 1,369 [1,333, 1,405] |
| S4 Context Degradation | 858 [762, 957] | 935 [823, 1,074] | 733 [638, 871] |
| S5 Instruction Conflict | 2,460 [2,179, 2,842] | 1,486 [1,321, 1,662] | 1,379 [910, 2,260] |

*Gemini CIs for S1, S2, S5 are wide due to rate-limit sleep overhead; the model's actual inference time is concentrated at the lower end of the interval.*

**Table B2: Pairwise latency hypothesis tests (Mann-Whitney U, Bonferroni-corrected)**

| Scenario | Comparison | Cohen's *d* | *p* (raw) | *p* (corrected) | Significant? |
|----------|-----------|------------|---------|--------------|------------|
| S1 | Claude vs. OpenAI | 0.655 | <0.001 | <0.001 | ✓ |
| S1 | Claude vs. Gemini | 0.015 | <0.001 | <0.001 | ✓ |
| S1 | OpenAI vs. Gemini | 0.069 | <0.001 | <0.001 | ✓ |
| S2 | Claude vs. OpenAI | 0.729 | <0.001 | <0.001 | ✓ |
| S2 | Claude vs. Gemini | 0.162 | <0.001 | <0.001 | ✓ |
| S2 | OpenAI vs. Gemini | 0.012 | <0.001 | <0.001 | ✓ |
| S3 | Claude vs. OpenAI | 0.386 | 0.059 | 0.176 | — |
| S3 | Claude vs. Gemini | 4.061 | <0.001 | <0.001 | ✓ |
| S3 | OpenAI vs. Gemini | 1.866 | <0.001 | <0.001 | ✓ |
| S4 | Claude vs. OpenAI | 0.324 | 0.498 | 1.000 | — |
| S4 | Claude vs. Gemini | 0.542 | 0.035 | 0.104 | — |
| S4 | OpenAI vs. Gemini | 0.781 | 0.001 | 0.004 | ✓ |
| S5 | Claude vs. OpenAI | 0.750 | <0.001 | <0.001 | ✓ |
| S5 | Claude vs. Gemini | 0.351 | <0.001 | <0.001 | ✓ |
| S5 | OpenAI vs. Gemini | 0.037 | 0.001 | 0.003 | ✓ |

---

## Appendix C: S5 Instruction Conflict Breakdown by Conflict Type

**Table C1: System-prompt adherence rate by conflict type and provider**

| Conflict Type | Description | Claude | OpenAI | Gemini |
|--------------|-------------|--------|--------|--------|
| bullets_vs_prose | System: bullet points; User: flowing prose | 100% | 100% | 100% |
| disclaimer_vs_skip | System: include disclaimer; User: skip it | 0% | 33% | 0% |
| formal_vs_slang | System: formal English; User: casual slang | 100% | 100% | 100% |
| **Overall** | Weighted mean | **66.7%** | **77.8%** | **66.7%** |

The disclaimer_vs_skip conflict type accounts for all inter-provider quality differences in S5. Style-based conflicts (formatting and register) are uniformly resolved in favor of the system prompt by all three providers; content-based conflicts (inclusion of mandatory text) are uniformly resolved in favor of the user.

---

**Code repository:** https://github.com/mukund1985/llm-provider-failure-study  
**Results data:** `results_v2/experiment_v2_results_20260731_023039.json` (session 1), `results_v2_judge/experiment_v2_results_20260802_090939.json` (session 2, with response texts), `results_v2_judge/judge_scores_20260802_091501.json` (judge scores)  
**Experiment logs:** `experiment_v2.log`, `experiment_v2_judge.log`
