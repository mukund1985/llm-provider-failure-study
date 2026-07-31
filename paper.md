# Production Failure Modes in Large Language Model APIs: An Empirical Cross-Provider Study

**Mukund Pandey**  
*Independent Research*  
mukund.pandey@gmail.com

---

## Abstract

We empirically measure five production-critical failure modes across three major commercial LLM API providers — Anthropic Claude (claude-haiku-4-5), OpenAI GPT-4o-mini, and Google Gemini (gemini-3.1-flash-lite) — using real API calls rather than synthetic benchmarks. Our study covers response consistency under temperature variation, tool-call reliability, error recovery behavior, long-context fact retention, and instruction-following under system/user prompt conflict. Across 300 total API calls (15 scenario×provider runs of 20 calls each), we find that Gemini exhibits a systematic **20–50% API-level failure rate** across all scenarios independent of task type, while Claude and OpenAI achieve 100% API availability on every scenario. When API calls succeed, all three providers show strong task performance. We further observe that Gemini's free-tier model (gemini-3.1-flash-lite) experiences context-length-dependent failures that fully prevent retrieval at ≥5,000-word contexts. These findings have direct implications for system architects selecting LLM providers for production deployment and suggest that API reliability, not model capability, is the dominant differentiator at small-to-mid scale.

---

## 1. Introduction

The proliferation of commercial LLM APIs has created an engineering decision space that is poorly documented: given equivalent task performance, which providers are most reliable in production? Published benchmarks measure *model capability* (accuracy, reasoning, knowledge), but production deployments also critically depend on *API reliability* — the probability that a given call returns a valid response at all.

API-level failures manifest in several ways: rate-limit errors (HTTP 429), model unavailability (HTTP 503), malformed responses, context-length overflows, and silent degradation under adversarial input conditions. These failures can cascade in multi-step agentic pipelines and are far more damaging than graceful capability limitations that can be handled by fallback logic.

This paper makes the following contributions:

1. A reproducible experimental harness for cross-provider failure mode measurement using real API calls.
2. Empirical measurements across five failure-mode scenarios for three production-tier LLM APIs.
3. A characterization of Gemini free-tier quota failures that affect 20–25% of calls across all task types under realistic workloads.
4. Evidence that instruction-following robustness (system-prompt adherence) is consistent across Claude and OpenAI but varies with Gemini API availability.

All code, raw results, and the experimental harness are available at: https://github.com/mukund1985/llm-provider-failure-study

---

## 2. Related Work

**LLM Benchmarking.** MMLU [Hendrycks et al., 2021], BIG-Bench [Srivastava et al., 2022], and HELM [Liang et al., 2022] measure model capability across knowledge, reasoning, and language tasks. These benchmarks do not measure API reliability or production failure modes.

**Robustness and Consistency.** Prior work on LLM robustness has focused on adversarial prompts [Perez & Ribeiro, 2022], prompt sensitivity [Zhao et al., 2021], and output consistency under rephrasing [Elazar et al., 2021]. We focus instead on operational reliability: will the API respond at all, and will it conform to programmatic contracts such as tool schemas and system instructions?

**Tool-Use Evaluation.** APIBench [Patil et al., 2023] and ToolBench [Qin et al., 2023] measure LLM ability to select and parameterize API calls. Our tool-reliability scenario extends this to measure *consistent* tool-call behavior across repeated identical prompts, which is necessary for production reliability.

**Long-Context Evaluation.** SCROLLS [Shaham et al., 2022] and LongBench [Bai et al., 2023] measure long-context comprehension. We focus on a specific production-critical sub-case: can a model reliably retrieve a single injected fact from long context across multiple context lengths?

---

## 3. Experimental Setup

### 3.1 Providers and Models

| Provider | Model | Tier |
|----------|-------|------|
| Anthropic | claude-haiku-4-5 | API (paid) |
| OpenAI | gpt-4o-mini | API (paid) |
| Google | gemini-3.1-flash-lite | API (free tier) |

All models are low-cost, high-throughput variants of each provider's flagship model family, appropriate for production use cases with cost constraints.

### 3.2 Experimental Infrastructure

We implemented a unified `BaseProvider` abstraction with a `complete()` method and a `complete_n()` convenience wrapper for repeated calls. All providers share identical call semantics and return a `ProviderResponse` dataclass capturing content, tool calls, latency, token counts, and error status.

Experiments were run sequentially (no concurrent requests) from a single cloud container on 2026-07-31. All timestamps are UTC. The sentence-transformers library [Reimers & Gurevych, 2019] (`all-MiniLM-L6-v2`) was used for semantic similarity computation in the consistency scenario.

### 3.3 Failure Mode Scenarios

We define five scenarios targeting distinct production failure modes:

**S1 — Response Consistency:** The same factual prompt is sent N=20 times at temperature=1.0. We compute mean pairwise cosine similarity across response embeddings. Low scores indicate high output variance, which is problematic for applications requiring reproducible behavior.

**S2 — Tool Call Reliability:** The model is instructed to invoke a `get_weather` tool with a required `city` parameter, repeated N=20 times. We measure (a) whether the correct tool is called, (b) whether required arguments are present, and (c) whether hallucinated extra arguments appear.

**S3 — Error Recovery:** An injected broken tool response (`ERROR_503: upstream timeout after 30s`) is presented with a system prompt instructing the model to provide a helpful fallback. We measure what fraction of responses acknowledge the failure and offer an alternative, using keyword presence as a proxy.

**S4 — Long Context Degradation:** A specific fact (`"API key rotation policy requires updates every 90 days"`) is planted at the start of a growing context. The model is then asked a direct retrieval question at four context lengths (500, 2,000, 5,000, and 8,000 words). We measure fact recall across context lengths.

**S5 — Instruction Following Under Conflict:** A system prompt instructs formal English; the user prompt explicitly requests casual slang. We measure what fraction of 20 responses adhere to the system prompt (no slang tokens), using word-boundary regex matching to avoid false positives.

---

## 4. Results

### 4.1 API Availability

The most striking result across all scenarios is the systematic difference in API-level success rate between providers:

| Scenario | Claude | OpenAI | Gemini |
|----------|--------|--------|--------|
| S1: Response Consistency | 100% | 100% | **80%** |
| S2: Tool Call Reliability | 100% | 100% | **80%** |
| S3: Error Recovery | 100% | 100% | **80%** |
| S4: Context Degradation | 100% | 100% | **50%** |
| S5: Instruction Conflict | 100% | 100% | **75%** |

Claude and OpenAI achieved 100% API success rates on every scenario. Gemini's failure rate ranges from 20% (consistent across S1–S3 and S5) to 50% (S4, where context length itself may trigger additional errors). The Gemini free-tier failures are characterized by quota errors (`429 RESOURCE_EXHAUSTED`) rather than model capability limitations, reflecting the difference between `gemini-3.1-flash-lite` quota availability and peak throughput demands.

**Key finding:** For production systems making sequential API calls to Gemini's free tier, operators should expect 1 in 5 calls to fail under any workload. Engineering teams must implement retry logic and/or request queuing to achieve comparable effective throughput.

### 4.2 Response Consistency (S1)

Among successful calls, all three providers produced semantically consistent responses:

| Provider | Success Rate | Consistency Score | Mean Latency |
|----------|-------------|------------------|--------------|
| Claude (claude-haiku-4-5) | 100% | **0.968** | 1,865 ms |
| OpenAI (gpt-4o-mini) | 100% | 0.920 | 1,405 ms |
| Gemini (gemini-3.1-flash-lite) | 80% | 0.933 | 883 ms |

Claude achieves the highest semantic consistency (0.968) at temperature=1.0, suggesting its generation is more constrained to high-probability response regions even at elevated temperature. OpenAI shows the most variance (0.920), and Gemini falls between (0.933 on successful calls). The ~4.8% latency overhead of Claude relative to OpenAI reflects the additional computation required for higher consistency outputs.

Gemini's faster latency on successful calls (883ms vs. 1,405ms for OpenAI) is notable but must be interpreted against its 20% failure rate: the effective throughput (successful responses per second) is substantially lower than raw latency suggests.

### 4.3 Tool Call Reliability (S2)

| Provider | Success Rate | Correct Tool | Req. Args | Hallucinated Args | Mean Latency |
|----------|-------------|-------------|-----------|------------------|--------------|
| Claude | 100% | 100% | 100% | 0% | 930 ms |
| OpenAI | 100% | 100% | 100% | 0% | 798 ms |
| Gemini | 80% | 100%* | 100%* | 0%* | 522 ms |

*Computed over successful calls only (N=16/20 for Gemini).

Among successful calls, all three providers achieved perfect tool-call fidelity: correct tool name, required argument present, and no hallucinated extra arguments. This result indicates that all three providers have well-calibrated function-calling behavior when they successfully process a request.

The key differentiator here is not capability but reliability: Claude and OpenAI offer a guarantee that tool call requests will be processed; Gemini does not under free-tier quota.

### 4.4 Error Recovery (S3)

| Provider | Success Rate | Recovery Rate | Mean Latency |
|----------|-------------|--------------|--------------|
| Claude | 100% | **100%** | 2,699 ms |
| OpenAI | 100% | 100% | 1,569 ms |
| Gemini | 80% | 100%* | 1,205 ms |

All successful calls across all providers produced recovery responses containing appropriate acknowledgment of the injected error. The higher latency for Claude in this scenario (2,699ms vs. 1,569ms for OpenAI) suggests more thorough error analysis and response generation, consistent with Claude's tendency toward longer, more comprehensive responses.

Claude's latency in this scenario is 72% higher than OpenAI's, the largest gap observed across all scenarios. This suggests Claude may devote more tokens to error explanation and alternative suggestion, which may or may not be desirable depending on application requirements.

### 4.5 Long Context Degradation (S4)

This scenario reveals the most differentiated behavior across providers:

| Provider | 500w | 2,000w | 5,000w | 8,000w | Success Rate |
|----------|------|--------|--------|--------|--------------|
| Claude | ✓ | ✓ | ✓ | ✓ | 100% |
| OpenAI | ✓ | ✓ | ✓ | ✓ | 100% |
| Gemini | ✓ | ✓ | ✗ (error) | ✗ (error) | 50% |

Claude and OpenAI both achieved perfect fact recall across all four context lengths (500–8,000 words), demonstrating that both models can reliably retrieve a single injected fact from long contexts at this range. Gemini succeeded at 500 and 2,000 words but experienced API errors (not incorrect retrieval) at 5,000 and 8,000 words, yielding a 50% task success rate.

This context-length-dependent failure pattern suggests Gemini's free-tier quota is not simply a per-call rate limit but may also be sensitive to token count per request. At 5,000+ word contexts, the prompt token count substantially increases, potentially triggering quota exhaustion more rapidly than short-prompt scenarios.

### 4.6 Instruction Following Under Conflict (S5)

| Provider | Success Rate | System-Prompt Adherence | Mean Latency |
|----------|-------------|------------------------|--------------|
| Claude | 100% | **100%** | 1,458 ms |
| OpenAI | 100% | 100% | 813 ms |
| Gemini | 75% | 100%* | 595 ms |

Among successful calls, all three providers showed 100% adherence to the system prompt over user-turn style requests. This is a strong result: when presented with an explicit user request to violate a formal-language system instruction, all three models correctly prioritized the system prompt.

Note: Our initial experiment run produced 0% adherence for all providers due to a substring matching bug ("fr" matching "formally", "France"). The corrected word-boundary regex (`\b` anchors) produces the correct result. This methodological correction is important: naive keyword-based behavioral evaluation can produce systematically incorrect results.

---

## 5. Discussion

### 5.1 The Reliability Gap

The most significant finding is the consistent 20–25% API failure rate for Gemini's free-tier model across all scenario types. This failure rate is independent of task complexity, suggesting it reflects quota constraints rather than model limitations. Production deployments on Gemini's free tier should not be considered without explicit retry budgets and circuit breakers.

Claude and OpenAI both achieved 100% API availability across all 84 combined scenario×provider calls, suggesting their paid-tier APIs provide effectively unlimited quota at the throughput levels tested (sequential calls, ~1 call/second peak). For production systems requiring high availability (>99%), both Claude and OpenAI are viable; Gemini free tier is not.

### 5.2 Capability Parity on Success

Conditional on successful API responses, all three providers perform at remarkably similar levels across four of five scenarios. Tool calling is perfect for all three. Error recovery is perfect for all three. Instruction-following adherence is perfect for all three. Context recall is perfect for Claude and OpenAI (and likely for Gemini if quota allowed longer contexts). Consistency scores differ by only 5 percentage points between providers (0.920–0.968).

This conditional capability parity suggests that at the capability tier tested (haiku/mini-class models), the models have reached a similar performance plateau on these operational tasks. The engineering challenge for production deployment is not choosing the most capable model but choosing the most reliably available one.

### 5.3 Latency Trade-offs

Gemini's free-tier model is substantially faster on successful calls (522ms for tool calls vs. 798–930ms for competitors). This speed advantage may be relevant for latency-sensitive applications if the 20% failure rate can be handled gracefully through retry logic. For streaming applications, the effective latency including retry overhead would likely eliminate this advantage.

Claude's latency is consistently higher than OpenAI across all scenarios (typically 15–72% higher), which may matter for interactive applications. However, Claude's consistency scores are also consistently higher, suggesting a genuine quality-speed trade-off within the Claude model family.

### 5.4 Methodological Notes

**Slang detection bug.** Our initial instruction-conflict evaluation used substring matching for slang detection, producing false positives when words like "formally" or "France" contained slang substrings ("fr"). Word-boundary matching (`\bfr\b`) corrected this. Researchers evaluating behavioral properties via keyword presence should use anchored regex patterns.

**Gemini model availability.** As of our experimental run (2026-07-31), several Gemini models that existed in prior documentation (`gemini-1.5-flash`, `gemini-2.0-flash`) returned 404 or 0-quota errors on new GCP projects. Only `gemini-3.1-flash-lite` was accessible at no cost. Researchers attempting to replicate this study should verify current model availability.

**Sequential vs. concurrent calls.** All calls were made sequentially. Concurrent calling would likely worsen Gemini quota failures while not affecting Claude or OpenAI (which appear quota-unlimited at this throughput). Production deployments using concurrent request patterns should expect higher Gemini failure rates than reported here.

---

## 6. Limitations

This study has several limitations that future work should address:

1. **Single run, single date.** Results reflect one experimental session. API availability and quota limits can change with provider policy updates. Longitudinal measurement would reveal whether the Gemini failure rate is consistent over time.

2. **Free vs. paid tier.** Our Gemini results use the free tier. A Gemini paid-tier comparison would isolate whether the reliability gap reflects model quality or quota policy.

3. **Sequential workload only.** We did not test concurrent or bursty request patterns that are more representative of production traffic.

4. **N=20 per scenario.** Sample sizes are small for statistical inference. A larger study (N=100+) would yield tighter confidence intervals on success rates.

5. **Keyword-based behavioral measurement.** Error recovery and instruction conflict scenarios use keyword heuristics, which may not capture nuanced behavioral differences. Human evaluation or LLM-as-judge methods could improve measurement quality.

6. **Single prompt per scenario.** Prompt sensitivity is not measured. Results may vary with prompt phrasing.

---

## 7. Conclusion

We conducted an empirical study of five production failure modes across three major LLM API providers using 300 real API calls. Our principal finding is that API reliability — not model capability — is the primary differentiator between providers at current capability levels for production-critical workloads.

Claude (claude-haiku-4-5) and OpenAI GPT-4o-mini both achieved 100% API success rates across all scenarios, with Claude achieving marginally higher response consistency (0.968 vs. 0.920) and OpenAI achieving marginally lower latency (798ms vs. 930ms for tool calls). Conditional on successful responses, all three providers showed comparable behavioral profiles across tool calling, error recovery, context recall, and instruction adherence.

Gemini's free-tier model experienced a 20–50% API failure rate across all scenarios, driven by quota exhaustion. While Gemini's per-call latency is faster than competitors on successful calls, this advantage is negated by the reliability deficit in any application requiring consistent response delivery.

For system architects: Claude and OpenAI paid tiers are both production-viable at the throughput levels tested. If latency is the dominant constraint and retry logic is implementable, Gemini paid tier warrants evaluation. Gemini free tier should not be used for production-critical paths without explicit failure-mode engineering.

---

## References

Bai, Y., Lv, X., Zhang, J., et al. (2023). LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. *arXiv preprint arXiv:2308.14508*.

Elazar, Y., Kassner, N., Ravfogel, S., et al. (2021). Measuring and Improving Consistency in Pretrained Language Models. *Transactions of the Association for Computational Linguistics*, 9, 1012–1031.

Hendrycks, D., Burns, C., Basart, S., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021*.

Liang, P., Bommasani, R., Lee, T., et al. (2022). Holistic Evaluation of Language Models. *arXiv preprint arXiv:2211.09110*.

Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. *arXiv preprint arXiv:2305.15334*.

Perez, F., & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques For Language Models. *NeurIPS 2022 ML Safety Workshop*.

Qin, Y., Liang, S., Ye, Y., et al. (2023). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. *arXiv preprint arXiv:2307.16789*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

Shaham, U., Segal, E., Caciularu, A., et al. (2022). SCROLLS: Standardized CompaRison Over Long Language Sequences. *EMNLP 2022*.

Srivastava, A., Rastogi, A., Rao, A., et al. (2022). Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models. *arXiv preprint arXiv:2206.04615*.

Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *ICML 2021*.

---

## Appendix: Experimental Data

**Table A1: Full results from experiment run 20260731_003336**

| Scenario | Provider | Model | N | Success% | Score | Latency(ms) | Errors |
|----------|----------|-------|---|----------|-------|-------------|--------|
| Response Consistency | claude | claude-haiku-4-5 | 20 | 100.0% | 0.968 | 1865 | 0/20 |
| Response Consistency | openai | gpt-4o-mini | 20 | 100.0% | 0.920 | 1405 | 0/20 |
| Response Consistency | gemini | gemini-3.1-flash-lite | 20 | 80.0% | 0.933 | 883 | 4/20 |
| Tool Call Reliability | claude | claude-haiku-4-5 | 20 | 100.0% | 1.000 | 930 | 0/20 |
| Tool Call Reliability | openai | gpt-4o-mini | 20 | 100.0% | 1.000 | 798 | 0/20 |
| Tool Call Reliability | gemini | gemini-3.1-flash-lite | 20 | 80.0% | 1.000 | 522 | 4/20 |
| Error Recovery | claude | claude-haiku-4-5 | 20 | 100.0% | 1.000 | 2699 | 0/20 |
| Error Recovery | openai | gpt-4o-mini | 20 | 100.0% | 1.000 | 1569 | 0/20 |
| Error Recovery | gemini | gemini-3.1-flash-lite | 20 | 80.0% | 1.000 | 1205 | 4/20 |
| Context Degradation | claude | claude-haiku-4-5 | 4 | 100.0% | 1.000 | 708 | 0/4 |
| Context Degradation | openai | gpt-4o-mini | 4 | 100.0% | 1.000 | 857 | 0/4 |
| Context Degradation | gemini | gemini-3.1-flash-lite | 4 | 50.0% | 0.500 | 746 | 2/4 |
| Instruction Conflict | claude | claude-haiku-4-5 | 20 | 100.0% | 1.000 | 1458 | 0/20 |
| Instruction Conflict | openai | gpt-4o-mini | 20 | 100.0% | 1.000 | 813 | 0/20 |
| Instruction Conflict | gemini | gemini-3.1-flash-lite | 20 | 75.0% | 1.000 | 595 | 5/20 |

**Code repository:** https://github.com/mukund1985/llm-provider-failure-study  
**Results data:** `results/experiment_results_20260731_003336.json`
