# FactRisk-ST

**Reference-free risk estimation for number preservation in speech translation.**

*Anonymous code release for double-blind review.* &nbsp;|&nbsp; **English** · [简体中文](README.zh-CN.md)

**9 cohorts · 2 backends · 5 of 9 paired intervals exclude zero · one frozen protocol**

---

![FactRisk-ST overview: frozen translation paths, complementary risk signals, and the risk-and-selection stage](assets/method_overview.svg)

## The argument in one page

- **Problem.** Sentence-level metrics score a whole output, so one substituted number is diluted in the aggregate. They neither verify a local number nor support an accept-or-abstain decision.
- **What we build.** A frozen ST model plus a reference-free risk head that estimates the probability that the reference number was not preserved, over four signal families; retained error is reported with identification bounds that keep unresolved labels in the denominator.
- **What we find.** The recognition–translation path carries the ranking in eight of nine cohorts. Ranking transfers across languages and backends; the calibrated threshold does not, because calibration encodes source-domain prevalence.
- **What it means.** The head deploys as fitted. Only the threshold needs reselection, and that needs target-domain risk scores, not labels.

## 1. The problem

A speech translation (ST) system can render "sixteen" as "seventeen" and still
return a fluent sentence whose remaining content is correct. Numbers carry the
message in subtitle post-editing, meeting transcription and spoken information
access, where readers see only the translation, so a silently altered number
passes unnoticed.

Sentence-level metrics — BLEU, chrF, COMET, learned quality estimation — score a
whole output. A single substituted number is diluted in the aggregate, and none
of them answers the two questions a downstream user actually has: *is this
particular number faithful?* and *should I accept this segment or abstain?*
Selective classification supplies the risk–coverage framework for the second
question, but not the signals that predict number failure in ST.

## 2. Approach

FactRisk-ST scores a single translated segment without a reference translation.
A frozen ST model produces the translation; an Extra-Trees head then estimates
the probability that the reference number was not preserved, from four signal
families:

| Family | Example features | What it captures |
|---|---|---|
| Text / decoder | `sequence_nll`, `token_entropy`, `beam_margin_risk`, `output_length` | uncertainty of the generated translation |
| Acoustic | `asr_sequence_nll`, `asr_token_entropy`, `audio_spectral_entropy`, `audio_rms_db`, ... | whether the audio itself is hard |
| Perturbation stability | `perturb_mean_distance`, perturbation max distance, sampling disagreement | sensitivity of the output to small input changes |
| Recognition–translation path | `cascade_sequence_nll`, `evidence_text_distance`, `evidence_fact_mismatch`, `evidence_coverage_gap` | agreement between the direct translation and an independent recognize-then-translate path |

Retained error is reported as **identification bounds**: segments whose reference
number cannot be resolved stay in the denominator instead of being dropped.

The decision threshold is a quantile of the risk score, so reselecting an
operating point on a new domain needs target-domain scores only, no labels.
Probability calibration, by contrast, is fitted on the source domain and does
not carry over.

## 3. Main results

Nine cohorts, two backends (Qwen2-Audio-7B-Instruct, SeamlessM4T-v2-large), one
frozen protocol: no model is trained or tuned on any evaluation cohort.

### Main comparison

Lower AURC is better. R90 is the identification bound [E/K, (E+U)/K] on retained
error at top-90% selection (K = 643 controlled, 488 natural) rather than a
confidence interval; bold marks the best AURC in each cohort.

| Method | Controlled AURC | Controlled R90 (%) | Natural clean AURC | Natural clean R90 (%) |
|---|---:|---:|---:|---:|
| *Qwen2-Audio* | | | | |
| Beam NLL summary | 0.3080 | 29.24–37.48 | 0.0252 | 5.12–7.17 |
| Hypothesis NLL | 0.2582 | 28.77–37.01 | 0.0235 | 4.92–6.76 |
| Token entropy | 0.2320 | 28.62–36.86 | 0.0214 | 4.71–6.76 |
| COMET-QE | 0.2234 | 25.82–33.90 | 0.0386 | 5.12–6.97 |
| Evidence ET | 0.1652 | 26.13–34.06 | 0.0163 | 1.64–2.05 |
| Attention LR | 0.1466 | 24.57–32.97 | 0.0263 | 3.07–5.12 |
| ASR+evidence ET | 0.1425 | 25.66–33.59 | 0.0148 | 1.84–2.25 |
| **FactRisk-ST** | **0.1223** | 25.82–33.75 | **0.0092** | 1.64–2.05 |
| *SeamlessM4T-v2* | | | | |
| Evidence ET | 0.2592 | 25.51–32.97 | 0.0045 | 0.41–0.61 |
| Beam NLL summary | 0.2167 | 26.91–34.37 | 0.0115 | 1.43–1.64 |
| ASR+evidence ET | 0.2097 | 26.75–34.06 | **0.0040** | 0.41–0.61 |
| **FactRisk-ST** | **0.1623** | 24.42–31.88 | 0.0064 | 0.82–1.02 |

![Risk–coverage curves: German controlled (714 inputs) and German natural (542 inputs)](assets/risk_coverage.svg)

### Significance across nine cohorts

Delta is the AURC of FactRisk-ST minus the AURC of the strongest baseline
(ASR+evidence Extra Trees); negative is better, clusters are speaker–donor
connected components.

![ΔAURC against the strongest baseline, with paired 95% intervals](assets/forest.svg)

| Cohort | ΔAURC | 95% interval | Clusters |
|---|---:|---|---:|
| Qwen controlled diagnostic | -0.0202 | [-0.0329, -0.0098] | 7 |
| Qwen German stress | -0.0118 | [-0.0222, -0.0024] | 235 |
| Seamless controlled diagnostic | -0.0473 | [-0.0768, -0.0198] | 7 |
| Qwen French stress | -0.0311 | [-0.0410, -0.0215] | 371 |
| Seamless French stress | -0.0538 | [-0.0651, -0.0432] | 371 |
| Seamless German clean | +0.0024 | [-0.0019, +0.0091] | 506 |
| Qwen German clean | -0.0057 | [-0.0145, +0.0004] | 506 |
| Qwen French clean | -0.0002 | [-0.0061, +0.0064] | 841 |
| Seamless French clean | -0.0049 | [-0.0160, +0.0030] | 841 |

Five of nine intervals exclude zero; four do not, and on Seamless German clean
speech the point estimate favours the baseline. We therefore claim
**domain-dependent gains, not universal superiority**.

### Two findings that hold across the grid

- **Ranking transfers, the threshold does not.** The recognition–translation path
  carries the ranking in eight of nine cohorts. Ordering is stable across
  languages and backends, but a threshold fitted on controlled German speech —
  which accepts 99.6% of natural German clean speech — can raise Seamless
  French-stress risk by 4.6–7.4 points.
- **Selection tightens retained error where it matters.** On natural German clean
  speech, top-90% selection lowers the retained-error bound from 5.17–7.38% to
  1.64–2.05%. On the largest stress cohort (1,626 inputs, 472 known errors) the
  same label-free rule gives 24.95–28.50% against the baseline's 24.88–28.57%, a
  tie in the point estimate.

### French transfer

Frozen transfer with no French-specific tuning. F/B is FactRisk-ST against
ASR+evidence; intervals are paired 95% CIs for the AURC difference over
speaker–donor components.

| Setting | AURC F/B | Δ 95% CI |
|---|---|---|
| Qwen clean | 0.0193 / 0.0195 | [-0.0061, 0.0064] |
| Seamless clean | 0.0153 / 0.0202 | [-0.0160, 0.0030] |
| Qwen stress | 0.2291 / 0.2601 | [-0.0410, -0.0215] |
| Seamless stress | 0.1795 / 0.2333 | [-0.0651, -0.0432] |

Full per-cohort numbers, the paired significance summary and provenance records
ship with the evaluation package accompanying the paper.

## 4. Scope and limitations

- The method targets **one locally checkable property**: number preservation.
  Other factual errors, fluency and overall adequacy are outside its estimate.
- Labels come from an automatic pipeline audited for consistency, not from a new
  human study; all "real" figures are operational.
- Gains are domain-dependent: intervals on natural clean speech mostly include
  zero, and the controlled diagnostic carries only 7 clusters, so those
  comparisons are small-sample and high-variance.
- Cost: the full serial pipeline runs at roughly 3.9x ST-only decoding, measured
  as the sum of stages on the same 64 recordings, excluding model loading and
  scheduling overhead.
- The method operates on top of a frozen ST model and does not modify the
  translator.

## 5. Repository layout

- `src/factrisk/`: the only maintained Python package, grouped by responsibility:
  `core/` (IO, artifact contracts, label and number primitives, model registry,
  sharding, frozen-input sealing), `backends/` (Qwen2-Audio, SeamlessM4T and
  Whisper adapters, audio transforms, attention probes, shard runners),
  `datasets/`, `method/` (risk features and matched heads), `eval/` (metrics,
  statistical summaries, robustness gates, readiness), `pipeline/` (end-to-end
  stages), `french/` (French transfer subproject), `paper/`, `cli/`.
- `configs/paper.yaml`: the full experiment specification — risk feature groups,
  primary model and model selection, augmentation conditions.
- `docs/`: evaluation protocol, automatic-evaluation policy and runbook.
- `scripts/`: stage, experiment, French, smoke and paper entry points.
- `tests/`: verification grouped like `src/factrisk`.

## 6. Environment and entry points

```bash
python -m pip install -e . --no-deps

bash scripts/run_stage.sh audit          # a single stage
bash scripts/run_experiments.sh          # the full experiment workflow
bash scripts/smoke.sh                    # smoke self-check
```

The completed experiment should not be rerun unless its frozen protocol is
intentionally replaced.

## 7. Not included in this repository

For size, licensing and anonymity during review, this release omits: raw and
derived audio data, per-sample predictions and frozen evaluation artifacts, the
manuscript LaTeX sources and build outputs, and all backend model weights.
