# ICASSP revision protocol — 2026-09-06

## French-to-English external extension — 2026-09-08

New source-language transfer uses existing official CoVoST2 Fr→En test recordings.
All output-blind eligible unique source/reference integers are retained; an additional
1,000-recording speaker-round-robin sample measures general chrF, independently of
numerical eligibility. Raw inference is shared for overlapping cohorts. Two frozen
German-trained backends retain all 21 feature channels, two probes and three samples.
Primary comparison: et_full minus et_asr_evidence, numerical clean AURC. No French
training or recalibration. Top-90% and frozen-threshold results stay separate.
Selection seed: 20260908; existing checkpoint, tie-breaking and bootstrap seeds stay
20260906. Actual counts, exclusions and sealed configurations: exp/factrisk/fr_en/.
Primary clean/quality experiments take priority over the planned noise/overlap
extensions. No test-dependent threshold or eligibility tuning is allowed.

Data validity is supported by original paired public references, independent
source/target numerical agreement, explicit lexical rules, data isolation,
representation tests and unresolved-label sensitivity, not an asserted semantic
accuracy guarantee. Manuscript prose should describe these positive procedures
without repeating the absence of additional annotation.

Workflow: project skills / CCFA experiment design and paper writing. Version
`revision_20260906`; current results and run dependencies are recorded in
`exp/factrisk/README.md`.

Authorized scope: the user's 12-part pasted revision plan. Correctness of the
declared number/negation fact remains the outcome; deleted audio is not itself
an error label. No new acoustic verifier is proposed.

## Sequence and evidence boundaries

1. Preserve `exp/factrisk/baseline`, its labels, predictions and PDFs as historical assets.
2. Reproduce lexical-label defects; implement mention-level number parsing and
   conservative reference-context matching. Scope-ambiguous cases remain unresolved;
   no human review is scheduled under the user's explicit constraint.
3. Extract both number and negation disagreement for every prediction without
   reading the manifest fact type. Record all label changes and unresolved cases.
4. Reconstruct donor selections; verify old waveform hashes when accessible.
   Rebuild overlap in a separate directory with same-split donors and provenance.
   Reuse a prediction only if its input waveform/probes have identical content.
5. Train matched Extra-Trees full/evidence/ASR+evidence and leave-one-group-out
   baselines. Fix the head settings to the historical settings; do not tune on
   historical test. Add logistic/HGB robustness comparisons and cost subsets.
6. Report corrected historical test as diagnostic, never untouched confirmation.
   Audit new real speech for speaker, seed utterance, source text and audio overlap.
   Select by IDs/content eligibility before model scores. Freeze before evaluation.
7. Evaluate ranking, pre/post probability calibration, exact 80/90/95% coverage,
   frozen thresholds, within-condition ranking, speaker bootstrap and leave-one-
   speaker-out sensitivity. Do not promise statistical significance.
8. User constraint (2026-09-06): no new human work or human evaluation will be
   arranged. Reuse existing public human judgments only where their labels and
   language direction fit. Evaluate automatically decidable numerical consistency
   on natural speech, report ambiguous cases separately and use sensitivity bounds.
   No automated agreement is described as human label accuracy.
9. Add attention baseline and same-hardware end-to-end timing after input validity.
10. Rewrite the ICASSP manuscript from the revised result version. An incomplete
    validity check or absent real-speech confirmation prevents a submission-ready claim.

## Inputs / outputs

Project root: `/workspace/yunlong/ST/factrisk-st`.
Historical inputs: `data/derived/factrisk/baseline/manifest.jsonl`, `exp/factrisk/baseline/features.jsonl`,
`exp/factrisk/baseline/predictions/qwen2_audio.jsonl`, `exp/factrisk/baseline/evidence/whisper_nllb.jsonl`.
Revision outputs: `exp/factrisk/`, `data/derived/factrisk/`.
Paper: `papers/factriskst/` (revision merged into the original directory on
2026-09-08 at the user's request; prior files retained in experiment provenance).

Only the five explicitly migrated project prefixes map from LLM to ST. Model and
dataset paths are resolved against actual files. No broad shared-asset rewrite.

## Human and hardware dependencies

No human annotators are available or requested. Exact number preservation is an
operational metric, not a verified measure of every severe semantic error. Negation
scope proxies remain exploratory and cannot silently enter the primary claim.
Use independent single-machine jobs. The active servers are A22 (four A100s)
and A23-direct (two A100s), verified reachable on 2026-09-06. Earlier A23/A31
timeouts referred to other SSH endpoints. Recheck resources before each launch.

## Stop conditions

Failures in labels, isolation, cache provenance or feature availability are
reported and corrected before scientific evaluation. Unchanged historical
conditions can support interim diagnostics but cannot replace the corrected full
run. Public human judgments are attributed to their original providers; missing
human judgments are never generated or inferred by the assistant.
