"""Conservative, auditable correctness labels; never infer audio sufficiency.

Unresolved numerical targets return None. Negation-scope proxies are marked
separately. In the no-new-annotation protocol these cases are bounded or kept
exploratory, not silently promoted to verified labels. References are allowed
here for labels only; this module must not feed risk features.
"""
from __future__ import annotations

import re
from factrisk.core.numbers import number_mentions
from factrisk.core.text import extract_numbers, normalize_text, slot_surfaces, text_is_negative

VERSION = 'critical_fact_v2_context_20260906'
STOP = set('a an the of to in on at by for with and or is are was were be been it its this that there he she they we i you as from'.split())


def anchors(text: str) -> set[str]:
    return set(re.findall(r'[a-z]{2,}', text.lower())) - STOP


def context(text: str, mention: dict, window: int = 5) -> set[str]:
    before = text[:mention['start']].split()[-window:]
    after = text[mention['end']:].split()[:window]
    return anchors(' '.join(before + after))


def label_fact(translation: str, expected, contrast, fact_type: str, reference: str) -> dict:
    result = dict(label_rule_version=VERSION, severe_fact_error=None,
                  contrast_value_present=None, needs_human_review=True,
                  label_status='unresolved', label_reason='unresolved')
    if fact_type == 'number':
        values = set().union(*(extract_numbers(s) for s in slot_surfaces(expected)))
        alternatives = set().union(*(extract_numbers(s) for s in slot_surfaces(contrast)))
        ref_mentions = number_mentions(reference)
        targets = [m for m in ref_mentions if m['value'] in values]
        observed = number_mentions(translation)
        if len(targets) != 1:
            result['label_reason'] = 'reference_target_not_unique'
            return result
        if not observed:
            chosen = None
        elif len(ref_mentions) == 1 and len(observed) == 1:
            chosen = observed[0]
        else:
            target_context = context(reference, targets[0])
            scores = [(len(target_context & context(translation, m)), i) for i, m in enumerate(observed)]
            scores.sort(reverse=True)
            if not scores or scores[0][0] == 0 or (len(scores) > 1 and scores[0][0] == scores[1][0]):
                result['label_reason'] = 'number_target_context_ambiguous'
                return result
            chosen = observed[scores[0][1]]
        value = chosen['value'] if chosen else None
        result.update(severe_fact_error=int(value not in values),
                      contrast_value_present=bool(value in alternatives and value not in values),
                      needs_human_review=False, label_status='automatic_rule',
                      label_reason='target_number_match' if value in values else 'wrong_or_missing_target_number')
        return result
    if fact_type == 'negation':
        # Exact reference is safe for reference correctness (not acoustic support).
        if normalize_text(translation) == normalize_text(reference):
            result.update(severe_fact_error=0, contrast_value_present=False,
                          needs_human_review=False, label_status='automatic_rule',
                          label_reason='exact_reference')
            return result
        # Whole-sentence polarity remains explicitly provisional. It cannot
        # disambiguate scope/paraphrase; no proxy may enter a human-confirmed table.
        result.update(severe_fact_error=int(text_is_negative(translation) != text_is_negative(reference)),
                      contrast_value_present=None, label_status='scope_proxy_requires_review',
                      label_reason='reference_polarity_proxy')
        return result
    result['label_reason'] = 'unsupported_fact_type'
    return result
