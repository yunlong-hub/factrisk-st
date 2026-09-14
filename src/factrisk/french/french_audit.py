"""Output-blind reference/label property checks for the French cohort."""
from collections import Counter
import re

from factrisk.french.mentions import french_mentions
from factrisk.core.io import read_jsonl, sha256_file, write_json
from factrisk.core.labels_v2 import label_fact
from factrisk.core.numbers import number_mentions
from factrisk.eval.reliability import spelled
from factrisk.pipeline.workflow import DATA, OUT


def reference_checks(row):
    """Check annotation mechanics, not source audio or semantic correctness."""
    reference = row['reference']
    mentions = number_mentions(reference)
    if len(mentions) != 1:
        return {'unique_reference': False}
    mention = mentions[0]
    n = int(row['expected_slot'])
    def replace(value):
        return reference[:mention['start']] + value + reference[mention['end']:]
    variants = {
        'reference_identity': (reference, 0),
        'digit_equivalence': (replace(str(n)), 0),
        'word_equivalence': (replace(spelled(n)), 0),
        'replacement_negative': (replace(str(n+1)), 1),
        'deletion_negative': (replace(''), 1),
    }
    return {name: label_fact(text, str(n), '', 'number', reference)['severe_fact_error'] == expected
            for name, (text, expected) in variants.items()}


def main():
    path = DATA / 'fr_en/clean/manifest.jsonl'
    digest = sha256_file(path)
    rows = [r for r in read_jsonl(path) if r['numeric_cohort']]
    passed, failed, details = Counter(), Counter(), []
    for row in rows:
        for name, ok in reference_checks(row).items():
            (passed if ok else failed)[name] += 1
            if not ok:
                details.append({'id': row['id'], 'check': name})
    patterns = {
        'fraction_word': r'\b(?:demi|tiers|quart|moitié)\b',
        'ordinal_digit_suffix': r'\d\s*(?:er|re|e|ème|ième)\b',
        'minus_word_or_sign': r'(?<!\w)[−-]\s*\d|\bmoins\b',
        'decimal_word': r'\bvirgule\b',
    }
    candidates = {name: [{'id': r['id'], 'source_text': r['source_text'], 'reference': r['reference']}
                          for r in rows if re.search(pattern, r['source_text'])]
                  for name, pattern in patterns.items()}
    if sha256_file(path) != digest:
        raise RuntimeError('Manifest changed while auditing; rerun against the frozen version')
    result = dict(status='passed' if not failed else 'failed', numeric_rows=len(rows),
                  manifest_sha256=digest, passed=dict(passed), failed=dict(failed), failures=details,
                  boundary_candidates=candidates,
                  scope='Reference identity, representation equivalence, and controlled numeric replacement/deletion only; not an estimate of audio correctness or semantic accuracy.',
                  reads_st_predictions=False)
    write_json(OUT/'fr_en/reference_property_audit.json', result)
    print(f'{len(rows)} rows; {sum(passed.values())} passed checks; {sum(failed.values())} failed checks')


if __name__ == '__main__':
    main()
