"""Build blinded number-label annotation sheets and score returned sheets.

`build` writes two independent copies of the same shuffled sample, one per
annotator. Sheets expose only source text, reference, hypothesis, and the
expected number slot; the automatic label, the fact type, and every risk score
are withheld so the judgement is blind.

`score` reads the returned sheets and reports three counts: how many rows were
filled, how many agree with the automatic normalizer, and how many agree across
the two annotators.

Judgement rule written for annotators: report 1 when the number carried by the
reference is not preserved in the hypothesis (altered, dropped, or invented),
and 0 when it is preserved, including synonymous rewording. Judge meaning, not
the normalizer's literal-mention rule; disagreements with it are a finding, not
an error.
"""
import argparse
import csv
import json
import random
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1] / 'exp/factrisk/audit'
SHEET = ['blind_id', 'source_text', 'reference', 'translation', 'expected_slot',
         'human_number_error']
SHEETS = {group: AUDIT / f'annotation_number_{group}.csv' for group in ('a', 'b')}


def load():
    labels = {}
    for line in open(AUDIT / 'label_changes.jsonl'):
        if line.strip():
            row = json.loads(line)
            labels[row['id']] = row
    sample = [json.loads(l) for l in open(AUDIT / 'human_sampling_key_private.jsonl')
              if l.strip()]
    return labels, sample


def build(seed=20260913):
    """Write one blank sheet per annotator over the shuffled number sample."""
    labels, sample = load()
    rows = [r for r in sample if labels[r['id']]['fact_type'] == 'number']
    random.Random(seed).shuffle(rows)
    for group, path in SHEETS.items():
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(SHEET)
            for row in rows:
                writer.writerow([row['blind_id'], row['source_text'], row['reference'],
                                 row['translation'], row['expected_slot'], ''])
        print(f'wrote {path}  rows={len(rows)}  group={group}')


def read_sheet(path):
    """Return {blind_id: judgement} for filled rows only."""
    if not path.exists():
        return {}
    filled = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            value = (row.get('human_number_error') or '').strip()
            if value in ('0', '1'):
                filled[row['blind_id']] = int(value)
    return filled


def score():
    """Report fill count, agreement with the normalizer, and inter-annotator agreement."""
    labels, sample = load()
    truth = {r['blind_id']: labels[r['id']]['previous_error'] for r in sample}
    sheets = {g: read_sheet(p) for g, p in SHEETS.items()}

    for group, filled in sheets.items():
        if not filled:
            print(f'group {group}: no filled rows yet')
            continue
        agree = sum(1 for k, v in filled.items() if truth.get(k) == v)
        print(f'group {group}: filled={len(filled)} agree_with_normalizer={agree} '
              f'rate={agree/len(filled):.4f}')

    both = set(sheets['a']) & set(sheets['b'])
    if both:
        agree = sum(1 for k in both if sheets['a'][k] == sheets['b'][k])
        print(f'inter-annotator: n={len(both)} agree={agree} rate={agree/len(both):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'score'), nargs='?', default='build')
    args = parser.parse_args()
    build() if args.action == 'build' else score()
