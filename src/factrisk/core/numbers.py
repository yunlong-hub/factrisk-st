"""Versioned English number normalization; offsets refer to the input string.

This is a lexical normalizer, not a semantic fact or measurement-unit checker.
Adjacent independent numbers are kept separate (e.g. 'one or two').
"""
from __future__ import annotations

import re
from decimal import Decimal

UNITS = dict(zip(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),
    range(20),
))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
ORDINALS = dict(zip('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth'.split(), range(1, 20)))
ORDINALS.update(dict(zip('twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth'.split(), range(20, 100, 10))))
SCALES = {'hundred': 100, 'thousand': 1000, 'million': 10**6, 'billion': 10**9}
SCALE_ORDINALS = {key + 'th': val for key, val in SCALES.items()}
ALIASES = {'once': 1, 'twice': 2, 'dozen': 12}
TOKEN = re.compile(r'(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:st|nd|rd|th)?(?!\w)|[a-z]+', re.I)


def canonical(value: str | int | Decimal) -> str:
    return format(Decimal(str(value)).normalize(), 'f')


def number_mentions(text: str) -> list[dict]:
    tokens = list(TOKEN.finditer(text))
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i].group().lower()
        start = i
        if re.match(r'[+-]?\d', token):
            value = re.sub(r'(st|nd|rd|th)$', '', token).replace(',', '')
            result.append(dict(value=canonical(value), start=tokens[i].start(), end=tokens[i].end()))
            i += 1
            continue
        if token in ALIASES:
            result.append(dict(value=str(ALIASES[token]), start=tokens[i].start(), end=tokens[i].end()))
            i += 1
            continue
        negative = token in {'minus', 'negative'}
        if negative:
            i += 1
        if i >= len(tokens) or tokens[i].group().lower() not in UNITS | TENS | ORDINALS | SCALES | SCALE_ORDINALS:
            i = start + 1
            continue
        total = group = 0
        previous = None
        seen_scale = False
        while i < len(tokens):
            word = tokens[i].group().lower()
            if i > start and not re.fullmatch(r'[\s-]*', text[tokens[i-1].end():tokens[i].start()]):
                break
            if word == 'and' and seen_scale and i+1 < len(tokens) and tokens[i+1].group().lower() in UNITS | TENS | ORDINALS:
                i += 1
                previous = 'and'
                continue
            if word in UNITS | TENS | ORDINALS:
                n = (UNITS | TENS | ORDINALS)[word]
                if previous in UNITS | ORDINALS or (previous in TENS and n >= 10):
                    break
                group += n
                previous = word
                i += 1
                if word in ORDINALS:
                    break
            elif word in SCALES | SCALE_ORDINALS:
                scale = (SCALES | SCALE_ORDINALS)[word]
                if scale == 100:
                    group = max(group, 1) * scale
                else:
                    total += max(group, 1) * scale
                    group = 0
                seen_scale = True
                previous = word
                i += 1
                if word in SCALE_ORDINALS:
                    break
            else:
                break
        value = Decimal(total + group)
        if i < len(tokens) and tokens[i].group().lower() == 'point':
            j = i + 1
            digits = ''
            while j < len(tokens) and tokens[j].group().lower() in UNITS and UNITS[tokens[j].group().lower()] < 10:
                digits += str(UNITS[tokens[j].group().lower()])
                j += 1
            if digits:
                value += Decimal('0.' + digits)
                i = j
        if negative:
            value = -value
        result.append(dict(value=canonical(value), start=tokens[start].start(), end=tokens[i-1].end()))
    return result
