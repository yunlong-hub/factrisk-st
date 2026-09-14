"""Conservative French integer mentions for output-blind cohort selection.

Grammar covers standard metropolitan French integers below one million.
Bare un/une are excluded as article-ambiguous; malformed numeric runs fail closed.
This parser never examines a model prediction.
"""
import re

SMALL = 'zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize'.split()


def spell_under_100(n):
    if n < 17:
        return SMALL[n]
    if n < 20:
        return 'dix ' + SMALL[n-10]
    if n < 70:
        tens = {20:'vingt',30:'trente',40:'quarante',50:'cinquante',60:'soixante'}
        a, b = divmod(n, 10)
        return tens[10*a] + ((' et un' if b == 1 else ' '+SMALL[b]) if b else '')
    if n < 80:
        return 'soixante ' + ('et onze' if n == 71 else spell_under_100(n-60))
    return 'quatre vingt' + ('s' if n == 80 else ' '+spell_under_100(n-80))


def spell_under_1000(n):
    if n < 100:
        return spell_under_100(n)
    a,b=divmod(n,100)
    return (SMALL[a]+' ' if a>1 else '')+'cent'+('s' if a>1 and not b else '')+(' '+spell_under_100(b) if b else '')


def normalize(text):
    return re.sub(r'[\s\-‐‑–]+',' ',text.lower().replace('zero','zéro')).strip()


LEXICON = {normalize(spell_under_1000(n)): n for n in range(1000)}
# Number spelling reforms vary hyphenation, which normalization handles.
VOCAB = set(' '.join(LEXICON).split()) | {'mille','une','million','millions','milliard','milliards'}
TOKEN = re.compile(r'\d+|[^\W\d_]+', re.UNICODE)


def parse_integer(text):
    text=normalize(text)
    if text.isdigit(): return int(text)
    if text in LEXICON: return LEXICON[text]
    words=text.split()
    if words.count('mille') == 1:
        k=words.index('mille')
        left=' '.join(words[:k]);right=' '.join(words[k+1:])
        a=LEXICON.get(left) if left else 1
        b=LEXICON.get(right) if right else 0
        if a is not None and b is not None and 1<=a<1000:
            return a*1000+b
    return None


def french_mentions(text):
    tokens=list(TOKEN.finditer(text.lower()));out=[];i=0
    while i<len(tokens):
        word=tokens[i].group()
        if word.isdigit():
            out.append(dict(value=str(int(word)),start=tokens[i].start(),end=tokens[i].end()));i+=1;continue
        if word not in VOCAB:
            i+=1;continue
        start=i;i+=1
        while i<len(tokens) and tokens[i].group() in VOCAB and re.fullmatch(r'[\s\-‐‑–]*',text[tokens[i-1].end():tokens[i].start()]):
            # 'et' alone starts no quantity; include it only inside a valid run.
            i+=1
        phrase=text[tokens[start].start():tokens[i-1].end()]
        if normalize(phrase) in {'un','une','et'}: continue
        n=parse_integer(phrase)
        if n is None: return None
        out.append(dict(value=str(n),start=tokens[start].start(),end=tokens[i-1].end()))
    return out
