"""Deterministic representation tests; not an estimate of human label accuracy."""
from factrisk.core.numbers import number_mentions
from factrisk.core.labels_v2 import label_fact
from factrisk.core.io import write_json
from factrisk.pipeline.workflow import OUT


def spelled(n):
    units='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    tens=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
    if n<20:return units[n]
    if n<100:return tens[n//10]+('-'+units[n%10] if n%10 else '')
    if n<1000:return units[n//100]+' hundred'+(' and '+spelled(n%100) if n%100 else '')
    return spelled(n//1000)+' thousand'+(' '+spelled(n%1000) if n%1000 else '')


def run():
    tests=0
    for n in range(10000):
        word=spelled(n)
        for representation in (word,word.replace('-',' '),str(n),f'{n:,}'):
            actual=number_mentions('There are '+representation+' participants.')
            if [m['value'] for m in actual]!=[str(n)]:raise AssertionError((n,representation,actual))
            tests+=1
        reference=f'There are {n} participants.'
        ok=label_fact('There are '+word+' participants.',str(n),'','number',reference)
        bad=label_fact(f'There are {n+1} participants.',str(n),'','number',reference)
        if ok['severe_fact_error']!=0 or bad['severe_fact_error']!=1:raise AssertionError(n)
        tests+=2
    write_json(OUT/'audit/representation_tests.json',dict(status='passed',checks=tests,integer_range=[0,9999],
        measures='lexical equivalence and controlled replacement behavior',
        human_validation=False,semantic_accuracy_estimate=None))
    print(f'{tests} deterministic representation checks passed',flush=True)


if __name__=='__main__':run()
