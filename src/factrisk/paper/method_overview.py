"""Render the FactRisk-ST method overview as editable SVG, PDF and PNG.

Run: PYTHONPATH=src python -m factrisk.paper.method_overview
The waveform and token alignment are schematic, not experimental measurements.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from factrisk.pipeline.workflow import ROOT

FIGURES = ROOT / 'papers/factriskst/figures'
INK, GRAY, LIGHT, ACCENT = '#252525', '#666666', '#D9D9D9', '#38677B'


def text(ax, x, y, value, size=8, color=INK, bold=False, align='left'):
    ax.text(x, y, value, fontsize=size, color=color,
            weight='bold' if bold else 'normal', ha=align, va='center')


def line(ax, points, color=GRAY, width=.7, dashed=False):
    x, y = zip(*points)
    ax.plot(x, y, color=color, lw=width,
            linestyle=(0, (3, 2)) if dashed else '-', zorder=1)


def arrow(ax, p, q, color=GRAY):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7,
                               lw=.75, color=color, shrinkA=0, shrinkB=0))


def model(ax, x, y, w, title, subtitle):
    ax.add_patch(Rectangle((x, y-13), w, 26, facecolor='white',
                           edgecolor=GRAY, lw=.75, linestyle=(0, (4, 2)), zorder=2))
    text(ax, x+w/2, y+5.5, title, 8, bold=True, align='center')
    text(ax, x+w/2, y-5.5, subtitle, 6.8, GRAY, align='center')


def tokens(ax, x, y, mismatch=False):
    for i, width in enumerate((11, 9, 14, 10)):
        ax.add_patch(Rectangle((x, y-2.5), width, 5,
            facecolor=ACCENT if i == 2 and not mismatch else LIGHT,
            edgecolor=ACCENT if i == 2 else 'none', lw=.7))
        x += width+3


def build():
    fig = plt.figure(figsize=(7.5, 190/72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 540), ylim=(0, 190))
    ax.axis('off')
    blue, rust, teal, ochre = '#376889', '#A65E49', '#397D73', '#997527'
    text(ax, 71.5, 181, 'Frozen translation paths', 8.5, bold=True, align='center')
    text(ax, 270.5, 181, 'Complementary risk signals', 8.5, bold=True, align='center')
    text(ax, 476, 181, 'Risk & selection', 8.5, bold=True, align='center')
    for a, b in ((13,130), (154,387), (422,530)):
        line(ax, [(a,173),(b,173)], LIGHT, .7)

    # One speech input supplies the direct and auxiliary frozen paths.
    text(ax, 68, 164, 'Speech $x$', 8.5, align='center')
    t = np.linspace(0, 1, 240)
    env = np.exp(-((t-.25)/.15)**2) + .7*np.exp(-((t-.7)/.17)**2)
    ax.plot(28+82*t, 150+5.4*env*np.sin(100*t), color=blue, lw=.7)
    arrow(ax, (69,142), (69,134))
    model(ax, 13, 121, 116, 'Speech translation', 'Qwen2-Audio / SeamlessM4T-v2')
    # The speech bus bypasses ST; the cascade does not consume translation y.
    line(ax, [(28,150),(5,150),(5,73)], GRAY)
    arrow(ax, (5,73), (13,73))
    model(ax, 13, 73, 116, 'Whisper-large-v3', 'ASR uncertainty + transcript')
    arrow(ax, (71,60), (71,51), ochre)
    model(ax, 13, 38, 116, 'NLLB-200', 'distilled-600M')
    text(ax, 71, 98, 'Direct translation $y$', 7.2, blue, align='center')

    # Flat, quiet colour fields organise the four routes without card borders.
    rows = [
        (152, blue, '#F0F5F8', 'Generation',
         'Decoder confidence for translation $y$', 'NLL, entropy, margin, output length'),
        (115, rust, '#FAF3EF', 'Perturbation stability',
         '2 audio probes + 3 sampled decodes', 'Edit distance; number / negation disagreement'),
        (78, teal, '#F0F6F3', 'Acoustics',
         'Whisper uncertainty + signal-level statistics', 'Duration, RMS, spectral entropy, silence, clipping'),
        (41, ochre, '#F8F5EC', 'Cross-path evidence',
         'Compare $y$ with the Whisper--NLLB translation', 'NLL, text / signature mismatch, coverage gap')]
    for y, color, fill, title, desc, detail in rows:
        ax.add_patch(Rectangle((154,y-17), 233, 34, facecolor=fill,
                               edgecolor='none', zorder=0))
        line(ax, [(154,y-17),(154,y+17)], color, 1.7)
        text(ax, 163, y+11, title, 8.5, color, True)
        text(ax, 163, y, desc, 7.4)
        text(ax, 163, y-11, detail, 6.9, GRAY)
        line(ax, [(387,y),(399,y)], color, .85)
    # Explicit branching from the frozen ST decoder to its two signal routes.
    line(ax, [(129,125),(140,125),(140,152)], blue)
    arrow(ax, (140,152), (152,152), blue)
    # These three sources already sit at their route's height, so they connect
    # with a straight horizontal arrow instead of a dogleg.
    arrow(ax, (129,115), (152,115), rust)
    arrow(ax, (129,78), (152,78), teal)
    arrow(ax, (129,41), (152,41), ochre)

    # A single concatenation spine, followed by the fitted risk components.
    line(ax, [(399,41),(399,152)], GRAY, .8)
    arrow(ax, (399,130.5), (421,130.5), blue)
    text(ax, 476, 162, '21 base features', 8, bold=True, align='center')
    text(ax, 476, 152, '+ missingness indicators', 6.8, GRAY, align='center')
    ax.add_patch(Rectangle((423,117), 106, 27, facecolor='#EAF0F4',
                           edgecolor=blue, lw=.8))
    text(ax, 476, 135, 'Extra-Trees', 9, blue, True, 'center')
    text(ax, 476, 124, '600 trees', 7.2, GRAY, align='center')
    arrow(ax, (476,117), (476,103), blue)
    text(ax, 484, 111, '$p$', 8, blue)
    text(ax, 476, 95, 'Held-out calibration', 8, bold=True, align='center')
    text(ax, 476, 83, r'$r=\sigma(a\,\mathrm{logit}(p)+b)$', 8, align='center')
    arrow(ax, (476,75), (476,64), blue)
    text(ax, 476, 57, r'$r<\tau$', 10, blue, align='center')
    line(ax, [(462,54),(438,41)], teal, .85)
    line(ax, [(490,54),(516,41)], GRAY, .85)
    text(ax, 438, 33, 'Accept $y$', 8, teal, True, 'center')
    text(ax, 514, 33, 'Abstain', 8, GRAY, align='center')

    line(ax, [(13,17),(530,17)], LIGHT, .6)
    text(ax, 13, 8, 'Dashed: frozen models. References supply offline labels only.', 6.5, GRAY)
    text(ax, 530, 8, r'$\tau$: 90th percentile of calibration risks.', 6.5, GRAY, align='right')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default=str(FIGURES))
    out = Path(parser.parse_args().outdir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'],
                         'mathtext.fontset':'dejavuserif', 'svg.fonttype':'none',
                         'pdf.fonttype':42, 'ps.fonttype':42})
    fig = build()
    for suffix in ('svg', 'pdf', 'png'):
        target = out / f'method_overview.{suffix}'
        fig.savefig(target, format=suffix, dpi=300, facecolor='white')
        print(f'wrote {target}')
    plt.close(fig)


if __name__ == '__main__':
    main()
