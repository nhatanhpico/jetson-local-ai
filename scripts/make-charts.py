#!/usr/bin/env python3
"""Draw docs/images/chart-*.png from docs/tegrastats-trace.csv and docs/benchmark-results.json (needs matplotlib)."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(ROOT / 'docs/tegrastats-trace.csv')))
res = json.load(open(ROOT / 'docs/benchmark-results.json'))
t = [float(r['seconds']) for r in rows]
t0 = t[0]
t = [x - t0 for x in t]
teal, orange, slate = '#0f766e', '#c2410c', '#475569'
phase_color = {'chat': '#ccfbf1', 'vision': '#e0e7ff', 'image-10': '#ffedd5', 'image-20': '#fed7aa', 'image-30': '#fdba74'}

fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
series = [('GPU load (%)', [float(r['gpu_pct']) for r in rows], teal),
          ('Used RAM (GB, shared CPU+GPU)', [int(r['ram_mb']) / 1024 for r in rows], slate),
          ('Board power (W)', [float(r['power_w']) for r in rows], orange)]
for a, (label, y, c) in zip(ax, series):
    a.plot(t, y, color=c, lw=1.6)
    a.set_ylabel(label, fontsize=9)
    a.grid(alpha=.25)
    a.spines[['top', 'right']].set_visible(False)
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]['phase'] != rows[start]['phase']:
            ph = rows[start]['phase']
            if ph in phase_color:
                a.axvspan(t[start], t[i - 1], color=phase_color[ph], alpha=.55, lw=0)
            start = i
ymax = ax[0].get_ylim()[1]
start = 0
for i in range(1, len(rows) + 1):
    if i == len(rows) or rows[i]['phase'] != rows[start]['phase']:
        ph = rows[start]['phase']
        if ph in phase_color:
            ax[0].text((t[start] + t[i - 1]) / 2, 108, ph.replace('-', ' ') + (' steps' if ph.startswith('image') else ''),
                       ha='center', fontsize=8, color=slate)
        start = i
ax[0].set_ylim(0, 118)
ax[2].set_xlabel('Seconds since benchmark start')
fig.suptitle('Jetson Orin NX 16GB: hardware load during chat, vision and image generation', fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / 'docs/images/chart-hardware-trace.png', dpi=120)

fig, ax = plt.subplots(figsize=(6.5, 4))
steps = [i['steps'] for i in res['image']]
secs = [i['wall_s'] for i in res['image']]
bars = ax.bar([str(s) for s in steps], secs, color=teal, width=.55)
ax.bar_label(bars, fmt='%.0f s', fontsize=10)
ax.set_xlabel('Sampling steps')
ax.set_ylabel('Seconds per 512×512 image (incl. model load)')
ax.set_title('Stable Diffusion 1.5 on Orin NX (CUDA, f16)')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, max(secs) * 1.15)
fig.tight_layout()
fig.savefig(ROOT / 'docs/images/chart-image-time.png', dpi=120)
print('charts written')
