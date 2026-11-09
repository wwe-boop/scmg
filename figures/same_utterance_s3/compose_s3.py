import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
K = '/home/xr-25b7/bulk/fig-1109/s3/kf/'
T = ['1.00', '2.60', '4.32', '5.00', '6.20']
COLT = ['t = 1.0 s\n"In this hall"', 't = 2.6 s\n"...THREE zones"', 't = 4.3 s\nuser barges in', 't = 5.0 s\n(+0.7 s)', 't = 6.2 s\n(+1.9 s)']
ROWS = [('echog_g1', 'ECHO-G\n(Unitree G1)', (145, 110, 595, 560)), ('robogesture_g1', 'RoboGesture\n(Unitree G1)', (145, 110, 595, 560)),
        ('scmg_ex002_front', 'SCMG (ours)\n(EX002)', (165, 170, 615, 620))]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig = plt.figure(figsize=(13.5, 11.2))
gs = fig.add_gridspec(5, 6, height_ratios=[0.36, 1, 1, 1, 0.9], width_ratios=[0.55, 1, 1, 1, 1, 1], hspace=0.16, wspace=0.03)
# transcript timeline
ax = fig.add_subplot(gs[0, 1:]); ax.set_xlim(0, 6.5); ax.set_ylim(0, 1); ax.axis('off')
words = [(0.24, 1.48, 'In this hall,'), (1.48, 2.32, 'we prepared'), (2.32, 2.56, 'THREE'), (2.56, 4.04, 'interactive zones.'), (4.04, 4.32, ''), (4.32, 6.30, 'everyone, try them one by one.')]
for a, b, w in words:
    cut = a >= 4.32
    ax.add_patch(Rectangle((a, 0.38), b - a, 0.3, color='#d0d0d0' if cut else ('#f2b134' if w == 'THREE' else '#8fb3d9'), alpha=0.5 if cut else 0.9, lw=0))
    if w and w != 'THREE': ax.text(a + 0.06 if cut else (a + b) / 2, 0.53, w, ha='left' if cut else 'center', va='center', fontsize=9 if cut else 9.5, color='#888888' if cut else 'black', style='italic' if cut else 'normal')
ax.plot([4.32, 4.32], [0.3, 0.95], color='#d62728', lw=2); ax.text(4.37, 0.88, 'barge-in: robot speech stops', color='#d62728', fontsize=10, va='center')
ax.add_patch(Rectangle((0.0, 0.82), 0.12, 0.13, color='#f2b134', lw=0)); ax.text(0.17, 0.885, 'explicit semantic word', fontsize=9, va='center')
ax.add_patch(Rectangle((1.75, 0.82), 0.12, 0.13, color='#d0d0d0', lw=0)); ax.text(1.92, 0.885, 'withdrawn after barge-in', fontsize=9, va='center')
for x in T: ax.plot([float(x)] * 2, [0.26, 0.38], color='#444444', lw=1.2)
ax.set_title('Same utterance, same barge-in time (robot TTS voice, Chinese; English gloss)', fontsize=12.5, pad=4)
for r, (name, lab, box) in enumerate(ROWS):
    a0 = fig.add_subplot(gs[r + 1, 0]); a0.axis('off'); a0.text(0.95, 0.5, lab, ha='right', va='center', fontsize=12, fontweight='bold' if 'ours' in lab else 'normal')
    for c, t in enumerate(T):
        a = fig.add_subplot(gs[r + 1, c + 1]); im = Image.open(f'{K}{name}_{t}.png').convert('RGB').crop(box)
        a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color('#d62728' if t == '4.32' else '#bbbbbb'); s.set_linewidth(2.2 if t == '4.32' else 0.8)
        if name.startswith('scmg') and t == '2.60':
            ia = a.inset_axes([0.02, 0.50, 0.44, 0.44]); ia.imshow(Image.open(f'{K}{name}_{t}.png').convert('RGB').crop((455, 185, 615, 345))); ia.set_xticks([]); ia.set_yticks([])
            for sp in ia.spines.values(): sp.set_color('#f2b134'); sp.set_linewidth(2)
            ia.text(0.5, -0.02, 'three fingers', transform=ia.transAxes, ha='center', va='top', fontsize=8.5, color='#b07800')
        if r == 0: a.set_title(COLT[c], fontsize=10.5, color='#d62728' if c >= 2 else 'black')
# wrist speed
W = json.load(open('/home/xr-25b7/bulk/fig-1109/s3/wrist_speed.json'))
ax = fig.add_subplot(gs[4, 1:])
for k, lab, col in (('echog', 'ECHO-G (G1)', '#7f7f7f'), ('robogesture', 'RoboGesture (G1)', '#9467bd'), ('scmg', 'SCMG (ours, EX002)', '#1f9e89')):
    t, v = map(np.array, W[k]); n = 5 if k == 'scmg' else 3; v = np.convolve(v, np.ones(n) / n, 'same')
    ax.plot(t, v, color=col, lw=2.2 if k == 'scmg' else 1.6, label=lab)
ax.axvline(4.32, color='#d62728', lw=2); ax.axvspan(4.32, 6.5, color='#d62728', alpha=0.06)
ax.set_xlim(0, 6.5); ax.set_xlabel('time (s)'); ax.set_ylabel('max wrist speed (m/s)'); ax.legend(loc='upper left', fontsize=9.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('/home/xr-25b7/bulk/fig-1109/figure/same_utterance_s3.png', dpi=150, bbox_inches='tight'); fig.savefig('/home/xr-25b7/bulk/fig-1109/figure/same_utterance_s3.pdf', bbox_inches='tight')
print('ok')
