import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
K = '/home/xr-25b7/bulk/fig-1109/render/kf/'
T = ['1.00', '3.88', '5.28', '6.00', '7.20']
COLT = ['t = 1.0 s\n"Welcome, everyone"', 't = 3.9 s\n"...if YOU have..."', 't = 5.3 s\nuser barges in', 't = 6.0 s\n(+0.7 s)', 't = 7.2 s\n(+1.9 s)']
ROWS = [('echog_g1', 'ECHO-G\n(Unitree G1)', (170, 60, 550, 800)), ('robogesture_g1', 'RoboGesture\n(Unitree G1)', (170, 60, 550, 800)),
        ('scmg_ex002_q34', 'SCMG (ours)\n(EX002)', (150, 40, 590, 700))]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig = plt.figure(figsize=(13.5, 12.2))
gs = fig.add_gridspec(5, 6, height_ratios=[0.36, 1, 1, 1, 0.95], width_ratios=[0.55, 1, 1, 1, 1, 1], hspace=0.16, wspace=0.03)
# transcript timeline
ax = fig.add_subplot(gs[0, 1:]); ax.set_xlim(0, 7.5); ax.set_ylim(0, 1); ax.axis('off')
words = [(0.40, 1.56, 'Welcome, everyone,'), (1.56, 3.56, 'welcome to our hall.'), (3.56, 3.76, ''), (3.76, 3.84, 'YOU'), (3.84, 5.00, 'have any questions,'), (5.00, 5.28, ''), (5.28, 6.64, 'free to raise your hand.')]
for a, b, w in words:
    cut = a >= 5.28
    ax.add_patch(Rectangle((a, 0.38), b - a, 0.3, color='#d0d0d0' if cut else ('#f2b134' if w == 'YOU' else '#8fb3d9'), alpha=0.5 if cut else 0.9, lw=0))
    if w and w != 'YOU': ax.text((a + b) / 2, 0.53, w, ha='center', va='center', fontsize=9.5, color='#888888' if cut else 'black', style='italic' if cut else 'normal')
ax.plot([5.28, 5.28], [0.3, 0.95], color='#d62728', lw=2); ax.text(5.33, 0.88, 'barge-in: robot speech stops', color='#d62728', fontsize=10, va='center')
ax.add_patch(Rectangle((0.0, 0.82), 0.12, 0.13, color='#f2b134', lw=0)); ax.text(0.17, 0.885, 'explicit semantic word', fontsize=9, va='center')
ax.add_patch(Rectangle((1.75, 0.82), 0.12, 0.13, color='#d0d0d0', lw=0)); ax.text(1.92, 0.885, 'withdrawn after barge-in', fontsize=9, va='center')
for x in T: ax.plot([float(x)] * 2, [0.26, 0.38], color='#444444', lw=1.2)
ax.set_title('Same utterance, same barge-in time (robot TTS voice, Chinese; English gloss)', fontsize=12.5, pad=4)
for r, (name, lab, box) in enumerate(ROWS):
    a0 = fig.add_subplot(gs[r + 1, 0]); a0.axis('off'); a0.text(0.95, 0.5, lab, ha='right', va='center', fontsize=12, fontweight='bold' if 'ours' in lab else 'normal')
    for c, t in enumerate(T):
        a = fig.add_subplot(gs[r + 1, c + 1]); im = Image.open(f'{K}{name}_{t}.png').convert('RGB').crop(box)
        a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color('#d62728' if t == '5.28' else '#bbbbbb'); s.set_linewidth(2.2 if t == '5.28' else 0.8)
        if r == 0: a.set_title(COLT[c], fontsize=10.5, color='#d62728' if c >= 2 else 'black')
# wrist speed
W = json.load(open('/home/xr-25b7/bulk/fig-1109/wrist_speed.json'))
ax = fig.add_subplot(gs[4, 1:])
for k, lab, col in (('echog', 'ECHO-G (G1)', '#7f7f7f'), ('robogesture', 'RoboGesture (G1)', '#9467bd'), ('scmg', 'SCMG (ours, EX002)', '#1f9e89')):
    t, v = map(np.array, W[k]); n = 5 if k == 'scmg' else 3; v = np.convolve(v, np.ones(n) / n, 'same')
    ax.plot(t, v, color=col, lw=2.2 if k == 'scmg' else 1.6, label=lab)
ax.axvline(5.28, color='#d62728', lw=2); ax.axvspan(5.28, 7.5, color='#d62728', alpha=0.06)
ax.set_xlim(0, 7.5); ax.set_xlabel('time (s)'); ax.set_ylabel('max wrist speed (m/s)'); ax.legend(loc='upper left', fontsize=9.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('/home/xr-25b7/bulk/fig-1109/figure/same_utterance.png', dpi=150, bbox_inches='tight'); fig.savefig('/home/xr-25b7/bulk/fig-1109/figure/same_utterance.pdf', bbox_inches='tight')
print('ok')
