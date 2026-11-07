"""(c)1 G5: front-view stick figure of the retargeter's human side (SMPL-X joints, y-up) at 25 fps, hands coloured by the
per-frame state: grey = far, blue = near-body (wnear > 0.5), red = repair > 5 cm. Paired with the Isaac robot render (hstack).
    python regate_human_panel.py <human.npz> <out.mp4> [title]"""
import subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]     # SMPL-X body 22
z = np.load(sys.argv[1]); J = z['joints']; near = z['near']; rc = z['repair_cm']; T = min(len(J), len(near), len(rc))
J = J[:T:2]; near = near[:T:2]; rc = rc[:T:2]                                                 # 50 -> 25 fps
x, y = J[..., 0], J[..., 1]; cx = np.median(x[:, 0]); cy = np.median(y[:, 0])
W, H = 480, 720; fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', sys.argv[2]], stdin=subprocess.PIPE)
title = sys.argv[3] if len(sys.argv) > 3 else ''
for k in range(len(J)):
    ax.clear(); ax.set_xlim(cx - .6, cx + .6); ax.set_ylim(cy - .9, cy + .9); ax.set_aspect('equal'); ax.axis('off')
    for j, p in enumerate(PARENT):
        if p >= 0: ax.plot([x[k, j], x[k, p]], [y[k, j], y[k, p]], '-', color='0.3', lw=3)
    for h, w in ((0, 20), (1, 21)):
        c = 'red' if rc[k, h] > 5 else ('tab:blue' if near[k, h] else '0.5')
        ax.plot(x[k, w], y[k, w], 'o', color=c, ms=14)
    ax.text(.02, .98, f'human (front view)  t={k / 25:5.2f}s\nblue = near body, red = robot repair > 5 cm\n{title}', transform=ax.transAxes,
            va='top', fontsize=8)
    fig.canvas.draw(); ff.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
ff.stdin.close(); ff.wait(); print('PANEL_DONE', sys.argv[2], len(J))
