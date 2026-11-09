"""Side-by-side G1 MuJoCo videos with the original audio (11-05).

Each panel is one robot_repr .pt (30 fps, physical) exported and rendered with the official ECHO-G visualization code
(echo_g.visualization: export.py -> render.py, MJCF assets/unitree_g1/g1_mocap_29dof.xml), then the panels are stacked
horizontally, labelled, and muxed with the clip's WAV.

    python scripts/echog_g1/render_compare.py --out <dir> --stem <stem> --wav <wav> --mjcf <xml> label=<pt> [label=<pt> ...]
Run with the echog python (has mujoco); ffmpeg on PATH.
"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path

OFF = os.environ.get('ECHOG_OFFICIAL_SRC', '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official/src')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--stem', required=True); ap.add_argument('--wav', required=True)
ap.add_argument('--mjcf', required=True); ap.add_argument('--width', type=int, default=480); ap.add_argument('--height', type=int, default=540)
ap.add_argument('panels', nargs='+')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in (OFF, os.environ.get('PYTHONPATH', '')) if x), MUJOCO_GL=os.environ.get('MUJOCO_GL', 'egl'))
FFMPEG = os.environ.get('RENDER_FFMPEG', 'ffmpeg')
tmp = Path(tempfile.mkdtemp(prefix='g1render_', dir=out))
vids = []
for spec in a.panels:
    lab, pt = spec.split('=', 1); key = lab.replace(' ', '_').replace('(', '').replace(')', '')
    npz, mp4 = tmp / f'{key}.npz', tmp / f'{key}.mp4'
    subprocess.run([sys.executable, '-m', 'echo_g.visualization.export', '--input', pt, '--output', str(npz), '--overwrite'], env=env, check=True)
    subprocess.run([sys.executable, '-m', 'echo_g.visualization.render', '--npz', str(npz), '--out', str(mp4), '--mjcf', a.mjcf,
                    '--width', str(a.width), '--height', str(a.height), '--fixed-camera'], env=env, check=True)
    vids.append((lab, mp4))
inputs = sum([['-i', str(m)] for _, m in vids], []) + ['-i', a.wav]
label = ';'.join(f"[{i}:v]drawtext=fontfile={FONT}:text='{lab}':x=(w-text_w)/2:y=12:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6[v{i}]"
                 for i, (lab, _) in enumerate(vids))
stack = ''.join(f'[v{i}]' for i in range(len(vids))) + f'hstack=inputs={len(vids)}[v]' if len(vids) > 1 else '[v0]copy[v]'
dst = out / f'{a.stem}.mp4'
subprocess.run([FFMPEG, '-y', '-loglevel', 'error', *inputs, '-filter_complex', f'{label};{stack}', '-map', '[v]', '-map', f'{len(vids)}:a',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-c:a', 'aac', '-shortest', str(dst)], check=True)
for f in tmp.iterdir(): f.unlink()
tmp.rmdir()
print('wrote', dst)
