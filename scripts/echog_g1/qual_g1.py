"""11-08 qualitative G1 set (user: "需要确保实际的效果"): side-by-side official ECHO-G G1 MuJoCo renders with audio + 3x4 contact sheets.
Renderer: echo_g.visualization.export -> render (official code, g1_mocap_29dof.xml, fixed camera), panels labelled and stacked, audio muxed.
Clip selection is a fixed rule written to RULE.json BEFORE rendering (no cherry-picking):
  main : the 406-clip every-8th val_common subset, stems sorted by sha1(stem) hex, first N; panels GT / ours psync + CBF-RG / ECHO-G
         official offline / EMAGE + GMR (existing outputs only; CBF-RG = safety_run.py --rg-pairs applied to the existing psync output)
  int  : G1 dev interruptions (interrupts_dev.json) of the czm6kws silence-input dev twin, keys stem@frac sorted by sha1, first N, plus
         the K items with the largest R0causal_p per-item descriptor distance (mean_k |kin_k - median_k| / sd_k over the 5 RKD descriptors,
         human retraction windows of human_ref_v2_train_drop); panels B3c / R0causal_p / B2_p; frames before t_i = the uninterrupted
         output (executed prefix), audio silent from t_i (the model's input), red INTERRUPT marker from t_i; window [t_i - 3 s, t_i + 2.13 s]
Contact sheet: 12 frames of the stacked video (main: evenly over the clip; int: evenly over [t_i - 0.5 s, t_i + 2.0 s]).
    python scripts/echog_g1/qual_g1.py main --out O [--n 6]        python scripts/echog_g1/qual_g1.py int --out O [--n 6 --k 2]"""
import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parents[2]))
OFF = os.environ.get('ECHOG_OFFICIAL_SRC', '/home/xr-25b7/bulk/echog-official-ro/official/src')
RDEPS = os.environ.get('RENDER_DEPS', '/home/xr-25b7/bulk/pydeps/render:/home/xr-25b7/bulk/pydeps/mujoco-only')
MJCF = '/home/xr-25b7/bulk/g1-assets/unitree_g1/g1_mocap_29dof.xml'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FF = os.environ.get('RENDER_FFMPEG', '/home/xr-25b7/bulk/pydeps/render/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2')   # has drawtext
B = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'); CZ = B / 'causal-train-1105'
W, H = 480, 544
sha = lambda s: hashlib.sha1(s.encode()).hexdigest()


def render_panel(pt, tmp, key, splice=None):
    """robot_repr .pt -> official export npz -> mp4. splice=(prefix_pt, f): frames < f from prefix_pt (executed prefix)."""
    env = dict(os.environ, PYTHONPATH=f'{OFF}:{RDEPS}:' + os.environ.get('PYTHONPATH', ''), MUJOCO_GL='egl')
    src = pt
    if splice is not None:
        x = torch.load(pt, map_location='cpu', weights_only=False); u = torch.load(splice[0], map_location='cpu', weights_only=False)
        r, ru = x['robot_repr'].clone(), u['robot_repr']; f = splice[1]; r[:f] = ru[:f]
        x = dict(x, robot_repr=r, real_num_frames=int(len(r)))   # truncated continuations (f_i + 64) keep the parent's real_num_frames
        src = tmp / f'{key}_spliced.pt'; torch.save(x, src)
    npz, mp4 = tmp / f'{key}.npz', tmp / f'{key}.mp4'
    subprocess.run([sys.executable, '-m', 'echo_g.visualization.export', '--input', str(src), '--output', str(npz), '--overwrite'], env=env, check=True,
                   capture_output=True)
    subprocess.run([sys.executable, '-m', 'echo_g.visualization.render', '--npz', str(npz), '--out', str(mp4), '--mjcf', MJCF, '--width', str(W),
                    '--height', str(H), '--fixed-camera'], env=env, check=True, capture_output=True)
    return mp4


def compose(panels, wav, dst, t0=None, t1=None, t_i=None):
    n = len(panels); inputs = sum([['-i', str(m)] for _, m in panels], []) + ['-i', str(wav)]
    tr = f'trim=start={t0}:end={t1},setpts=PTS-STARTPTS,' if t0 is not None else ''
    atr = f'atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS,' if t0 is not None else ''
    rel = None if t_i is None else t_i - (t0 or 0.)
    mark = '' if rel is None else (f",drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=6:enable='gte(t,{rel:.3f})'"
                                   f",drawtext=fontfile={FONT}:text='INTERRUPT':x=(w-text_w)/2:y=h-48:fontsize=30:fontcolor=red:box=1:boxcolor=white@0.7:enable='gte(t,{rel:.3f})'")
    vf = ';'.join(f"[{i}:v]{tr}drawtext=fontfile={FONT}:text='{lab}':x=(w-text_w)/2:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6{mark}[v{i}]"
                  for i, (lab, _) in enumerate(panels))
    st = ''.join(f'[v{i}]' for i in range(n)) + f'hstack=inputs={n}[v]'
    af = f"[{n}:a]{atr}" + (f"volume=enable='gte(t,{rel:.3f})':volume=0" if rel is not None else 'anull') + '[a]'
    subprocess.run([FF, '-y', '-loglevel', 'error', *inputs, '-filter_complex', f'{vf};{st};{af}', '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-c:a', 'aac', '-shortest', str(dst)], check=True)


def contact_sheet(mp4, times, png, labels=None):
    for d in RDEPS.split(':'):
        if d not in sys.path: sys.path.append(d)
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(dir=png.parent)); ims = []
    for k, t in enumerate(times):
        f = tmp / f'{k}.png'
        subprocess.run([FF, '-y', '-loglevel', 'error', '-ss', f'{max(t, 0):.3f}', '-i', str(mp4), '-frames:v', '1', str(f)], check=True)
        if f.exists(): ims.append(Image.open(f).convert('RGB'))
    w, h = ims[0].size; sc = 0.5; w2, h2 = int(w * sc), int(h * sc)
    sheet = Image.new('RGB', (4 * w2, 3 * h2), 'white')
    from PIL import ImageDraw, ImageFont
    try: fnt = ImageFont.truetype(FONT, 22)
    except OSError: fnt = ImageFont.load_default()
    for k, im in enumerate(ims[:12]):
        x0, y0 = (k % 4) * w2, (k // 4) * h2; sheet.paste(im.resize((w2, h2)), (x0, y0))
        if labels: ImageDraw.Draw(sheet).text((x0 + 8, y0 + h2 - 30), labels[k], fill=(255, 255, 0), font=fnt, stroke_width=2, stroke_fill=(0, 0, 0))
    sheet.save(png)
    for f in tmp.iterdir(): f.unlink()
    tmp.rmdir()


def clip_metrics(pt, splice=None, rg=None, coll=None, lim=None):
    r = torch.load(pt, map_location='cpu', weights_only=False)['robot_repr'].double().numpy()
    if splice is not None:
        u = torch.load(splice[0], map_location='cpu', weights_only=False)['robot_repr'].double().numpy(); r = r.copy(); r[:splice[1]] = u[:splice[1]]
    q = r[:, 10:]; out = dict(frames=int(len(q)))
    if lim is not None: out['pos_limit_frames_pct'] = float(100 * ((q < lim.lo - 1e-6) | (q > lim.hi + 1e-6)).any(1).mean())
    if coll is not None: out['g1_contact_frames_pct'] = float(100 * np.mean([coll.collides(x) for x in q]))
    if rg is not None:
        d = np.stack([rg.distances(x) for x in q]); nb = len(rg.C.FINGER_FINGER_PAIRS) + len(rg.C.FINGER_HAND_PAIRS)
        out.update(rg_penetration_frames_pct=float(100 * (d < 0).any(1).mean()), rg_penetration_armtorso_frames_pct=float(100 * (d[:, nb:] < 0).any(1).mean()))
    a = np.diff(q, 2, axis=0) * 900.; out['mean_abs_acc_rad_s2'] = float(np.abs(a).mean())
    return out


def checkers():
    from scmg import safety_proj as S
    from scmg.safety_rg import RGCollision
    return S.Limits(), S.G1Collision(), RGCollision(S.G1_JOINTS, fast=True)


def main_set(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sub = [l.strip() for l in open(CZ / 'rgcheck/vc_every8.txt') if l.strip()]
    F = CZ / 'final3-1106-perclip/test/czmws-s0/final-test/psync/seed_000'
    E = B / 'echog-stream-1105/g1-confirm/preds/official_offline/seed_000'
    G = Path('/mnt/bulk/zehan/baselines-g1-1105/predmirror/g1_emage_seed0'); GT = B / 'baseline-repro-1104/echog/data_root/motion_39d_30fps'
    WAV = Path('/mnt/bulk/zehan/echog-dataset/ds/audio')
    pool = [s for s in sub if all((d / f'{s}.pt').exists() for d in (F, E, G, GT)) and (WAV / f'{s}.wav').exists()]
    stems = sorted(pool, key=sha)[:a.n]
    rule = dict(set='main', rule='406-clip every-8th val_common subset (rgcheck/vc_every8.txt) with all four outputs and the wav; sorted by sha1(stem) hex; first N',
                n=a.n, pool=len(pool), stems=stems, ours='czmws-s0 psync per_clip seed 0 (final3-1106-perclip) + safety_run.py --rg-pairs (exp/safe-proj)',
                echog='official_offline seed_000', gmr='EMAGE seed 0 + GMR (baselines-g1-1105 predmirror g1_emage_seed0)', note='existing test-side outputs only; no new inference')
    (out / 'RULE.json').write_text(json.dumps(rule, indent=1))
    st_txt = out / 'stems.txt'; st_txt.write_text('\n'.join(stems) + '\n')
    cbf = out / '_cbf_rg'
    if not (cbf / 'cbf_qp' / f'{stems[-1]}.pt').exists():
        subprocess.run([sys.executable, str(HERE.parent / 'safety_run.py'), '--pred-dir', str(F), '--stems', str(st_txt), '--out-root', str(cbf),
                        '--layers', 'cbf_qp', '--rg-pairs', '--workers', '1'], check=True, cwd=str(HERE.parents[2]))
    lim, coll, rg = checkers(); man = dict(rule=rule, clips={})
    for s in stems:
        panels = [('GT', GT / f'{s}.pt'), ('Ours psync + CBF-RG', cbf / 'cbf_qp' / f'{s}.pt'), ('ECHO-G official', E / f'{s}.pt'), ('EMAGE + GMR', G / f'{s}.pt')]
        tmp = Path(tempfile.mkdtemp(prefix='q_', dir=out)); vids = [(lab, render_panel(p, tmp, f'p{i}')) for i, (lab, p) in enumerate(panels)]
        dst = out / f'{s}.mp4'; compose(vids, WAV / f'{s}.wav', dst)
        dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(dst)], capture_output=True, text=True).stdout)
        ts = list(np.linspace(0.2, dur - 0.2, 12)); contact_sheet(dst, ts, out / f'{s}_sheet.png', [f't {t:.1f} s' for t in ts])
        man['clips'][s] = dict(video=dst.name, sheet=f'{s}_sheet.png', duration_s=dur,
                               panels={lab: dict(src=str(p), **clip_metrics(p, rg=rg, coll=coll, lim=lim)) for lab, p in panels})
        for f in tmp.iterdir(): f.unlink()
        tmp.rmdir(); print('MAIN', s, flush=True)
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1))


def int_set(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Wd = Path('/home/xr-25b7/bulk/interrupt-1105'); O = Wd / 'dev/psync_czm6kws_sil'
    ints = json.load(open(Wd / 'interrupts_dev.json')); ints = ints['interrupts'] if isinstance(ints, dict) else ints
    dirs = dict(B3c=O / 'B3c', R0causal_p=O / 'R0causal_p300', B2_p=O / 'B2_p300')
    key = lambda r: f"{r['stem']}@{r['frac']}"
    pool = [r for r in ints if all((d / f'{key(r)}.pt').exists() for d in dirs.values()) and (O / 'no_int' / f"{r['stem']}.pt").exists()]
    first = sorted(pool, key=lambda r: sha(key(r)))[:a.n]
    rows = {}
    for l in open(Wd / 'metrics/psync_czm6kws_sil.jsonl'):
        x = json.loads(l); rows[(x['method'], x['stem'], float(x['frac']))] = x
    sys.path.insert(0, str(Path('/home/xr-25b7/projects/gesture/scmg-wt-intcausal/scripts/interrupt')))
    import icommon as IC
    hw = np.load(Wd / 'human_ref_v2_train_drop_windows.npz')
    HK = np.stack([IC.kin_features(IC.fk(w), 30) for w in hw['rr_ctx'].astype(np.float64)])        # = eval_interrupt.py H_kin_all
    HK = HK[hw['retract'].astype(bool)]; med, sd = np.median(HK, 0), HK.std(0) + 1e-6
    score = lambda r: float(np.mean(np.abs(np.array(rows[('R0causal_p', r['stem'], float(r['frac']))]['kin']) - med) / sd))
    scored = [r for r in pool if ('R0causal_p', r['stem'], float(r['frac'])) in rows]
    worst = [r for r in sorted(scored, key=score, reverse=True) if key(r) not in {key(x) for x in first}][:a.k]
    sel = [(r, 'first-by-sha1') for r in first] + [(r, 'worst-R0causal-descriptor-distance') for r in worst]
    rule = dict(set='g1_interrupt', rule='dev interruptions with B3c / R0causal_p300 / B2_p300 / no_int present; sorted by sha1(stem@frac); first N; '
                'plus K largest R0causal_p per-item descriptor distance mean_k |kin_k - median_k| / sd_k (human retraction windows, '
                'human_ref_v2_train_drop, icommon.kin_features)', n=a.n, k=a.k, pool=len(pool), run='dev/psync_czm6kws_sil (silence input)',
                selected=[dict(key=key(r), why=w, t_i=r['t_i'], f_i=r['f_i']) for r, w in sel], human_median=med.tolist(), human_sd=sd.tolist())
    (out / 'RULE.json').write_text(json.dumps(rule, indent=1))
    WAV = Path('/home/xr-25b7/bulk/modality-ablation-1105/echog-g1-hos-assets/audio'); man = dict(rule=rule, clips={})
    for r, why in sel:
        k_ = key(r); t_i, f_i = float(r['t_i']), int(r['f_i']); t0, t1 = max(0., t_i - 3.0), t_i + 2.13
        tmp = Path(tempfile.mkdtemp(prefix='q_', dir=out)); U = O / 'no_int' / f"{r['stem']}.pt"
        vids = [(lab, render_panel(d / f'{k_}.pt', tmp, lab, splice=(U, f_i))) for lab, d in dirs.items()]
        dst = out / f'{k_}.mp4'; compose(vids, WAV / f"{r['stem']}.wav", dst, t0=t0, t1=t1, t_i=t_i)
        ts = list(np.linspace(t_i - t0 - 0.5, t_i - t0 + 2.0, 12))
        contact_sheet(dst, ts, out / f'{k_}_sheet.png', [f't-t_i {t - (t_i - t0):+.2f} s' for t in ts])
        man['clips'][k_] = dict(video=dst.name, sheet=f'{k_}_sheet.png', why=why, t_i=t_i, window=[t0, t1],
                                descriptor_distance_R0causal=score(r) if ('R0causal_p', r['stem'], float(r['frac'])) in rows else None,
                                metrics={m: {kk: rows.get((m if m != 'B3c' else 'ours_B3c', r['stem'], float(r['frac'])), {}).get(kk)
                                             for kk in ('react', 'v3_ratio', 'yield_R_sil', 'unsafe_any', 'kin')} for m in ('B3c', 'R0causal_p', 'ours_B2_p')})
        for f in tmp.iterdir(): f.unlink()
        tmp.rmdir(); print('INT', k_, why, flush=True)
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    m = sp.add_parser('main'); m.add_argument('--out', required=True); m.add_argument('--n', type=int, default=6)
    i = sp.add_parser('int'); i.add_argument('--out', required=True); i.add_argument('--n', type=int, default=6); i.add_argument('--k', type=int, default=2)
    a = ap.parse_args(); main_set(a) if a.cmd == 'main' else int_set(a)
