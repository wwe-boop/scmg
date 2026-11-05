"""EX002-specific tests of the ECHO-G streaming fine-tune (11-05).  test_stream.py (generic: zero init, degenerate stream ==
offline, no leakage, lockstep batching) is run separately with an EX002 checkpoint; this file covers what only EX002 has.
    PYTHONPATH=<dlc-package-r2>/official/src python scripts/echog_stream/test_ex002.py --meta <dev_meta with lengths.csv>
1  ex002_export.py == the frozen export_dev.py --retrained on its released smoke export (evidence/train-smoke): gen_Hinf / gt
   bit-identical, seeds 0 and 1, and the subset audit (audit_exports.py checks) passes.
2  stream_infer --offline (StreamSGDiT, zero init) of the smoke checkpoint == the released official EX002 dev predictions of the
   same checkpoint (echo_g.inference.export_predictions, batch 3 with padding: float order only), physical units.
3  prefix audio on EX002: PrefixAudio(src_fps = payload['resampling'].source_fps = 30) on the WHOLE chunk waveform reproduces the
   released condition_50fps features (official 30 FPS extraction + 30 -> 50 resampling), and a prefix [0, t) equals
   features that depend on the waveform prefix only (perturbing audio after t leaves them bit-identical).
4  training windows on real full-data-r1 clips (--profile ex002): token_times read, cut on the 16-frame 50 FPS block grid,
   window inside the model capacity, no audio at / after now + ahead, visible tokens arrived (x2) and voiced.
5  block_stats --ex002-dev grid: chunk-anchored 320 ms grid (16 frames) inside each data-dev chunk.
"""
import argparse, hashlib, json, math, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'echog_g1'))
E = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official'); SM = E / 'evidence/train-smoke'
ap = argparse.ArgumentParser()
ap.add_argument('--meta', type=Path, required=True, help='dir with the dev lengths.csv (ex002_stream.sh writes <OUT>/dev_meta)')
ap.add_argument('--data-root', type=Path, default=Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/echog-official/full-data-r1'))
ap.add_argument('--w2v', type=Path, default=E / 'models/wav2vec2-large-xlsr-53-english')
ap.add_argument('--ci', type=int, nargs='+', default=[0, 1, 2]); ap.add_argument('--device', default='cuda')
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device(a.device); RES = {}


def ok(name, cond, msg=''):
    RES[name] = bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (' | ' + msg if msg else ''), flush=True)


# ---- 1 exporter == frozen export_dev.py
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, str(HERE / 'ex002_export.py'), '--native', str(SM / 'dev-native'), '--prefix', 'x', '--output', td,
                        '--ci', *map(str, a.ci)], capture_output=True, text=True)
    same = r.returncode == 0; worst = 0.0
    for ci in a.ci:
        for mine, ref in (('x', 'smoke_retrained'), ('x_seed1', 'smoke_retrained_seed1')):
            if not same: break
            y, yr = np.load(f'{td}/{mine}/gen_Hinf_{ci}.npy'), np.load(SM / f'eval-dev/{ref}/gen_Hinf_{ci}.npy')
            g, gr = np.load(f'{td}/{mine}/gt_{ci}.npy'), np.load(SM / f'eval-dev/{ref}/gt_{ci}.npy')
            same &= y.dtype == yr.dtype and np.array_equal(y, yr) and np.array_equal(g, gr)
            if y.shape == yr.shape: worst = max(worst, float(np.abs(y - yr).max()))
ok('1 ex002_export.py == frozen export_dev.py --retrained (released smoke export, seeds 0/1)', same,
   f'clips {a.ci}, max|diff| {worst:.3g}' + ('' if r.returncode == 0 else ' | ' + r.stderr[-400:]))

# ---- 2 stream_infer --offline == official export_predictions (same checkpoint)
ic = json.loads((SM / 'dev-native/inference_complete.json').read_text())
sha = hashlib.sha256((SM / 'best.pt').read_bytes()).hexdigest()
chunks = json.loads((E / 'data-dev/MANIFEST.json').read_text()); stems = [c['stem'] for c in chunks if c['ci'] in a.ci][:4]
with tempfile.TemporaryDirectory() as td:
    sf = Path(td) / 'stems.txt'; sf.write_text('\n'.join(stems) + '\n')
    r = subprocess.run([sys.executable, str(HERE / 'stream_infer.py'), '--checkpoint', str(SM / 'best.pt'), '--cond-dir', str(E / 'data-dev/condition_50fps'),
                        '--meta', str(a.meta), '--stems', str(sf), '--out', f'{td}/o', '--seed', '0', '--offline', '--device', a.device],
                       capture_output=True, text=True)
    d2 = float('inf')
    if r.returncode == 0:
        d2 = max(float((torch.load(f'{td}/o/{s}.pt', weights_only=False)['robot_repr']
                        - torch.load(SM / f'dev-native/seed_000/{s}.pt', weights_only=True)['robot_repr']).abs().max()) for s in stems)
ok('2 stream_infer --offline == released official EX002 predictions (same checkpoint, physical rad)', sha == ic['checkpoint_sha256'] and d2 < 1e-4,
   f'checkpoint sha match {sha == ic["checkpoint_sha256"]}, {len(stems)} chunks, max|diff| {d2:.3g}' + ('' if r.returncode == 0 else ' | ' + r.stderr[-400:]))

# ---- 3 prefix audio (EX002 pipeline)
import librosa
from echog_slide import PrefixAudio, resample_source_fps
pa = PrefixAudio(a.w2v, dev); worst_abs = 0.0; worst_rel = 0.0; srcs = set(); prefix_ok = True
for s in stems[:2]:
    p = torch.load(E / f'data-dev/condition_50fps/{s}.pt', map_location='cpu', weights_only=True)
    src = resample_source_fps(p); srcs.add(src); ref = torch.as_tensor(p['audio_features']).float()
    wav = librosa.load(E / f'data-dev/audio/{s}.wav', sr=16_000, mono=True)[0]
    full = pa(wav, math.inf, 50.0, src).cpu()
    if full.shape != ref.shape: worst_abs = float('inf'); continue
    worst_abs = max(worst_abs, float((full - ref).abs().max())); worst_rel = max(worst_rel, float((full - ref).norm() / ref.norm()))
    t = 3.0; w2 = wav.copy(); w2[int(t * 16000):] = np.random.default_rng(0).normal(0, 0.1, len(w2) - int(t * 16000)).astype(w2.dtype)
    prefix_ok &= torch.equal(pa(wav, t, 50.0, src), pa(w2, t, 50.0, src)) and len(pa(wav, t, 50.0, src)) == round(t * 50)
ok('3a EX002 condition payloads record the 30 -> 50 FPS resampling', srcs == {30.0}, f'source fps {srcs}')
ok('3b prefix audio on the whole chunk == released condition_50fps (official extraction + resampling)', worst_rel < 2e-2,
   f'max|diff| {worst_abs:.3g}, max relative L2 {worst_rel:.3g} (GPU kernels / FP16 rounding of the released cache)')
ok('3c prefix features depend only on the arrived waveform (audio after t perturbed: bit-identical, round(t x 50) frames)', prefix_ok)
del pa; torch.cuda.empty_cache()

# ---- 4 training windows on real EX002 clips
from echo_g.config import ExperimentConfig
from echo_g.data import load_motion_stats
from stream_core import StreamConfig, frames_before, sample_window, visible_tokens
from train_stream import PROFILES, StreamDataset
ck = torch.load(SM / 'best.pt', map_location='cpu', weights_only=False); cfg = ExperimentConfig.from_dict(ck['experiment']['config'])
fps = float(cfg.data.fps); mean, std = load_motion_stats(a.data_root / cfg.data.stats_file, 34)
ds = StreamDataset(a.data_root, cfg.data, cfg.model, 'val', mean, std, 24)
sc = StreamConfig(); bad = []; nt = 0
for i in range(len(ds)):
    s = ds[i]; nt += len(s['token_times'])
    if s['token_times'].shape != (s['text'].shape[0], 2) or not np.allclose(s['token_times'].mean(1), s['token_centers'].numpy(), atol=1e-3): bad.append((i, 'token_times'))
    for j in range(20):
        w = sample_window(s, np.random.default_rng(1000 * i + j), sc, fps); m = w['meta']
        if m['c'] % 16 or m['c'] != 16 * m['k']: bad.append((i, 'grid'))
        if m['fe'] - m['ws'] > cfg.model.max_t: bad.append((i, 'capacity'))
        rows = torch.nonzero(w['audio'].abs().sum(1)).flatten() + m['ws']
        if len(rows) and rows.max().item() >= frames_before(m['now'] + m['ahead'], fps): bad.append((i, 'audio future'))
        vis = visible_tokens(s['token_times'], m['now'], m['ahead'], 'voiced', sc.x2_rate, sc.x2_lag)
        if w['meta']['n_visible'] != int(min(vis.sum(), cfg.model.max_text_tokens)): bad.append((i, 'tokens'))
ok('4 EX002 training windows (full-data-r1 val clips): token_times, 16-frame grid, capacity, no future audio / text',
   not bad and PROFILES['ex002'] == dict(fps=50, motion_dim=34) and cfg.model.motion_dim == 34 and fps == 50,
   f'{len(ds)} clips x 20 windows, {nt} tokens, bad {bad[:3]}')

# ---- 5 block_stats grid
import block_stats as BS
g = BS.grid(599, 0.32, 50.0)
ok('5 block grid at 50 FPS: chunk-local multiples of 16 frames', np.array_equal(g, np.arange(16, 599, 16)), f'{len(g)} boundaries in a 599-frame chunk')
print('ALL_OK' if all(RES.values()) else 'SOME_FAILED', sum(RES.values()), '/', len(RES))
