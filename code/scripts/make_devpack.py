"""Small self-contained pack for latency tests on a host without fast CPFS access (5090 workstation, sshfs only).
    python scripts/make_devpack.py <out dir> <n clips> <ckpt> [<ckpt> ...]
Writes clips.pkl (per dev clip, eval_dots.py order: 24 kHz PCM, text, even-split anchors, T, cond dict, name, lang),
STATS.json, META.json, vae.pt, and for every checkpoint a slim copy (EMA weights + configs only) named <run>__<lane>.pt."""
import json, pickle, shutil, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scmg.data import View, WindowDataset, VIEW

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); N = int(sys.argv[2]); FS = 24000
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True, tok_cache=D + '/tok-qwen25-v36_1'); v.glitch = None
KEYS = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg')
clips = [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] // 16 >= 3][:N]
rows = []
for ci in clips:
    c = v.clips[ci]; T = c['frames'] // 16 * 16
    pcm, text, anchors = [], '', []
    for it in c['items']:
        x, sr = sf.read(it['manifest_audio']['path'], dtype='float32'); x = x if x.ndim == 1 else x.mean(1)
        n = int(round(it['frames'] / 50 * sr)); o = int(round(it['manifest_audio'].get('offset_s', 0) * sr))
        x = np.pad(x[o:o + n], (0, max(0, n - len(x[o:o + n]))))
        x = np.interp(np.arange(int(len(x) * FS / sr)) * sr / FS, np.arange(len(x)), x).astype(np.float32)
        t0 = sum(len(p) for p in pcm); s = it.get('text') or ''
        for j in range(len(s)):
            step = len(x) / len(s)
            anchors.append(dict(raw_codepoint_start=len(text) + j, raw_codepoint_end=len(text) + j + 1,
                                output_sample_start=int(t0 + j * step), output_sample_end=int(t0 + (j + 1) * step)))
        text += s; pcm.append(x)
    smp = WindowDataset(v, T, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    rows.append(dict(ci=ci, clip=c['clip'], lang=c['items'][0].get('lang'), T=T, pcm=np.concatenate(pcm), text=text,
                     anchors=anchors, cond={k: smp[k] for k in KEYS if k in smp}))
pickle.dump(rows, open(out / 'clips.pkl', 'wb'))
(out / 'STATS.json').write_text(json.dumps(st)); shutil.copy(f'{VIEW}/META.json', out / 'META.json')
shutil.copy('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt', out / 'vae.pt')
for ck in sys.argv[3:]:
    d = torch.load(ck, map_location='cpu', weights_only=False)
    p = Path(ck); name = f'{p.parent.parent.name}__{p.parent.name}.pt'
    torch.save({k: d[k] for k in ('base_cfg', 'extra_cfg', 'ema', 'sc', 'stage') if k in d}, out / name)
print('packed', len(rows), 'clips ->', out, sorted(x.name for x in out.iterdir()))
