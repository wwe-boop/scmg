"""Why the prefix Wav2Vec2 of streamed ECHO-G peaks at ~17 GB: peak memory / time of PrefixAudio over every block prefix
(0.16 s + k x 0.32 s) of the longest 50-set clip, under cuDNN settings: default | deterministic | cudnn disabled.
  python w2v_prefix_mem.py --variant default|deterministic|nocudnn --out DIR"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lat_echog_stream import ES, W, DR  # noqa: E402
import lat_common as LC  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
if a.variant == 'deterministic': torch.backends.cudnn.deterministic = True
if a.variant == 'nocudnn': torch.backends.cudnn.enabled = False
sys.path.insert(0, str(W / 'official/src')); sys.path.insert(0, str(ES / 'echog_g1'))
import librosa
from echog_slide import PrefixAudio
dev = torch.device('cuda'); pa = PrefixAudio(str(W / 'models/wav2vec2-large-xlsr-53-english'), dev)
base = torch.cuda.memory_allocated() / 2 ** 20
stems = LC.stems50()[LC.N_WARM:]
lens = {s: librosa.get_duration(path=str(DR / 'audio' / f'{s}.wav')) for s in stems}
stem = max(stems, key=lens.get); wav = librosa.load(DR / 'audio' / f'{stem}.wav', sr=16000, mono=True)[0]
rows = []
with torch.no_grad():
    for k in range(int(math.ceil(len(wav) / 16000 / 0.32)) + 1):
        t = k * 0.32 + 0.16
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        pa(wav, t, 30.0, None); torch.cuda.synchronize()
        rows.append(dict(t_end=min(t, len(wav) / 16000), ms=(time.perf_counter() - t0) * 1e3, peak_delta_mib=torch.cuda.max_memory_allocated() / 2 ** 20 - base))
pk = np.array([r['peak_delta_mib'] for r in rows]); ms = np.array([r['ms'] for r in rows])
res = dict(variant=a.variant, stem=stem, dur=len(wav) / 16000, blocks=len(rows), peak_delta_mib_max=float(pk.max()),
           peak_delta_mib_p50=float(np.median(pk)), n_over_2gib=int((pk > 2048).sum()), ms_p50=float(np.median(ms[1:])), ms_max=float(ms[1:].max()),
           worst=sorted(rows, key=lambda r: -r['peak_delta_mib'])[:5], max_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / f'w2v_prefix_{a.variant}.json').write_text(json.dumps(dict(summary=res, rows=rows), indent=1))
print(json.dumps(res, indent=1))
