"""End-to-end streaming latency per 320 ms chunk: raw 24 kHz PCM arriving in real time -> online features
(edge-live PrefixFeatures: log-mel + prosody + speech flag, the deployed extractor) -> Streamer.tick (planner + local
flow + VAE decode, CFG as configured) -> CBF-QP projection of the emitted frames.
    python scripts/e2e_latency.py <ckpt> <out.json> <H latents (0 = strictly causal, 4 = 320 ms look-ahead)> [cfg_w] [cfg_text 0/1] [n_clips]
Timeline (virtual clock driven by MEASURED compute): the audio needed by tick k (chunk k plus H latents of look-ahead)
has fully arrived at a_k; ready_k = max(a_k, ready_{k-1}) + compute_k. The frames tick k emits start at time e_k on
the motion timeline. Required sound delay D_k = ready_k - e_k: if the robot plays the speech D later than it is
produced, every gesture frame is ready when its sound is heard iff D >= max_k D_k.
Text anchors (TTS timestamps in deployment) are replaced by an even split of the item text over its duration -- the
feature COST is the same; the feature VALUES are not used for any quality claim here."""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch, soundfile as sf

sys.path.insert(0, '/home/xr-25b7/projects/gesture/edge-live/robot_client/edge_live_client')
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/edge-bj50')
from bj50 import prosody as P                                      # noqa: E402
from bj50.fe_numpy import PrefixFeatures                           # noqa: E402
from scmg.common import CHUNK_FRAMES                               # noqa: E402
from scmg.data import View, WindowDataset, VIEW                    # noqa: E402
from scmg.evaluate import load                                     # noqa: E402
from scmg.infer import Streamer                                 # noqa: E402
from scmg.model import INF                                     # noqa: E402
from scmg.common import U80                                     # noqa: E402
from scmg.safety import limits34, CBFFilter                        # noqa: E402

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
CK, OUT, HA = sys.argv[1], Path(sys.argv[2]), sys.argv[3]          # H: int latents | 'inf' offline | 'eN' elastic up to N
ELASTIC = int(HA[1:]) if HA.startswith('e') else None
HL = -1 if HA == 'inf' else (0 if ELASTIC is not None else int(HA))
RTF = float(os.environ.get('SCMG_TTS_RTF', '1.0'))      # TTS audio production: x s of speech exist at ttft + x * RTF
TTFT = float(os.environ.get('SCMG_TTS_TTFT', '0.0'))
CFGW = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0; CFGT = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
NC = int(sys.argv[6]) if len(sys.argv) > 6 else 12; STEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 10
dev = 'cuda'; FS = 24000; CH_S = CHUNK_FRAMES / 50
PACK = os.environ.get('SCMG_DEVPACK')                  # local pack from make_devpack.py (no CPFS reads at all)
if PACK:
    import pickle
    st = json.load(open(f'{PACK}/STATS.json')); meta = json.load(open(f'{PACK}/META.json'))
    m, vae = load(CK, f'{PACK}/vae.pt', dev); PK = pickle.load(open(f'{PACK}/clips.pkl', 'rb'))[:NC]
else:
    st = json.load(open(D + '/STATS_v36_1.json')); meta = json.load(open(f'{VIEW}/META.json'))
    m, vae = load(CK, '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt', dev)
    v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True, tok_cache=D + '/tok-qwen25-v36_1'); v.glitch = None
fm, fs_ = np.asarray(st['f_mean'], np.float32), np.asarray(st['f_std'], np.float32)
lim = limits34(meta['lower'], meta['upper'])
H = INF if HL < 0 else HL * U80
taps = np.load('/home/xr-25b7/projects/gesture/edge-live/robot_client/edge_live_client/bj50/rp_taps.npy') if Path(
    '/home/xr-25b7/projects/gesture/edge-live/robot_client/edge_live_client/bj50/rp_taps.npy').exists() else np.load(
    '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/edge-bj50/bj50/rp_taps.npy')


def clip_audio(c):
    """24 kHz PCM of the whole clip + even-split char anchors."""
    pcm, text, anchors = [], '', []
    for it in c['items']:
        x, sr = sf.read(it['manifest_audio']['path'], dtype='float32')
        x = x if x.ndim == 1 else x.mean(1)
        n = int(round(it['frames'] / 50 * sr)); o = int(round(it['manifest_audio'].get('offset_s', 0) * sr))
        x = np.pad(x[o:o + n], (0, max(0, n - len(x[o:o + n]))))
        x = np.interp(np.arange(int(len(x) * FS / sr)) * sr / FS, np.arange(len(x)), x).astype(np.float32)
        t0 = sum(len(p) for p in pcm); s = it.get('text') or ''
        if s:
            step = len(x) / len(s)
            for j in range(len(s)):
                anchors.append(dict(raw_codepoint_start=len(text) + j, raw_codepoint_end=len(text) + j + 1,
                                    output_sample_start=int(t0 + j * step), output_sample_end=int(t0 + (j + 1) * step)))
        text += s; pcm.append(x)
    return np.concatenate(pcm), text, anchors


torch.cuda.synchronize(); rows = []
def clips_iter():
    if PACK:
        for r in PK: yield r['clip'], r['lang'], r['T'], r['pcm'], r['text'], r['anchors'], r['cond']
        return
    for ci in [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] // 16 >= 3][:NC]:
        c = v.clips[ci]; T = c['frames'] // CHUNK_FRAMES * CHUNK_FRAMES
        pcm, text, anchors = clip_audio(c)
        s = WindowDataset(v, T, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        yield c['clip'], c['items'][0].get('lang'), T, pcm, text, anchors, {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg') if k in s}


for cname, clang, T, pcm, text, anchors, cond in clips_iter():
    nch = T // CHUNK_FRAMES
    S = Streamer(m, vae, np.zeros((T, fm.shape[0]), np.float32), cond, st, H=H, steps=STEPS, standby=st['q_mean'],
                 ctx_t=getattr(m, 'ctx_t', 1.0), cfg_w=CFGW, cfg_text=CFGT, elastic=ELASTIC)
    pf = PrefixFeatures(P, taps); cbf = CBFFilter(lim, np.asarray(st['q_mean'], float))
    ready = 0.; recs = []; dur = len(pcm) / FS
    arrived = lambda t: float(np.clip((t - TTFT) / RTF, 0., dur))         # seconds of speech received at wall time t
    at = lambda x: TTFT + min(x, dur) * RTF                               # wall time when speech second x has arrived
    for k in range(nch):
        t_start = max(ready, at(k * CH_S))                                  # earliest: the chunk start itself has arrived
        if H >= INF: t_start = max(t_start, at(dur))
        elif ELASTIC is None: t_start = max(t_start, at(k * CH_S + HL * 0.08))
        a_s = arrived(t_start)
        t0 = time.perf_counter()
        need_s = dur if H >= INF else a_s
        n = max(int(need_s * FS), 960)
        feat, _, _ = pf.at(text, pcm[:n], [a for a in anchors if a['output_sample_start'] < n], n >= len(pcm), len(pcm))
        nf = min(len(feat), T); S.feat[:nf] = torch.as_tensor((feat[:nf] - fm) / fs_, device=dev)
        t1 = time.perf_counter()
        fr = S.tick(arrived_s=a_s); torch.cuda.synchronize()
        t2 = time.perf_counter()
        fr = fr.cpu().numpy() if torch.is_tensor(fr) else np.asarray([f.cpu().numpy() for f in fr]) if len(fr) else np.zeros((0, 34))
        for f in fr: cbf.step(f.astype(float))
        t3 = time.perf_counter()
        comp = t3 - t0; ready = t_start + comp
        e = (S.n_emitted - len(fr)) / 50                                   # timeline start of the emitted frames
        recs.append(dict(k=k, start=t_start, arrived=a_s, H_used=S.H if S.H < INF else -1, feat_ms=1e3 * (t1 - t0),
                         model_ms=1e3 * (t2 - t1), cbf_ms=1e3 * (t3 - t2), total_ms=1e3 * comp, ready=ready, emit_start=e,
                         nframes=len(fr), D=ready - TTFT - e if len(fr) else None))
    rows.append(dict(clip=cname, lang=clang, seconds=T / 50, ticks=recs))
    Dk = [r['D'] for r in recs if r['D'] is not None]
    print(cname, f"ticks {len(recs)} total p50 {np.median([r['total_ms'] for r in recs]):.0f} ms  D max {max(Dk):.3f} s", flush=True)
allr = [r for x in rows for r in x['ticks'][1:]]                         # tick 0 includes CUDA warm-up effects per clip
q = lambda key, p: float(np.percentile([r[key] for r in allr], p))
Dall = [r['D'] for x in rows for r in x['ticks'] if r['D'] is not None]
summ = dict(ckpt=CK, H_arg=HA, tts_rtf=RTF, tts_ttft=TTFT, H_used_mean=float(np.mean([r['H_used'] for r in allr])), steps=STEPS, bf16=os.environ.get('SCMG_BF16') == '1', H_latents=HL, cfg_w=CFGW, cfg_text=CFGT, clips=len(rows), ticks=len(allr),
            feat_ms=[q('feat_ms', 50), q('feat_ms', 95)], model_ms=[q('model_ms', 50), q('model_ms', 95)],
            cbf_ms=[q('cbf_ms', 50), q('cbf_ms', 95)], total_ms=[q('total_ms', 50), q('total_ms', 95), q('total_ms', 99)],
            realtime_frac=float(np.mean([r['total_ms'] < 320 for r in allr])),
            sound_delay_s=dict(p50=float(np.percentile(Dall, 50)), p95=float(np.percentile(Dall, 95)), max=float(np.max(Dall))),
            first_motion_s=float(np.median([x['ticks'][0]['ready'] - TTFT for x in rows])),     # after the first audio
            first_frames_s=float(np.median([next(r['ready'] for r in x['ticks'] if r['nframes']) - TTFT for x in rows])),
            gpu=torch.cuda.get_device_name(0))
OUT.write_text(json.dumps(dict(summary=summ, rows=rows), ensure_ascii=False, indent=1)); print(json.dumps(summ))
