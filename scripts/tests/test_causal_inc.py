"""11-06 incremental causal feat89 (causal_feat.IncFeat89) == the per-tick prefix recomputation (prefix_feat89).
    SCMG_J=39 python scripts/tests/test_causal_inc.py        env DS (audio/, annotations/words/), PSTATS, HOSV, N_DEV (12), N_ZH (2)
 a every psync tick of every clip (t_k = 16 k + 8 frames, in tick order, then the clip end): IncFeat89.prefix(t_k) vs
   prefix_feat89(t_k): prosody block 80-88 bitwise; mel 0-79 within one f16 ulp (rare flips from the windowed mel, counted);
   D-spk dev clips (en) + Chinese train clips (character units)
 b causality: PCM after t replaced by noise and the words not ended by t deleted -> every IncFeat89 output at t_k <= t unchanged
 c per-tick time (ms) of both paths (CPU)"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import causal_feat as CF

E = os.environ; DS = Path(E['DS']); PST = CF.load_pstats(E['PSTATS']); HOSV = Path(E['HOSV'])
items = {it['id']: it for it in json.loads((HOSV / 'train.json').read_text())}
sp = json.loads((HOSV / 'SPLIT.json').read_text())
dev = sorted(sp['val_ids'])[:int(E.get('N_DEV', '12'))]
zh = [i for i in sorted(sp['train_ids']) if i.startswith('chinese')][:int(E.get('N_ZH', '2'))]
FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


t_inc, t_pre = [], []; flips = [0, 0]
for stem in dev + zh:
    it = items[stem]; T = int(it['frames'])
    words = CF.clip_words(DS / 'annotations/words' / f'{stem}.json', T); pcm, wave = CF.clip_audio(DS / 'audio' / f'{stem}.wav')
    inc = CF.IncFeat89(pcm, words, it['lang'], T, PST)
    ticks = sorted({min(T, 16 * k + 8) for k in range(-(-T // 16))} | {T}); bad = []
    for tk in ticks:
        t0 = time.perf_counter(); a = inc.prefix(tk); t_inc.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); b = np.asarray(CF.prefix_feat89(pcm, words, it['lang'], T, tk, PST), np.float16); t_pre.append(time.perf_counter() - t0)
        a32, b32 = a.astype(np.float32), b.astype(np.float32)
        if not np.array_equal(a32[:, 80:], b32[:, 80:]):
            d = np.abs(a32 - b32); bad.append((tk, int(np.argmax(d[:, 80:].max(0))) + 80, float(d[:, 80:].max())))
        ulp = np.spacing(np.abs(b[:, :80])).astype(np.float32)
        flips[0] += int((a32[:, :80] != b32[:, :80]).sum()); flips[1] += a32[:, :80].size
        if (np.abs(a32[:, :80] - b32[:, :80]) > ulp).any(): bad.append((tk, 'mel > 1 ulp', float(np.abs(a32[:, :80] - b32[:, :80]).max())))
    ok(f'a IncFeat89 == prefix_feat89 at every tick: {stem} ({it["lang"]}, {len(ticks)} ticks, {len(words)} words)', not bad, f'first diffs {bad[:3]}')
    # b: causality under a future perturbation
    for t in (T // 3, T // 2):
        w2 = wave.copy(); w2[t * CF.HOP:] = np.random.default_rng(t).normal(0, .1, len(w2) - t * CF.HOP).astype(np.float32)
        p2 = np.clip(np.round(w2 * 32768.), -32768, 32767).astype(np.int16)
        words2 = [w for w in words if CF.ended(w['end_time'], t / 50.)]
        i1, i2 = CF.IncFeat89(pcm, words, it['lang'], T, PST), CF.IncFeat89(p2, words2, it['lang'], T, PST)
        same = all(np.array_equal(i1.prefix(tk), i2.prefix(tk)) for tk in ticks if tk <= t)
        ok(f'b causal: {stem} noise after {t / 50:.2f}s, {len(words) - len(words2)} words deleted', same)
t_inc, t_pre = np.array(t_inc) * 1e3, np.array(t_pre) * 1e3
print(f'c per-tick ms: incremental mean {t_inc.mean():.1f} p50 {np.median(t_inc):.1f} p95 {np.percentile(t_inc, 95):.1f} | '
      f'prefix recompute mean {t_pre.mean():.1f} p50 {np.median(t_pre):.1f} p95 {np.percentile(t_pre, 95):.1f}  ({len(t_inc)} ticks)')
print(f'mel entries differing by 1 f16 ulp: {flips[0]} of {flips[1]}')
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}')
sys.exit(1 if FAIL else 0)
