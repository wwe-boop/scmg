"""Can the 89-d audio features (feat89, the pinned training extractor) be computed ONLINE from the audio that has
arrived?  (11-04, deploy question; analysis only, nothing here changes training or inference defaults.)
    feat_causality.py <out dir> [ckpt] [vae]        (speech_server: .venv-asr-hf, CPFS native, GPU1 only)

Dev clips (val split of data-v36_1, BEAT2, forced-alignment char times, 16 kHz audio of the view).  For every
engine-protocol tick k (H = 2 latents = 160 ms look-ahead) the arrived prefix is t_k = 16k + 8 frames; the online
extractor sees pcm[:t_k] and the units (zh char / en word) whose OFFSET is <= t_k (a unit is only measurable once it is
complete).  It is compared with the whole-utterance extraction on the overlapping frames [0, t_k).
Variants (one function, monkey-patched prosody hooks; the mel path is never touched):
  v0  feat89 as trained-with (clip scope here): prominence = z-score of energy / log f0 / duration over ALL syllables
      of the utterance; beats = pick_beats (+-2 syllable local max, 0.45 s merge, global top-k rate cap) + assign_sides
  v1  v0 but prominence z-scored with FIXED per-speaker statistics (other train clips of the same speaker)
  v2  v1 + committed beats: a syllable is a beat candidate only once its 2 following syllables are complete (or the
      audio is final), the last beat is withheld while a later candidate within 0.45 s could still replace it, and
      no global rate cap
Model level: the candidate (engine口径: x2 text 15 tok/s lag .15 causal e5, deploy timing, H=2, trim_start,
cfg_null_fix, CFG 2.5, 10 steps) is run with the whole-utterance features and with the per-tick prefix features
(tick k reads exactly prefix_k), same seed; joint differences in rad.  Noise floor = seed 0 vs seed 1.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025')
import feat89 as F                                                    # noqa: E402

V = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
PARTS = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/parts')
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
PICK = [int(x) for x in os.environ.get('FC_CLIPS', '0,3,4,8,20').split(',')]   # val clip indices (View order)
NAMES = ['onset', 'prominence', 'energy', 'logf0', 'voiced', 'beat', 'beat_left', 'beat_right', 'speech_flag']
ORIG_FEATURES, ORIG_PICK = F.features, F.pick_beats
H_FR = 8                                                              # H = 2 latents x 4 frames


def clips_val():
    items = json.loads((V / 'train.json').read_text()); sp = json.loads((V / 'SPLIT.json').read_text())
    out = {}
    for split in ('val', 'train'):
        keep = set(sp['val_ids'] if split == 'val' else sp['train_ids'])
        cl = {}
        for it in items:
            if it['id'] in keep: cl.setdefault(it['clip'], []).append(it)
        out[split] = []
        for c, xs in cl.items(): xs.sort(key=lambda x: int(x['offset'])); out[split].append((c, xs))
    return out


def chars_of(xs, start):                                              # = scripts/stress_feats.py
    plain, chars = '', []
    for x in xs:
        al = json.load(open(PARTS / x['part'].split('/')[-1] / x['dir'] / 'alignment.json')) if x.get('dir') else []
        t0 = (int(x['offset']) - start) / 50.
        sep = ' ' if (x['lang'] == 'en' and plain) else ''
        base = len(plain) + len(sep); plain += sep + x['text']
        for a in al:
            chars.append(dict(index=base + int(a['index']), char=a['char'], onset_s=t0 + a['onset_s'], offset_s=t0 + a['offset_s']))
    return plain, chars


def unit_end(chars, plain, lang):
    """offset of the unit each char belongs to (zh: itself; en: its whitespace word, as Recording.from_clip groups)."""
    cs = sorted(chars, key=lambda c: int(c['index'])); end = [0.] * len(cs)
    if lang != 'en':
        return cs, [float(c['offset_s']) for c in cs]
    groups, cur = [], []
    for c in cs:
        i = int(c['index'])
        if cur and plain[int(cur[-1]['index']) + 1:i].strip() == '' and i == int(cur[-1]['index']) + 1: cur.append(c)
        else:
            if cur: groups.append(cur)
            cur = [c]
    if cur: groups.append(cur)
    end = []
    for g in groups: end += [float(g[-1]['offset_s'])] * len(g)
    return cs, end


# ---------------- prosody hooks ----------------
def fixed_features(st):
    mE, sE, mF, sF, mD, sD = st

    def features(syllables, pcm, fs=24000, cache=None):
        pcm = np.asarray(pcm, np.float32); rows = [(s,) + F_measure(s, pcm, fs) for s in syllables]
        if not rows: return []
        E = np.array([r[1] for r in rows]); Fh = np.array([r[2] for r in rows]); Dd = np.array([r[0].offset_s - r[0].onset_s for r in rows])
        zf = np.where(Fh > 0, (np.log(np.maximum(Fh, 1.)) - mF) / (sF + 1e-6), 0.)
        score = (E - mE) / (sE + 1e-6) + .7 * zf + .4 * (Dd - mD) / (sD + 1e-6)
        return [F.Syllable(s.index, s.char, s.onset_s, s.offset_s, float(e), float(f), float(p)) for (s, e, f), p in zip(rows, score)]
    return features


def F_measure(s, pcm, fs):
    import gesturelab.prosody as GP
    return GP.measure(s, pcm, fs)


def committed_beats(final, threshold=.6, min_gap_s=.45):
    def pick(syllables, **kw):
        n = len(syllables); ncand = n if final else max(0, n - 2)
        beats = []
        for i in range(ncand):
            s = syllables[i]; nb = syllables[max(0, i - 2):i] + syllables[i + 1:i + 3]
            if s.prominence < threshold or any(x.prominence > s.prominence for x in nb): continue
            t = s.nucleus_s; st = float(np.clip((s.prominence - threshold) / 1.5 + .4, .3, 1.))
            if beats and t - beats[-1][0] < min_gap_s:
                if s.prominence > beats[-1][2]: beats[-1] = (t, st, s.prominence)
                continue
            beats.append((t, st, s.prominence))
        if not final and beats and ncand < n and syllables[ncand].onset_s - beats[-1][0] < min_gap_s:
            beats = beats[:-1]                                         # could still be replaced by an undecided candidate
        return [(t, s) for t, s, _ in beats]
    return pick


def extract(pcm16, chars, plain, lang, T, variant, st=None, final=True):
    """feat [T, 89] float32 of the (prefix) audio; prefix = units complete by T/50 s."""
    F.features = ORIG_FEATURES if variant == 'v0' else fixed_features(st)
    F.pick_beats = committed_beats(final) if variant == 'v2' else ORIG_PICK
    try:
        dur = T / 50. - 1e-3
        cs, ue = unit_end(chars, plain, lang)
        use = [dict(c, onset_s=min(c['onset_s'], dur - 1e-3), offset_s=min(c['offset_s'], dur)) for c, e in zip(cs, ue)
               if (e <= T / 50. if not final else c['onset_s'] < dur - 0.02)]
        x16 = np.asarray(pcm16[:T * 320])
        try:
            rec = F.Recording.from_clip(x16, 16000, use, plain, lang, T)
            return np.asarray(rec.segment(0, T)['feat'], np.float32)
        except (ValueError, IndexError):                               # no complete unit yet: mel only
            from scipy.signal import resample_poly
            a16 = np.concatenate([x16, np.zeros(max(0, T * 320 - len(x16)), np.int16)])
            p24 = resample_poly(a16.astype(np.float32) / 32768., F.gl_build.FS, 16000)
            return np.concatenate([F.gl_build.logmel(p24, T, F.mel()), np.zeros((T, 9), np.float32)], 1).astype(np.float16).astype(np.float32)
    finally:
        F.features, F.pick_beats = ORIG_FEATURES, ORIG_PICK


def speaker_stats(pool, spk, lang, audio, exclude, n=6, max_s=60.):
    """Fixed prominence statistics of a voice from OTHER recordings of it (dev speakers are not in train: the other dev
    clips of the speaker, never the evaluated one) -- the deployment analogue is a calibration set of the TTS voice."""
    E, Fv, Dd = [], [], []
    def rec(syllables, pcm, fs=24000, cache=None):
        pcm = np.asarray(pcm, np.float32)
        for s in syllables:
            e, f = F_measure(s, pcm, fs); E.append(e); Fv.append(f); Dd.append(s.offset_s - s.onset_s)
        return ORIG_FEATURES(syllables, pcm, fs)
    k = 0
    for c, xs in pool:
        if c == exclude: continue
        if xs[0].get('speaker') != spk or xs[0]['lang'] != lang or not xs[0].get('dir'): continue
        start = int(xs[0]['offset']); fr = min(sum(int(x['frames']) for x in xs), int(max_s * 50)) // 16 * 16
        plain, chars = chars_of(xs, start)
        F.features = rec
        try:
            dur = fr / 50. - 1e-3
            ch = [dict(cc, onset_s=min(cc['onset_s'], dur - 1e-3), offset_s=min(cc['offset_s'], dur)) for cc in chars if cc['offset_s'] <= dur]
            F.Recording.from_clip(np.asarray(audio[start * 320:(start + fr) * 320]), 16000, ch, plain, lang, fr)
        except Exception as e:
            print('stats skip', c, e, flush=True); continue
        finally:
            F.features = ORIG_FEATURES
        k += 1
        if k >= n: break
    E, Fv, Dd = map(np.array, (E, Fv, Dd))
    lf = np.log(Fv[Fv > 0]); Ev = E[E > -80]; Dv = Dd[Dd > 0]
    return [float(Ev.mean()), float(Ev.std()), float(lf.mean()), float(lf.std()), float(Dv.mean()), float(Dv.std())], k, len(E)


def cmp_rows(a, b, fs):
    """normalised |a - b| per column ([n, 89])."""
    return np.abs(a - b) / fs[None]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    ck = sys.argv[2] if len(sys.argv) > 2 else None; vae_p = sys.argv[3] if len(sys.argv) > 3 else None
    stats = json.loads(Path(D + '/STATS_v36_1.json').read_text())
    fm, fs = np.asarray(stats['f_mean'], np.float32), np.asarray(stats['f_std'], np.float32)
    cl = clips_val(); audio = np.load(Path(str(V) + '-audio16') / 'AUDIO16.npy', mmap_mode='r')
    vf = np.load(V / 'train_feat.npy', mmap_mode='r')
    model = None
    if ck:
        import torch
        torch.set_num_threads(2)
        from scmg.data import View, WindowDataset, clip_lm_capacity
        from scmg.evaluate import load
        from scmg.infer import Streamer
        from scmg.common import U80, CHUNK_FRAMES
        from scmg.sim import ARMS
        model, vae = load(ck, vae_p, 'cuda')
        view = View(str(V), 'val', D + '/text-e5-v36_1', stats, mmap=True, tok_cache=D + '/tok-qwen25-v36_1')
        qm = np.asarray(stats['q_mean'], np.float32)
        print('cuda device', torch.cuda.get_device_name(0), os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid', os.getpid(), flush=True)
    report = dict(clips=[], protocol=dict(H_frames=H_FR, prefix_rule='pcm[:t_k] + units with offset <= t_k', ticks='t_k = 16k + 8'))
    st_cache = {}
    for ci in PICK:
        c, xs = cl['val'][ci]; lang = xs[0]['lang']; spk = xs[0].get('speaker')
        start = int(xs[0]['offset']); L = sum(int(x['frames']) for x in xs); nch = L // 16; T = nch * 16
        pcm16 = np.asarray(audio[start * 320:(start + T) * 320]); plain, chars = chars_of(xs, start)
        key = (spk, lang, c)
        if key not in st_cache: st_cache[key] = speaker_stats(cl['val'], spk, lang, audio, c)
        st = st_cache[key][0]
        t0 = time.time()
        full = {v: extract(pcm16, chars, plain, lang, T, v, st, final=True) for v in ('v0', 'v1', 'v2')}
        viewf = np.asarray(vf[start:start + T], np.float32)
        ticks = [min(T, 16 * k + H_FR) for k in range(nch)]
        pref = {v: [] for v in full}
        agg = {v: dict(front=[], interior=[], diff_front=[], diff_interior=[]) for v in full}
        for tk in ticks:
            for v in full:
                p = extract(pcm16, chars, plain, lang, tk, v, st, final=(tk >= T))
                pref[v].append(p)
                d = cmp_rows(p, full[v][:tk], fs)
                fr0 = max(0, tk - 25)
                agg[v]['front'].append(d[fr0:]); agg[v]['interior'].append(d[:fr0])
        crow = dict(ci=ci, clip=c, lang=lang, speaker=spk, seconds=T / 50, n_ticks=len(ticks), units=len(chars),
                    speaker_stats=dict(zip(['E_mean', 'E_std', 'logf0_mean', 'logf0_std', 'dur_mean', 'dur_std'], st), clips=st_cache[key][1], syllables=st_cache[key][2]),
                    extract_s=round(time.time() - t0, 1), features={})
        for v in full:
            fr = np.concatenate(agg[v]['front']); it = np.concatenate(agg[v]['interior'])
            r = {}
            for name, d in (('front_0.5s', fr), ('interior', it)):
                r[name] = dict(rows=len(d), mel_mean=float(d[:, :80].mean()), mel_max=float(d[:, :80].max()) if len(d) else 0.,
                               mel_rows_changed=float((d[:, :80].max(1) > 1e-3).mean()) if len(d) else 0.,
                               **{f'{n}_mean': float(d[:, 80 + j].mean()) for j, n in enumerate(NAMES)},
                               **{f'{n}_rows_changed': float((d[:, 80 + j] > 1e-3).mean()) for j, n in enumerate(NAMES)})
            dv = cmp_rows(full[v], viewf, fs)                         # vs the training-view (recording-scope) features
            r['full_vs_view'] = dict(mel_mean=float(dv[:, :80].mean()), mel_max=float(dv[:, :80].max()),
                                     **{f'{n}_mean': float(dv[:, 80 + j].mean()) for j, n in enumerate(NAMES)},
                                     **{f'{n}_rows_changed': float((dv[:, 80 + j] > 1e-3).mean()) for j, n in enumerate(NAMES)})
            nb = lambda f: int((f[:, 85] > 0).sum())
            r['beats_full'] = nb(full[v]); r['beats_view'] = nb(viewf)
            crow['features'][v] = r
        # global rate cap activity (v0 pick_beats with vs without the cap, full utterance)
        print(json.dumps(dict(ci=ci, clip=c, T=T, extract_s=crow['extract_s'],
                              v0_front=crow['features']['v0']['front_0.5s']['prominence_mean'], v0_int=crow['features']['v0']['interior']['prominence_mean'])), flush=True)
        if model is not None:
            import torch
            ds = WindowDataset(view, T, max_tokens=256, max_ids=clip_lm_capacity(view, ci))
            s = ds.sample(ci, 0, np.random.default_rng(ci))
            cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if k in s}
            assert np.allclose(s['feat'], (viewf - fm) / fs, atol=1e-3), 'view features / clip alignment mismatch'
            norm = lambda f: (f - fm) / fs
            kw = dict(H=2 * U80, steps=10, standby=qm, ctx_t=model.ctx_t, cfg_w=2.5, text_mode='x2', x2_rate=15., x2_lag=.15,
                      deploy_timing=True, window_lm_text=True, trim_start=True, cfg_null_fix=True)

            def run(feat_full, per_tick=None, seed=0):
                sm = Streamer(model, vae, norm(feat_full), cond, stats, seed=seed, **kw)
                for k in range(nch):
                    if per_tick is not None:
                        f = np.zeros_like(feat_full); f[:ticks[k]] = per_tick[k]
                        sm.feat = torch.as_tensor(norm(f), device='cuda').float()
                    sm.tick()
                sm.flush()
                return torch.stack(sm.frames).cpu().numpy(), np.array(sm.timing)
            gens = {}
            gens['view_s0'], tm = run(viewf); gens['view_s1'], _ = run(viewf, seed=1)
            for v in full:
                gens[f'{v}_full'], _ = run(full[v]); gens[f'{v}_online'], _ = run(full[v], pref[v])
            gens['v0_full_s1'], _ = run(full['v0'], seed=1)
            np.savez_compressed(out / f'gen_{ci}.npz', **gens)
            def dq(a, b):
                n = min(len(a), len(b)); d = np.abs(a[:n] - b[:n])
                return dict(arms_mean=float(d[:, ARMS].mean()), arms_p95=float(np.percentile(d[:, ARMS], 95)), arms_max=float(d[:, ARMS].max()),
                            all34_mean=float(d.mean()), waist_head_mean=float(d[:, :4].mean()))
            crow['motion_rad'] = {
                'v0_online_vs_v0_full (causality, as trained)': dq(gens['v0_online'], gens['v0_full']),
                'v1_online_vs_v1_full (fixed prominence stats)': dq(gens['v1_online'], gens['v1_full']),
                'v2_online_vs_v2_full (+ committed beats, no cap)': dq(gens['v2_online'], gens['v2_full']),
                'v0_full_vs_view (clip vs recording scope)': dq(gens['v0_full'], gens['view_s0']),
                'v1_full_vs_view': dq(gens['v1_full'], gens['view_s0']),
                'v2_full_vs_view': dq(gens['v2_full'], gens['view_s0']),
                'v2_online_vs_view (deployable vs training features)': dq(gens['v2_online'], gens['view_s0']),
                'v0_online_vs_view': dq(gens['v0_online'], gens['view_s0']),
                'noise_floor_view_s0_vs_s1': dq(gens['view_s0'], gens['view_s1']),
                'noise_floor_v0full_s0_vs_s1': dq(gens['v0_full'], gens['v0_full_s1']),
            }
            crow['tick_ms_p50'] = float(np.percentile(tm[1:], 50) * 1000)
            print(json.dumps(dict(ci=ci, motion={k: round(v['arms_mean'], 4) for k, v in crow['motion_rad'].items()})), flush=True)
        report['clips'].append(crow)
        (out / 'REPORT.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('FEAT_CAUSALITY_DONE', flush=True)


if __name__ == '__main__':
    main()
