"""v3.6 training-view export, one PART per source group (merge_parts.py concatenates parts into the MX28/train_s3 data dir).

BEAT2 rows (v3.5 manifests, recording-level prosody) go through the audited unified-train-A export_v31.process_row, with the
speaker-style r2 export_source.process wrapper for beat2_en_additional (clip-relative word times).  Every other source uses
the generic clip path: q37.npy + audio16k.wav (exactly T*320 samples) + chars.json -> feat89.Recording.from_clip
(clip-scope prosody, flagged prosody_scope='clip').  Rows with empty text keep logmel + VAD speech flag with zero prosody
(prosody_scope='clip_no_text'), so a missing ASR transcript never drops the motion.

Per item extra fields used by train_v1.py: source_token, speed_id (0 null / 1 ok / 2 not ok), spk{path,sha256,valid,policy}.
usage: export_part.py --manifest M.jsonl [...] --sources a,b --splits train[,dev] --out PART [--eval-dev DIR] [--speed-table T]
       [--spk-table T] [--include-not-admitted] [--limit N] [--shard i/n]
"""
import argparse, collections, hashlib, json, sys, time
from pathlib import Path
import numpy as np

A = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025')
R2 = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/speaker-style-emb-r2-20261030/code')
sys.path.insert(0, str(A)); sys.path.insert(1, str(R2))
import ex002_3f as X  # noqa: E402
import export_v31 as E  # noqa: E402

SEALED = ('testset-primary-20260929', 'heldout-t1test-sealed-20260929')
TOKENS = ['null', 'human_mocap_beat2', 'human_mocap_embody', 'human_video_bili', 'human_video_streamer', 'semsynth_emage', 'semsynth_splice']
REAL_TOKENS = TOKENS[1:5]; SYNTH_TOKENS = TOKENS[5:]
SPK_MODEL_SHA = 'a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73'


def sha(p):
    p = Path(p)
    if any(s in str(p) or s in str(p.resolve()) for s in SEALED): raise PermissionError('sealed path refused')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def token_of(r):
    s = r['source']
    if s.startswith('beat2'): return 'human_mocap_beat2'
    t = r.get('source_token')
    if t not in TOKENS[1:]: raise ValueError(f'{r["clip_id"]}: unknown source_token {t}')
    return t


def speed_id(sp):
    if not sp or sp.get('speed_ok_0p70') is None: return 0
    return 1 if sp['speed_ok_0p70'] else 2


def spk_of(r, spk_table):
    st = r.get('style') or {}
    e = st.get('spk_emb')
    if r['source'].startswith('beat2'):
        t = spk_table.get(r['clip_id'])
        if t is None: return dict(path=None, sha256=None, valid=False, policy='missing_in_archive')
        e = t['spk_emb']; valid = bool(t.get('spk_valid')); pol = t.get('spk_policy')
        if e is not None:
            # archive vectors were extracted from the audio the archive row pointed to; v3.4/v3.5 repaired float-WAV audio of
            # some BEAT2 clips, so a vector only counts when its recorded audio SHA equals this row's audio SHA
            meta = json.loads(Path(e['path']).with_suffix('.json').read_text())
            if meta.get('audio_sha256') != r['audio']['sha256']:
                return dict(path=None, sha256=None, valid=False, policy='archive_audio_differs_from_v35_audio')
    else:
        valid = bool(st.get('spk_valid')) and e is not None; pol = st.get('spk_policy')
    if e is None: return dict(path=None, sha256=None, valid=False, policy=pol or 'none')
    if e.get('model_sha256') != SPK_MODEL_SHA or e.get('dim') != 192: raise ValueError(f'{r["clip_id"]}: speaker model/dim')
    return dict(path=e['path'], sha256=e['sha256'], valid=valid, policy=pol)


MAX_PIECE = 1000      # 20 s at 50 Hz (design §4.2: clips <= 20 s); also keeps text far below the 512-token text-encoder limit
MIN_PIECE = 100       # 2 s
NULL_TEXT = '。'      # placeholder for clips without a transcript; train_v1.py masks the text memory of these items (text_null)


def pieces(T, chars):
    """Cut [0,T) into <= MAX_PIECE frame pieces; cuts at the widest inter-word gap inside [s+MIN_PIECE, s+MAX_PIECE]."""
    if T <= MAX_PIECE: return [(0, T)]
    gaps = []
    cs = sorted(chars, key=lambda c: float(c['onset_s']))
    for x, y in zip(cs, cs[1:]):
        gaps.append((float(x['offset_s']), float(y['onset_s'])))
    out, s = [], 0
    while T - s > MAX_PIECE:
        cand = [((b - a), int(round((a + b) / 2 * 50))) for a, b in gaps if s + MIN_PIECE <= (a + b) / 2 * 50 <= s + MAX_PIECE and b >= a]
        e = max(cand)[1] if cand else s + MAX_PIECE
        out.append((s, e)); s = e
    if T - s >= MIN_PIECE or not out: out.append((s, T))
    else: out[-1] = (out[-1][0], T) if T - out[-1][0] <= MAX_PIECE + MIN_PIECE else out[-1]
    return out


def piece_features(pcm, sr, chars, plain, lang, T, vad, F):
    """feat89 for one piece: clip-scope prosody from the chars; without chars logmel + VAD flag and zero prosody."""
    al = []
    if chars:
        try:
            rec = F.Recording.from_clip(pcm, sr, chars, plain, lang, T); seg = rec.segment(0, T)
        except ValueError:
            seg = None
        if seg is not None:
            from experiments.rsi.eval.resolve import significant
            units = seg['units']; uon = np.array([u['onset_s'] for u in units]) if units else np.zeros(0)
            for c in sorted(chars, key=lambda c: float(c['onset_s'])):
                j = int(np.argmin(np.abs(uon - float(c['onset_s'])))) if len(uon) else -1
                u = units[j] if j >= 0 and abs(uon[j] - float(c['onset_s'])) < 2e-3 else None
                sig = [(int(c['index']) + k, ch) for k, ch in enumerate(c['char']) if significant(ch)]
                a_, b_ = float(c['onset_s']), float(c['offset_s'])
                for k, (pos, ch) in enumerate(sig):
                    al.append(dict(index=pos, char=plain[pos], onset_s=round(a_ + (b_ - a_) * k / len(sig), 4), offset_s=round(a_ + (b_ - a_) * (k + 1) / len(sig), 4),
                                   prominence=u['prominence'] if u else None, energy_db=u['energy_db'] if u else None, f0_hz=u['f0_hz'] if u else None))
            if al: return seg['feat'], al, seg['speech'], 'clip', plain
    from scipy.signal import resample_poly
    logmel = F.gl_build.logmel(resample_poly(pcm.astype(np.float32) / 32768, 3, 2), T, F.mel())
    feat = np.concatenate([logmel, np.zeros((T, 8), np.float32), vad[:, None]], 1).astype(np.float16)
    on = np.flatnonzero(vad > .5)
    # one placeholder character (the text encoder needs >= 1 aligned token); the trainer masks this item's text memory
    null_al = [dict(index=0, char=NULL_TEXT, onset_s=0.0, offset_s=0.02, prominence=None, energy_db=None, f0_hz=None)]
    return feat, null_al, ([float(on[0] / 50), float(on[-1] / 50)] if len(on) else [0., 0.]), 'clip_no_text', NULL_TEXT


def process_clip(r, F, sf):
    """Generic (non-BEAT2) clip: q37 + exact-length clip audio + chars.json -> one or more <= 20 s pieces."""
    T = int(r['n_frames'])
    if sha(r['motion']['path']) != r['motion']['sha256']: raise ValueError(f'{r["clip_id"]}: motion sha')
    q = np.load(r['motion']['path'])
    if q.shape != (T, 37) or not np.isfinite(q).all(): raise ValueError(f'{r["clip_id"]}: motion shape/finite')
    res = X.storage_residuals(q)
    if res['lift_max_abs'] > 1e-6 or res['mimic_max_abs'] > 1e-5: raise ValueError(f'{r["clip_id"]}: 37->34 not lossless {res}')
    if sha(r['audio']['path']) != r['audio']['sha256']: raise ValueError(f'{r["clip_id"]}: audio sha')
    pcm, sr = sf.read(r['audio']['path'], dtype='int16')
    if sr != 16000 or pcm.ndim != 1: raise ValueError(f'{r["clip_id"]}: audio must be mono 16 kHz')
    o = int(round((r['audio'].get('offset_s') or 0.) * sr)); pcm = np.asarray(pcm[o:o + T * 320], np.int16)
    if len(pcm) < T * 320: pcm = np.concatenate([pcm, np.zeros(T * 320 - len(pcm), np.int16)])
    plain = (r['text'] or {}).get('plain') or ''; lang = (r['text'] or {}).get('lang') or 'zh'
    ct = (r['text'] or {}).get('char_times')
    chars = json.loads(Path(ct).read_text()) if ct and plain else []
    chars = [c for c in chars if 0 <= float(c['onset_s']) < float(c['offset_s']) <= T / 50 + .021]
    for c in chars:
        i = int(c['index'])
        if plain[i:i + len(c['char'])] != c['char']: raise ValueError(f'{r["clip_id"]}: char index {c}')
    vad = np.load(r['self_speaking']['path']).astype(np.float32).reshape(-1)[:T] if r.get('self_speaking') else np.zeros(T, np.float32)
    if len(vad) < T: vad = np.concatenate([vad, np.zeros(T - len(vad), np.float32)])
    fv = np.load(r['frame_valid']['path']) if r.get('frame_valid') else None
    q34 = X.to34(q).astype(np.float32); out = []
    cuts = pieces(T, chars)
    for k, (s0, s1) in enumerate(cuts):
        n = s1 - s0; t0 = s0 / 50
        sub = [c for c in chars if s0 / 50 <= float(c['onset_s']) < s1 / 50]
        if sub:
            i0 = min(int(c['index']) for c in sub); last = max(sub, key=lambda c: int(c['index'])); i1 = int(last['index']) + len(last['char'])
            ptxt = plain[i0:i1]
            sub = [dict(index=int(c['index']) - i0, char=c['char'], onset_s=round(float(c['onset_s']) - t0, 4),
                        offset_s=round(min(float(c['offset_s']) - t0, n / 50), 4)) for c in sub]
            sub = [c for c in sub if c['offset_s'] > c['onset_s']]
        else:
            ptxt = ''
        feat, al, speech, scope, text = piece_features(pcm[s0 * 320:s1 * 320], sr, sub, ptxt, lang, n, vad[s0:s1], F)
        if feat.shape != (n, 89): raise ValueError(f'{r["clip_id"]}: feat shape {feat.shape}')
        out.append(dict(row=r, piece=(k, len(cuts), s0, s1), q34=q34[s0:s1], feat=feat, a16=pcm[s0 * 320:s1 * 320], al=al, speech=speech,
                        fv=float(np.mean(fv[s0:s1])) if fv is not None else None, scope=scope, item_dir=str(Path(r['motion']['path']).parent),
                        pv=dict(item=str(Path(r['motion']['path']).parent), a=s0, b=s1), text=text, text_null=scope == 'clip_no_text'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', nargs='+', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sources', required=True); ap.add_argument('--splits', default='train')
    ap.add_argument('--eval-dev', type=Path); ap.add_argument('--speed-table', type=Path); ap.add_argument('--spk-table', type=Path)
    ap.add_argument('--include-not-admitted', action='store_true', help='non-BEAT2 rows: export training_ready rows even when training_admitted is false (explicit gap rows)')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--limit-per-source', type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    import feat89 as F
    import soundfile as sf
    import export_source as S          # r2 wrapper: beat2_en_additional clip-relative word times (pure function, E.process_row inside)
    sources, splits = set(a.sources.split(',')), set(a.splits.split(','))
    speed = {}
    if a.speed_table:
        for l in a.speed_table.read_text().splitlines():
            if l.strip(): x = json.loads(l); speed[x['clip_id']] = x
    spk_table = {}
    if a.spk_table:
        for l in a.spk_table.read_text().splitlines():
            if l.strip():
                x = json.loads(l); st = x.get('style') or {}
                spk_table[x['clip_id']] = dict(spk_emb=st.get('spk_emb'), spk_valid=st.get('spk_valid'), spk_policy=st.get('spk_policy'))
    rows, manifests, skipped = [], [], collections.Counter()
    for m in a.manifest:
        manifests.append(dict(path=str(m), sha256=sha(m)))
        for l in m.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r['source'] not in sources or r['split'] not in splits: continue
            if r['source'].startswith('beat2'):
                if r.get('training_ready') is not True: skipped['not_training_ready'] += 1; continue
            elif r.get('training_admitted') is not True:
                # explicit gap rows (training_admitted false, gaps[] listed) only on request, and only with a physics PASS
                if not a.include_not_admitted: skipped['not_admitted'] += 1; continue
                if (r.get('gates') or {}).get('physics') != 'PASS': skipped['gap_row_without_physics_pass'] += 1; continue
                skipped['gap_rows_included'] += 1
            if r['license']['training_use'] == 'forbidden': skipped['license_forbidden'] += 1; continue
            rows.append(r)
    if a.limit_per_source:
        c = collections.Counter(); keep = []
        for r in rows:
            k = (r['source'], r['split'])
            if c[k] < a.limit_per_source: keep.append(r)
            c[k] += 1
        rows = keep
    if a.limit: rows = rows[:a.limit]
    if len({r['clip_id'] for r in rows}) != len(rows): raise ValueError('duplicate clip_id')
    a.out.mkdir(parents=True, exist_ok=False)
    done, info, errors, rec_cache = [], [], [], {}
    by_rec = collections.OrderedDict()
    for r in rows: by_rec.setdefault(Path(r['motion']['path']).parent.parent.name if r['source'].startswith('beat2') else r['clip_id'], []).append(r)
    for gi, (rk, rs) in enumerate(by_rec.items()):
        for r in rs:
            try:
                if r['source'].startswith('beat2'):
                    E.check_row(r); d = S.process(r, rec_cache, F, sf, info); d['text'] = r['text']['plain']; ds = [d]
                else:
                    ds = process_clip(r, F, sf)
            except Exception as exc:          # per-row refusal is recorded, never silently dropped
                errors.append(dict(clip_id=r['clip_id'], source=r['source'], error=f'{type(exc).__name__}: {exc}')); continue
            done.extend(ds)
        rec_cache.clear()
        if gi % 200 == 0: print(f'group {gi + 1}/{len(by_rec)} done={len(done)} errors={len(errors)} t={time.time() - t0:.0f}s', flush=True)
    exp = [d for d in done if d['row']['split'] == 'test']   # 11-09 copy for the single EX002 test scoring run (only change vs export_part.py)
    nfr = lambda d: d['pv']['b'] - d['pv']['a'] if 'piece' in d else d['row']['n_frames']
    N = sum(nfr(d) for d in exp)
    Q = np.lib.format.open_memmap(a.out / 'q34.npy', 'w+', np.float32, (N, 34))
    Fm = np.lib.format.open_memmap(a.out / 'feat.npy', 'w+', np.float16, (N, 89))
    Au = np.lib.format.open_memmap(a.out / 'a16.npy', 'w+', np.int16, (N * 320,))
    items, off = [], 0
    for d in exp:
        r, T = d['row'], nfr(d)
        Q[off:off + T] = d['q34']; Fm[off:off + T] = d['feat']
        blk = np.zeros(T * 320, np.int16); blk[:min(T * 320, len(d['a16']))] = d['a16'][:T * 320]; Au[off * 320:(off + T) * 320] = blk
        iid = r['clip_id'] + (f'#p{d["piece"][0]}' if 'piece' in d and d['piece'][1] > 1 else '')
        kk = E.key_of(iid); ad = a.out / 'alignments' / kk; ad.mkdir(parents=True)
        E.write_json(ad / 'alignment.json', d['al'], indent=None)
        sp = speed.get(r['clip_id'], {}).get('speed') if r['source'].startswith('beat2') else r.get('speed')
        items.append(dict(id=iid, clip_id=r['clip_id'], piece=d.get('piece'), text_null=bool(d.get('text_null')), dir=f'alignments/{kk}', offset=off, frames=T, speech=d['speech'], events=[], text=d['text'],
                          conv_key=r['split_group'], speaker=(r['style'] or {}).get('speaker'), lang=(r['text'] or {}).get('lang'),
                          clip=Path(d['pv']['item']).name, clip_start_s=d['pv']['a'] / 50., source=r['source'], data_source=r['source'],
                          label_source=None, split_hint=r['split'], prev_clip_id=r.get('prev_clip_id'), prosody_scope=d['scope'],
                          frame_valid_frac=d['fv'], low_conf_frac=(r['quality'] or {}).get('low_conf_frac'),
                          manifest_motion=r['motion'], manifest_audio=r['audio'],
                          source_token=token_of(r), speed_id=speed_id(sp), speed=sp, spk=spk_of(r, spk_table),
                          training_admitted=bool(r.get('training_admitted', r['source'].startswith('beat2') and r.get('training_ready'))),
                          has_semantic=r.get('has_semantic'), rhythm_source=r.get('rhythm_source'), absmax=int(np.abs(blk.astype(np.int32)).max())))
        off += T
    Q.flush(); Fm.flush(); Au.flush(); del Q, Fm, Au
    with (a.out / 'items.jsonl').open('w') as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + '\n')
    view = None
    if a.eval_dev:
        sel = [d for d in done if d['row']['split'] == 'dev']
        a.eval_dev.mkdir(parents=True, exist_ok=False); eitems, off = [], 0
        Fe = np.lib.format.open_memmap(a.eval_dev / 'test_feat.npy', 'w+', np.float16, (sum(d['row']['n_frames'] for d in sel), 89))
        for d in sel:
            r, T = d['row'], d['row']['n_frames']; kk = E.key_of(r['clip_id']); idir = a.eval_dev / 'items' / kk; idir.mkdir(parents=True)
            E.write_json(idir / 'alignment.json', d['al'], indent=None); sf.write(idir / 'audio.wav', d['a16'], 16000, subtype='PCM_16')
            np.save(idir / 'q37_gt.npy', np.load(r['motion']['path'])); Fe[off:off + T] = d['feat']
            eitems.append(dict(id=r['clip_id'], text=d['text'], dir=f'items/{kk}', offset=off, frames=T, speech=d['speech'], split=r['split'],
                               speaker=(r['style'] or {}).get('speaker'), lang=r['text']['lang'], source=r['source'],
                               source_item=Path(d['pv']['item']).name, item_dir=d['pv']['item'], source_frames=[d['pv']['a'], d['pv']['b']],
                               retargeter_version=r.get('retargeter_version'), speed=speed.get(r['clip_id'], {}).get('speed')))
            off += T
        Fe.flush(); del Fe
        E.write_json(a.eval_dev / 'test.json', eitems)
        E.write_json(a.eval_dev / 'EVAL_VIEW.json', dict(split='dev', items=len(eitems), frames=off, hours=round(off / 180000, 4), manifests=manifests,
                                                         files={n: sha(a.eval_dev / n) for n in ('test.json', 'test_feat.npy')}))
        view = str(a.eval_dev)
    hours = collections.defaultdict(float); nrow = collections.Counter()
    for it in items: hours[f"{it['source']}|{it['split_hint']}"] += it['frames'] / 180000; nrow[f"{it['source']}|{it['split_hint']}"] += 1
    part = dict(schema='v36-export-part-1', rows_selected=len(rows), exported=len(items), frames=N, skipped=dict(skipped), errors=errors,
                error_count=len(errors), unit_notes=len(info), hours={k: round(v, 6) for k, v in hours.items()}, rows=dict(nrow),
                scopes=dict(collections.Counter(it['prosody_scope'] for it in items)), manifests=manifests, eval_dev=view,
                exporter=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)), feature_code=F.code_digests(), seconds=round(time.time() - t0, 1),
                args={k: str(v) for k, v in vars(a).items()},
                files={n: sha(a.out / n) for n in ('q34.npy', 'feat.npy', 'a16.npy', 'items.jsonl')})
    E.write_json(a.out / 'PART.json', part)
    E.write_json(a.out / 'NOTES.json', info[:20000])
    print(json.dumps({k: part[k] for k in ('exported', 'frames', 'error_count', 'hours', 'seconds')}))


if __name__ == '__main__':
    main()
