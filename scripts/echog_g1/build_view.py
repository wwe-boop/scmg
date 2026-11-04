"""ECHO-G (BEAT2 -> Unitree G1) as an SCMG training view, same raw data / splits / G1 39-d motion as the ECHO-G paper.

  extract   per stem: motion 30 -> 50 FPS (scmg/g1.py), feat89 audio features, word-level alignment, CAM++ speaker vector
  assemble  views: <out>/ (train_drop: train + a fixed take-level dev holdout ~2 %) and <out>/test/ (val_common = the
            ECHO-G benchmark set, its own arrays: training never opens them; SPLIT.json there has ONLY test_ids)

Inputs (read-only): <data>/{motion_39d_30fps,audio,annotations/words,transcripts}/<stem>.*, <meta>/splits/*.txt,
<meta>/audit/condition/per_clip_condition_motion_audit.csv (aligned_frames = the official common audio/motion prefix,
<= 600 frames @30; the motion used here is exactly that prefix, as in ECHO-G training / benchmark).

Per stem, the SAME code paths as the v3.6 views (unified-ex002-data-v3_6-20261101/train/export_part.py + merge_parts.py):
  audio features   feat89.Recording(words=<ECHO-G word times>, pcm=int16 16 kHz) -> .segment(0, T50) -- the BEAT2 human
                   path with the word list of the packaged clip (prosody z-scored over the clip: prosody_scope 'clip', the
                   official audio is per clip). No speech unit -> logmel + zero prosody, text_null (export_part fallback).
  text alignment   segment()['alignment'] (character onsets, index into segment text) -> alignments/<stem>/alignment.json;
                   scmg/prep_text.py then builds the e5 token cache from it unchanged
  speaker          speaker-style-emb-20261030 code (backend.read_audio, audio_window.speech_window(10 s), CAM++ onnx
                   a6ac6a63..., kaldi fbank, L2) on the stem's own audio; source token human_mocap_beat2 (BEAT2 mocap)
  motion           robot_repr[:aligned_frames] (physical units) -> 50 FPS, float32, channel order unchanged
    build_view.py extract  --data D --meta M --work W [--workers 20] [--limit N]
    build_view.py assemble --data D --meta M --work W --out VIEW [--dev-frac 0.02 --seed 20261104]
"""
import argparse, collections, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1]))
U = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan')
FEAT89_DIR = U / 'unified-train-A-20261025'
SPK_CODE = U / 'speaker-style-emb-20261030/code'
SPK_MODEL_SHA = 'a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73'
TOKENS = ['null', 'human_mocap_beat2', 'human_mocap_embody', 'human_video_bili', 'human_video_streamer', 'semsynth_emage', 'semsynth_splice']
SEALED = ('testset-primary-20260929', 'heldout-t1test-sealed-20260929')


def sha(p):
    p = Path(p)
    if any(s in str(p) or s in str(p.resolve()) for s in SEALED): raise PermissionError('sealed path refused')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def take_of(stem):
    return stem.rsplit('_utt_', 1)[0]


def lang_of(stem):
    return {'english': 'en', 'chinese': 'zh'}[stem.split('_', 1)[0]]


def speaker_of(stem):
    return '_'.join(stem.split('_')[1:3])


def aligned(meta):
    rows = {}
    with open(Path(meta) / 'audit/condition/per_clip_condition_motion_audit.csv') as f:
        for r in csv.DictReader(f):
            if r['error']: raise ValueError(f"audit error row {r['stem']}: {r['error']}")
            rows[r['stem']] = int(r['aligned_frames'])
    return rows


# ------------------------------------------------------------------ extract (one process per worker)
_W = {}


def _init(model_path):
    sys.path.insert(0, str(FEAT89_DIR)); sys.path.insert(1, str(SPK_CODE))
    import feat89 as F
    import soundfile as sf
    from backend import SpeakerEncoder, read_audio
    from audio_window import speech_window
    enc = SpeakerEncoder('campplus', model_path, device='cpu')
    if enc.weight_sha != SPK_MODEL_SHA: raise ValueError('CAM++ weight SHA')
    _W.update(F=F, sf=sf, enc=enc, read_audio=read_audio, speech_window=speech_window)


def null_features(F, pcm, T):
    """export_part.piece_features fallback (no speech unit): logmel + zero prosody + zero speech flag."""
    from scipy.signal import resample_poly
    logmel = F.gl_build.logmel(resample_poly(pcm.astype(np.float32) / 32768, 3, 2), T, F.mel())
    return np.concatenate([logmel, np.zeros((T, 9), np.float32)], 1).astype(np.float16)


def extract_one(args):
    stem, data, n30, out = args
    o = Path(out) / f'{stem}.npz'
    if o.exists() and Path(str(o) + '.json').exists(): return json.loads(Path(str(o) + '.json').read_text())
    import torch
    from scmg import g1
    F, sf = _W['F'], _W['sf']
    data = Path(data)
    mp = data / 'motion_39d_30fps' / f'{stem}.pt'; ap = data / 'audio' / f'{stem}.wav'; wp = data / 'annotations/words' / f'{stem}.json'
    d = torch.load(mp, map_location='cpu', weights_only=True)
    rr = d['robot_repr'].double().numpy()
    if d.get('fps') != 30.0 or d.get('representation_schema') != 'projecthermes-g1-39d-standard-v2': raise ValueError(f'{stem}: motion schema')
    if rr.shape[1] != 39 or len(rr) < n30 or not np.isfinite(rr).all(): raise ValueError(f'{stem}: motion {rr.shape} < {n30}')
    r30 = rr[:n30]
    q50 = g1.to50(r30).astype(np.float32); T = len(q50)
    info = sf.info(str(ap))
    if info.samplerate != 16000 or info.channels != 1: raise ValueError(f'{stem}: audio {info.samplerate} Hz x {info.channels}')
    x, _ = sf.read(str(ap), dtype='float32')
    absmax = float(np.abs(x).max()) if len(x) else 0.
    pcm = np.clip(np.round(x * 32768.), -32768, 32767).astype(np.int16)
    pcm = pcm[:T * 320]
    if len(pcm) < T * 320: pcm = np.concatenate([pcm, np.zeros(T * 320 - len(pcm), np.int16)])
    W = json.loads(wp.read_text())
    if W.get('schema') != 'echo-g-word-token-annotations-v1' or W.get('clip_id') != stem: raise ValueError(f'{stem}: words schema')
    words = [dict(text=str(w['text']).strip(), start_time=float(w['start']), end_time=float(w['end'])) for w in W['words']]
    words = [w for w in words if w['text'] and np.isfinite(w['start_time']) and np.isfinite(w['end_time']) and w['end_time'] >= w['start_time']]
    lang = lang_of(stem)
    feat, al, speech, text, null = None, [], [0., 0.], '', True
    err = None
    if words:
        try:
            rec = F.Recording(None, None, lang, T, words=words, pcm_fs=(pcm, 16000))
            seg = rec.segment(0, T)
            # text + character onsets from the Recording's own character_alignment (rec.text / rec.al), NOT seg['text'] /
            # seg['alignment']: for English the unit->character map of feat89 counts "'" and "-" that resolve.significant
            # skips, so seg['text'] loses one trailing character per apostrophe / hyphen ("... it's t"). The BEAT2 items of
            # the v3.6 views likewise take the full manifest text. Prosody features (unit times) are unaffected.
            text = rec.text; dur = T / 50.
            al = [dict(index=int(c['index']), char=c['char'], onset_s=round(float(c['onset_s']), 4), offset_s=round(min(float(c['offset_s']), dur), 4))
                  for c in rec.al if float(c['onset_s']) < dur]
            feat, speech, null = seg['feat'], seg['speech'], False
            if not al: raise ValueError('empty alignment')
        except ValueError as e:
            err = f'{type(e).__name__}: {e}'; feat = None; null = True
    if feat is None:
        feat = null_features(F, pcm, T); al = [dict(index=0, char='。', onset_s=0.0, offset_s=0.02, prominence=None, energy_db=None, f0_hz=None)]
        text = '。'; speech = [0., 0.]
    if feat.shape != (T, 89) or not np.isfinite(feat.astype(np.float32)).all(): raise ValueError(f'{stem}: feat {feat.shape}')
    # speaker vector: same crop / fbank / model as the archive (speaker-style-emb-20261030 extract.py)
    spk, spk_err = None, None
    try:
        w = _W['read_audio'](dict(path=str(ap)))
        chunk, crop = _W['speech_window'](w, 10.)
        if len(chunk) < 8000: raise ValueError('less than 0.5s audio')
        spk = _W['enc'].encode([chunk])[0]
    except Exception as e:
        spk_err = f'{type(e).__name__}: {e}'
    np.savez(o, q50=q50, feat=feat, spk=spk if spk is not None else np.zeros(192, np.float32))
    rec = dict(stem=stem, n30=n30, n50=T, motion_frames30=int(len(rr)), real_num_frames=int(d.get('real_num_frames', len(rr))),
               audio_samples=int(len(x)), audio_seconds=len(x) / 16000., audio_subtype=info.subtype, audio_absmax=absmax,
               audio_clipped=bool(absmax > 1.0), words=len(words), text=text, text_null=null, feat_error=err, speech=speech,
               alignment=al, spk_valid=spk is not None, spk_error=spk_err, lang=lang,
               transcript=(data / 'transcripts' / f'{stem}.txt').read_text(encoding='utf-8').strip(),
               source=W.get('source'), motion_sha256=sha(mp), audio_sha256=sha(ap), words_sha256=sha(wp))
    Path(str(o) + '.json').write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def cmd_extract(a):
    al = aligned(a.meta)
    stems = [s for f in ('train_drop', 'val_common') for s in (Path(a.meta) / 'splits' / f'{f}.txt').read_text().split()]
    if a.limit: stems = stems[:a.limit] if not a.stems else stems
    if a.stems: stems = [s for s in a.stems.split(',')]
    out = Path(a.work) / 'items'; out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, a.data, al[s], str(out)) for s in stems]
    import multiprocessing as mp
    t0 = time.time(); done = 0; errors = []
    with mp.get_context('spawn').Pool(a.workers, initializer=_init, initargs=(a.campplus,)) as pool:
        for r in pool.imap_unordered(_safe, jobs, chunksize=4):
            done += 1
            if 'fatal' in r: errors.append(r)
            if done % 500 == 0 or done == len(jobs): print(f'{done}/{len(jobs)} errors={len(errors)} {time.time() - t0:.0f}s', flush=True)
    (Path(a.work) / 'EXTRACT_ERRORS.json').write_text(json.dumps(errors, indent=1))
    print('EXTRACT_DONE', len(jobs), 'errors', len(errors))


def _safe(job):
    try:
        return extract_one(job)
    except Exception as e:
        import traceback
        return dict(stem=job[0], fatal=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-1500:])


# ------------------------------------------------------------------ assemble
def write_view(root, stems, recs, work, split_ids, rule, extra_meta):
    from scmg import g1
    root.mkdir(parents=True, exist_ok=False)
    N = sum(recs[s]['n50'] for s in stems)
    Q = np.lib.format.open_memmap(root / 'train_q.npy', 'w+', np.float32, (N, 39))
    Fm = np.lib.format.open_memmap(root / 'train_feat.npy', 'w+', np.float16, (N, 89))
    items, spk, cond, off = [], [], {}, 0
    for s in stems:
        r = recs[s]; z = np.load(Path(work) / 'items' / f'{s}.npz'); T = r['n50']
        if z['q50'].shape != (T, 39) or z['feat'].shape != (T, 89): raise ValueError(f'{s}: arrays')
        Q[off:off + T] = z['q50']; Fm[off:off + T] = z['feat']
        ad = root / 'alignments' / s; ad.mkdir(parents=True)
        (ad / 'alignment.json').write_text(json.dumps(r['alignment'], ensure_ascii=False))
        v = z['spk'].astype(np.float32)
        valid = bool(r['spk_valid'])
        if valid and (v.shape != (192,) or abs(np.linalg.norm(v) - 1) > 1e-3): raise ValueError(f'{s}: speaker vector not unit 192-d')
        cond[s] = [len(spk), int(valid), TOKENS.index('human_mocap_beat2'), 0]; spk.append(v if valid else np.zeros(192, np.float32))
        src = r['source'] or {}
        items.append(dict(id=s, clip_id=s, piece=None, text_null=r['text_null'], dir=f'alignments/{s}', offset=off, frames=T,
                          speech=r['speech'], events=[], text=r['text'], conv_key=take_of(s), speaker=speaker_of(s), lang=r['lang'],
                          clip=s, clip_start_s=0.0, source='echog_beat2', data_source='echog_beat2', label_source=None,
                          split_hint=extra_meta['split_of'][s], prev_clip_id=None, prosody_scope='clip' if not r['text_null'] else 'clip_no_text',
                          frame_valid_frac=None, low_conf_frac=None, source_token='human_mocap_beat2', speed_id=0, speed=None,
                          manifest_motion=dict(path=f'motion_39d_30fps/{s}.pt', sha256=r['motion_sha256']),
                          manifest_audio=dict(path=f'audio/{s}.wav', sha256=r['audio_sha256'], offset_s=0.0),
                          spk=dict(valid=valid, policy='own_clip_audio_campplus' if valid else f"invalid: {r['spk_error']}"),
                          echog=dict(stem=s, take=take_of(s), recording=src.get('recording'), start_sec=src.get('start_sec'),
                                     end_sec=src.get('end_sec'), frames30=r['n30'], frames50=T, words=r['words'],
                                     words_sha256=r['words_sha256'], transcript=r['transcript'], audio_subtype=r['audio_subtype'],
                                     audio_absmax=r['audio_absmax'], feat_error=r['feat_error'])))
        off += T
    Q.flush(); Fm.flush(); del Q, Fm
    (root / 'cond').mkdir(); np.save(root / 'cond' / 'SPK_EMB.npy', np.stack(spk).astype(np.float32))
    (root / 'cond' / 'COND.json').write_text(json.dumps(dict(columns=['spk_row', 'spk_valid', 'source_id', 'speed_id'], tokens=TOKENS,
                                                             speed_ids={'0': 'null', '1': 'speed_ok_0p70', '2': 'speed_not_ok'},
                                                             spk_model_sha256=SPK_MODEL_SHA, items=cond)))
    (root / 'train.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
    (root / 'SPLIT.json').write_text(json.dumps(dict(split_ids, rule=rule), indent=1))
    (root / 'labels.jsonl').write_text('')
    return items, N


def cmd_assemble(a):
    from scmg import g1
    from scmg.data import compute_stats
    work = Path(a.work); out = Path(a.out)
    tr = (Path(a.meta) / 'splits/train_drop.txt').read_text().split(); te = (Path(a.meta) / 'splits/val_common.txt').read_text().split()
    if set(tr) & set(te) or len(set(tr)) != len(tr) or len(set(te)) != len(te): raise ValueError('official splits overlap / duplicates')
    if {take_of(s) for s in tr} & {take_of(s) for s in te}: raise ValueError('take in train_drop and val_common')
    recs = {}
    for s in tr + te:
        p = work / 'items' / f'{s}.npz.json'
        if not p.exists(): raise SystemExit(f'missing extraction for {s}')
        recs[s] = json.loads(p.read_text())
    al = aligned(a.meta)
    for s in tr + te:
        if recs[s]['n30'] != al[s]: raise ValueError(f'{s}: n30 != aligned_frames')
    # dev holdout: whole takes (all segments of a source recording) of train_drop, fixed seed, ~dev_frac of the segments
    takes = sorted({take_of(s) for s in tr}); rng = np.random.default_rng(a.seed); order = rng.permutation(len(takes))
    by_take = collections.Counter(take_of(s) for s in tr); dev_takes, n = set(), 0
    for i in order:
        if n >= a.dev_frac * len(tr): break
        dev_takes.add(takes[i]); n += by_take[takes[i]]
    dev = [s for s in tr if take_of(s) in dev_takes]; train = [s for s in tr if take_of(s) not in dev_takes]
    split_of = {**{s: 'train' for s in train}, **{s: 'dev' for s in dev}, **{s: 'test' for s in te}}
    rule = (f'train = ECHO-G splits/train_drop.txt minus a take-level dev holdout ({len(dev_takes)} of {len(takes)} takes, '
            f'numpy default_rng({a.seed}) permutation of the sorted take list, filled until >= {a.dev_frac:.0%} of the segments; '
            'dev only picks checkpoints); test = ECHO-G splits/val_common.txt, stored ONLY in test/ (never in this view)')
    items, N = write_view(out, tr, recs, work, dict(train_ids=train, val_ids=dev), rule, dict(split_of=split_of))
    titems, NT = write_view(out / 'test', te, recs, work, dict(train_ids=[], val_ids=[], test_ids=te),
                            'test = ECHO-G splits/val_common.txt (official benchmark set, 3,242 clips); train_ids / val_ids empty '
                            'on purpose: only View(split="test") reads it', dict(split_of=split_of))
    # statistics over TRAIN rows only (dev rows excluded), floor 1e-3 instead of the EX002 finger floor 0.03 rad
    rows = np.concatenate([np.arange(it['offset'], it['offset'] + it['frames']) for it in items if it['split_hint'] == 'train'])
    st = compute_stats(out, n=400000, seed=0, rows=rows, std_floor=1e-3)
    st.update(view=a.canonical, rows='train split only (dev holdout and test excluded)', std_floor=1e-3, n_sample=400000, seed=0)
    Path(a.stats).parent.mkdir(parents=True, exist_ok=True); Path(a.stats).write_text(json.dumps(st))
    # META.json
    lim = g1_limits(a.mjcf)
    hours = collections.defaultdict(float); segs = collections.Counter(); lang = collections.Counter()
    for it in items + titems:
        k = it['split_hint']; hours[k] += it['frames'] / 50 / 3600; segs[k] += 1; lang[(k, it['lang'])] += 1
    rr = [recs[s] for s in tr + te]
    meta = dict(fps=50, joints=g1.CHANNELS, channel_units=g1.UNITS,
                lower=[None] * 10 + [lim[n][0] for n in g1.JOINT_NAMES], upper=[None] * 10 + [lim[n][1] for n in g1.JOINT_NAMES],
                limits_source=dict(path='assets/unitree_g1/g1_mocap_29dof.xml (ECHO-G robot_assets/unitree_g1.tar)', sha256=sha(a.mjcf),
                                   note='joint ranges of the 29 hinge joints; root channels have no limits (null)'),
                channel_groups=dict(root=g1.ROOT, root_rot6d=g1.ROT6D, root_yaw_delta=[g1.YAW], root_vel=g1.VEL, legs=g1.LEGS,
                                    waist=g1.WAIST, arms=g1.ARMS, joints=g1.JOINTS),
                leg_mask=dict(default='off', legs=g1.LEGS, root=g1.ROOT,
                              train='scmg.train --vae-cfg {"dim_mask": [...]} (VAE: zeroed input, no loss, decoder output pinned to the '
                                    'training mean) [+ --sc {"dim_mask": [...]} for decoded-motion losses]',
                              eval='scripts/echog_g1/mask_dirs.py legs|legs_root: GT and predictions masked identically, then the '
                                   'official evaluator; report-only, the official benchmark always uses all 39 channels'),
                representation=dict(schema='projecthermes-g1-39d-standard-v2', layout='0:6 base 6D (first two rows of R, frame-0 '
                                    'yaw removed), 6 base yaw change rad/frame (frame 0 = 0), 7:10 yaw-local root velocity m/s, 10:39 G1 '
                                    'joints rad (canonical order)', units='physical (NOT normalised); normalise with the STATS json'),
                resampling=dict(src_fps=30, fps=50, rule='scmg/g1.py: joints + root velocity cubic spline; heading cumsum spline '
                                'then differenced; base orientation RotationSpline; n50 = round(n30*5/3); output via g1.to30(q50, n30)',
                                n50_from_n30='round(n30 * 5 / 3)', roundtrip_check='scripts/echog_g1/check_resample.py'),
                feat_dim=89, prosody=['onset', 'prominence', 'energy', 'logf0', 'voiced', 'beat', 'beat_left', 'beat_right'],
                feature_code=dict(feat89=str(FEAT89_DIR / 'feat89.py'), feat89_sha256=sha(FEAT89_DIR / 'feat89.py'),
                                  speaker_code=str(SPK_CODE), speaker_model_sha256=SPK_MODEL_SHA, builder=str(Path(__file__).resolve()),
                                  builder_sha256=sha(__file__)),
                source=dict(dataset='gaopusen/ECHO-G', revision='a1280e7524817e74f06c928285698378ba5d85bc (v0.2.0)',
                            splits={k: sha(Path(a.meta) / f'splits/{k}.txt') for k in ('train_drop', 'val_common')},
                            audit_csv_sha256=sha(Path(a.meta) / 'audit/condition/per_clip_condition_motion_audit.csv'),
                            motion='robot_repr[:aligned_frames] (official common prefix)'),
                counts=dict(segments=dict(segs), hours={k: round(v, 4) for k, v in hours.items()},
                            lang={f'{k}|{l}': c for (k, l), c in lang.items()}, frames50=dict(view=N, test=NT),
                            frames30=dict(train_drop=sum(recs[s]['n30'] for s in tr), val_common=sum(recs[s]['n30'] for s in te)),
                            dev_takes=sorted(dev_takes), text_null=sum(r['text_null'] for r in rr), spk_invalid=sum(not r['spk_valid'] for r in rr),
                            feat_errors=sum(bool(r['feat_error']) for r in rr), audio_float=sum(r['audio_subtype'] == 'FLOAT' for r in rr),
                            audio_clipped=sum(r['audio_clipped'] for r in rr),
                            audio_shorter_than_motion=sum(r['audio_samples'] < r['n50'] * 320 for r in rr)),
                stats=a.canonical_stats, view=a.canonical)
    for root in (out, out / 'test'):
        (root / 'META.json').write_text(json.dumps(dict(meta, part='view (train + dev)' if root == out else 'test (val_common)'), indent=1, ensure_ascii=False))
        files = {str(p.relative_to(root)): sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != 'FROZEN.json'}
        files.update({f'cond/{p.name}': sha(p) for p in sorted((root / 'cond').iterdir())})
        (root / 'FROZEN.json').write_text(json.dumps(dict(schema='echog-g1-view-v1', files=files, alignments_tree_sha256=sha_tree(root / 'alignments')), indent=1))
    print(json.dumps(meta['counts'], indent=1, ensure_ascii=False)[:3000])


def sha_tree(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file(): h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def g1_limits(mjcf):
    import xml.etree.ElementTree as ET
    out = {}
    for j in ET.parse(mjcf).getroot().iter('joint'):
        if j.get('name') and j.get('range'):
            lo, hi = (float(x) for x in j.get('range').split()); out[j.get('name')] = (lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['extract', 'assemble'])
    ap.add_argument('--data', default='/mnt/bulk/zehan/echog-dataset/ds'); ap.add_argument('--meta', default='/mnt/bulk/zehan/echog-dataset/official-meta')
    ap.add_argument('--work', required=True); ap.add_argument('--out'); ap.add_argument('--stats')
    ap.add_argument('--workers', type=int, default=20); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--stems', default='')
    ap.add_argument('--campplus', default='/home/xr-25b7/bulk/models-ro/campplus')
    ap.add_argument('--mjcf', default='/mnt/bulk/zehan/echog-dataset/assets-x/assets/unitree_g1/g1_mocap_29dof.xml')
    ap.add_argument('--canonical', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/echog-g1-v1', help='CPFS path recorded in META/STATS')
    ap.add_argument('--canonical-stats', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_echog_g1_v1.json')
    ap.add_argument('--dev-frac', type=float, default=0.02); ap.add_argument('--seed', type=int, default=20261104)
    a = ap.parse_args()
    cmd_extract(a) if a.cmd == 'extract' else cmd_assemble(a)


if __name__ == '__main__':
    main()
