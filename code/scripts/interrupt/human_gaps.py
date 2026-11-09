"""Human "speech ends -> hands come back" reference segments for the G1 interruption benchmark (11-05).

The packaged ECHO-G clips end exactly at their last word (0 s tail), so post-speech motion only exists in the BEAT2 takes
BETWEEN packaged utterances. For every packaged utterance u of a take (both splits listed in utterances.json):
  offset t_off = u.end (= its last word end, take time); next speech = start of the next packaged utterance of the take
  (or the take end); keep u if the silence lasts >= --min-gap s AND the take audio confirms it (dropped utterances are not in
  the table): share of 20 ms frames in [t_off + 0.25, t_off + min(gap, --post) - 0.1] louder than (median dB of the take's
  packaged speech - --vad-db) must be < --vad-frac.
Deterministic selection: candidates sorted by sha1(recording@t_off), at most --per-spk per speaker and --max per split.
Each kept segment [t_off - --pre, t_off + min(gap, --post)] of the BEAT2 SMPL-X take (30 fps) is retargeted to the ECHO-G
G1 39-d representation with the SAME GMR pipeline as the reproduced baselines (gmr-repro/scripts/smplx_npz_to_g1.py:
Y-up -> Z-up, GMR unitree_g1, height 1.66 + 0.1 beta0, Savitzky-Golay(7,2) on qpos, g1_repr.encode_pose; calibrated against
ECHO-G GT: joint MAE 0.013 rad). 10 IK warm-up passes on the first frame; the --pre seconds before the offset also absorb
the IK start transient.
    <gmr python> scripts/interrupt/human_gaps.py --utt utterances.json --beat2 <beat2 root> --out <dir> --shard i/n
writes <out>/seg/<id>.npz (robot_repr [T,39] float32, off = offset frame index) and <out>/index.<i>of<n>.jsonl.
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

R = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro')
ap = argparse.ArgumentParser()
ap.add_argument('--utt', required=True); ap.add_argument('--beat2', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--min-gap', type=float, default=2.6); ap.add_argument('--pre', type=float, default=1.5); ap.add_argument('--post', type=float, default=4.0)
ap.add_argument('--vad-db', type=float, default=25.); ap.add_argument('--vad-frac', type=float, default=0.05)
ap.add_argument('--per-spk', type=int, default=60); ap.add_argument('--max', default='train_drop:900,val_common:500')
ap.add_argument('--shard', default='0/1'); ap.add_argument('--select-only', action='store_true')
a = ap.parse_args()
out = Path(a.out); (out / 'seg').mkdir(parents=True, exist_ok=True)
rows = json.loads(Path(a.utt).read_text())
take = defaultdict(list)
for r in rows: take[(r['lang'], r['recording'])].append(r)
LDIR = dict(chinese='beat_chinese_v2.0.0', english='beat_english_v2.0.0')


def wav_db(lang, rec):
    import soundfile as sf
    p = Path(a.beat2, LDIR[lang], 'wave16k', f'{rec}.wav')
    if not p.exists(): return None
    y, sr = sf.read(str(p), dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    hop = sr // 50; n = len(y) // hop
    rms = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-6)          # 20 ms frames


def candidates():
    """deterministic candidate list (same in every shard): VAD-confirmed silences after packaged utterances"""
    cache = out / 'candidates.json'
    if cache.exists(): return json.loads(cache.read_text())
    cand = []
    for (lang, rec), us in sorted(take.items()):
        npz = Path(a.beat2, LDIR[lang], 'smplxflame_30', f'{rec}.npz')
        if not npz.exists(): continue
        db = wav_db(lang, rec)
        if db is None: continue
        nfr = int(np.load(npz, allow_pickle=True)['poses'].shape[0]); t_end = min(nfr / 30., len(db) / 50.)
        us = sorted(us, key=lambda r: r['start'])
        sp = np.concatenate([db[int(u['start'] * 50):int(u['end'] * 50)] for u in us])
        if len(sp) == 0: continue
        thr = float(np.median(sp)) - a.vad_db
        for i, u in enumerate(us):
            t_off = u['end']; nxt = us[i + 1]['start'] if i + 1 < len(us) else t_end
            gap = min(nxt, t_end) - t_off
            if gap < a.min_gap or t_off - a.pre < 0: continue
            post = min(gap, a.post)
            seg = db[int((t_off + 0.25) * 50):int((t_off + post - 0.1) * 50)]
            loud = float((seg > thr).mean()) if len(seg) else 1.
            if loud >= a.vad_frac: continue
            cand.append(dict(id=f'{rec}@{t_off:.2f}', lang=lang, recording=rec, speaker=u['speaker'], split=u['split'], stem=u['stem'],
                             t_off=t_off, gap=float(gap), post=float(post), loud=loud, take_end=t_end,
                             last_utt_of_take=i + 1 == len(us), utt_dur=u['end'] - u['start']))
    caps = dict(x.split(':') for x in a.max.split(','))
    sel = []
    for split in caps:
        cs = sorted([c for c in cand if c['split'] == split], key=lambda c: hashlib.sha1(c['id'].encode()).hexdigest())
        per = defaultdict(int)
        for c in cs:
            if per[c['speaker']] >= a.per_spk: continue
            per[c['speaker']] += 1; sel.append(c)
            if sum(per.values()) >= int(caps[split]): break
    rep = dict(n_candidates=len(cand), by_split={s: sum(c['split'] == s for c in cand) for s in caps},
               selected={s: sum(c['split'] == s for c in sel) for s in caps}, params=vars(a))
    tmp = cache.with_suffix('.tmp'); tmp.write_text(json.dumps(dict(report=rep, selected=sel, all=cand))); os.replace(tmp, cache)
    return json.loads(cache.read_text())


C = candidates()
print(json.dumps(C['report']), flush=True)
if a.select_only: sys.exit(0)

sys.path.insert(0, str(R / 'scripts'))
import torch
torch.set_num_threads(1)
import smplx_npz_to_g1 as S            # its _body_model / _retargeter / C (gmr_calib) / G (g1_repr) / smooth_qpos
import mujoco

si, sn = (int(x) for x in a.shard.split('/'))
mine = C['selected'][si::sn]
t0 = time.time(); fo = open(out / f'index.{si}of{sn}.jsonl', 'a')
cur = (None, None)
for k, c in enumerate(mine):
    op = out / 'seg' / f"{c['id']}.npz"
    if op.exists(): continue
    if cur[0] != c['recording']:
        cur = (c['recording'], np.load(Path(a.beat2, LDIR[c['lang']], 'smplxflame_30', f"{c['recording']}.npz"), allow_pickle=True))
    tk = cur[1]
    s = int(round((c['t_off'] - a.pre) * 30)); off = int(round(c['t_off'] * 30)) - s; e = int(round((c['t_off'] + c['post']) * 30))
    e = min(e, len(tk['poses']))
    d = dict(betas=np.asarray(tk['betas']), poses=np.asarray(tk['poses'][s:e], np.float32), trans=np.asarray(tk['trans'][s:e], np.float64),
             gender=str(tk['gender']) if 'gender' in tk else 'neutral', mocap_frame_rate=np.array(30))
    d = S.C.yup_to_zup(d); n = len(d['poses'])
    bm = S.smplx.create(str(R / 'weights/smplx_models'), 'smplx', gender=d['gender'], use_pca=False, batch_size=n).eval()
    human = S.C.build_human(d, bm, n)
    betas = np.asarray(d['betas'], np.float32).reshape(-1)
    gmr = S._retargeter(float(1.66 + 0.1 * betas[0]))
    for _ in range(10): gmr.retarget(human[0])
    qpos = np.stack([gmr.retarget(h) for h in human])
    jn = [mujoco.mj_id2name(gmr.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(gmr.model.njnt)]
    jmap = [next(i for i, x in enumerate(jn) if x in (nm + '_joint', nm)) for nm in S.G.G1_JOINT_NAMES]
    qs = S.smooth_qpos(qpos, 'savgol', (7, 2))
    rr = S.G.encode_pose(qs[:, :3], qs[:, 3:7], qs[:, [gmr.model.jnt_qposadr[j] for j in jmap]])
    tmp = op.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, robot_repr=rr.astype(np.float32), off=off, start_frame=s, end_frame=e)
    os.replace(tmp, op)
    fo.write(json.dumps(dict(c, off=off, frames=int(len(rr)))) + '\n'); fo.flush()
    if k % 20 == 0: print(f'{k + 1}/{len(mine)} {c["id"]} T={len(rr)} {time.time() - t0:.0f}s', flush=True)
print('GAPS_DONE', len(mine), f'{time.time() - t0:.0f}s', flush=True)
