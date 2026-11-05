"""Training-matched causal inputs (design v5 section 3, "训练匹配"): for every clip of a G1 view split, the prefix CAM++
speaker vector and the causal (v2) feat89 at the stream times 1 / 2 / 4 / 8 s and the whole clip (five grades), with the
same code as the streaming inference (scmg/causal_feat.py):
  speaker  grade g s: SpeakerPrefix.at(min(g s, clip) of PCM) (speech_window 10 s crop, < 1 s -> empty: valid 0);
           'full' = the view rule (whole file, >= 0.5 s), identical to the view's SPK_EMB row
  feat89   grade g s: prefix_feat89(Tg = min(50 g, T) frames): pcm[:Tg], words with end <= Tg / 50, fixed prosody statistics
           (--pstats), committed beats (final only at Tg = T); 'full' = Tg = T (the causal features once everything arrived)
Outputs (per shard, <out>/shard_<i>of<n>/): SPK_PREFIX.npy [N, 5, 192] float32, SPK_VALID.npy [N, 5] uint8,
FEAT_PREFIX.npy [R, 89] float16 (all grades of all clips, concatenated), INDEX.json {id: {"clip_row": i, "frames": T,
"feat": [[row, n], x5]}}, META.json (rules, grades, inputs, timing); `merge` concatenates the shards into <out>/.
    python scripts/echog_g1/precompute_causal.py run --view V --split train --audio-dir D/audio --words-dir D/annotations/words \
        --pstats scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json --out OUT [--shard i/n] [--limit N] [--workers W]
    python scripts/echog_g1/precompute_causal.py merge --out OUT
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GRADES = [1., 2., 4., 8., None]          # seconds; None = whole clip
_W = {}


def _init(campplus, pstats):
    from scmg import causal_feat as CF
    _W.update(CF=CF, enc=CF.speaker_encoder(campplus), pst=CF.load_pstats(pstats))


def one(job):
    stem, T, lang, audio_dir, words_dir = job
    CF, enc, pst = _W['CF'], _W['enc'], _W['pst']
    words = CF.clip_words(Path(words_dir) / f'{stem}.json', T)
    pcm16, wave = CF.clip_audio(Path(audio_dir) / f'{stem}.wav')
    sp = CF.SpeakerPrefix(wave, enc)
    spk = np.zeros((len(GRADES), 192), np.float32); val = np.zeros(len(GRADES), np.uint8); feats = []
    for j, g in enumerate(GRADES):
        if g is None:
            try: spk[j] = sp.full(); val[j] = 1
            except ValueError: pass
            Tg = T
        else:
            e, v = sp.at(min(int(round(g * CF.SR)), T * CF.HOP)); spk[j] = e; val[j] = int(v)
            Tg = min(T, int(round(g * CF.FPS)))
        feats.append(np.asarray(CF.prefix_feat89(pcm16, words, lang, T, Tg, pst), np.float16))
    return stem, T, spk, val, feats


def cmd_run(a):
    v = Path(a.view); items = json.loads((v / 'train.json').read_text()); sp = json.loads((v / 'SPLIT.json').read_text())
    keep = set(sp['train_ids'] if a.split == 'train' else sp['val_ids'])
    its = sorted([it for it in items if it['id'] in keep], key=lambda it: it['id'])
    if a.limit: its = its[:a.limit]
    si, sn = (int(x) for x in a.shard.split('/')); its = its[si::sn]
    jobs = [(it['id'], int(it['frames']), it['lang'], a.audio_dir, a.words_dir) for it in its]
    out = Path(a.out) / f'shard_{si}of{sn}'; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); res = []
    if a.workers <= 1:
        _init(a.campplus, a.pstats); res = [one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.workers, initializer=_init, initargs=(a.campplus, a.pstats)) as pool:
            for i, r in enumerate(pool.imap(one, jobs, chunksize=4)):
                res.append(r)
                if (i + 1) % 500 == 0: print(f'{i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    index, rows, row = {}, [], 0
    for i, (stem, T, spk, val, feats) in enumerate(res):
        ent = []
        for f in feats: ent.append([row, len(f)]); rows.append(f); row += len(f)
        index[stem] = dict(clip_row=i, frames=T, feat=ent)
    np.save(out / 'SPK_PREFIX.npy', np.stack([r[2] for r in res]) if res else np.zeros((0, 5, 192), np.float32))
    np.save(out / 'SPK_VALID.npy', np.stack([r[3] for r in res]) if res else np.zeros((0, 5), np.uint8))
    np.save(out / 'FEAT_PREFIX.npy', np.concatenate(rows) if rows else np.zeros((0, 89), np.float16))
    (out / 'INDEX.json').write_text(json.dumps(index))
    sec = time.time() - t0
    meta = dict(grades_s=GRADES, view=str(a.view), split=a.split, shard=a.shard, clips=len(res), audio_dir=a.audio_dir,
                words_dir=a.words_dir, pstats=a.pstats, pstats_values=_W.get('pst') or json.loads(Path(a.pstats).read_text()),
                campplus=a.campplus, workers=a.workers, seconds=round(sec, 1), s_per_clip=round(sec / max(1, len(res)), 4),
                audio_s=round(sum(T for _, T, *_ in res) / 50., 1),
                rule=__doc__.split('Outputs')[0].strip())
    (out / 'META.json').write_text(json.dumps(meta, indent=1))
    print('PRECOMPUTE_DONE', len(res), f'{sec:.1f}s', f'{meta["s_per_clip"]} s/clip', out)


def cmd_merge(a):
    out = Path(a.out); sh = sorted(out.glob('shard_*of*'))
    spk, val, feat, index, row, ci = [], [], [], {}, 0, 0
    for d in sh:
        s, vv, f = np.load(d / 'SPK_PREFIX.npy'), np.load(d / 'SPK_VALID.npy'), np.load(d / 'FEAT_PREFIX.npy')
        for stem, e in json.loads((d / 'INDEX.json').read_text()).items():
            if stem in index: raise ValueError(f'{stem} in two shards')
            index[stem] = dict(clip_row=e['clip_row'] + ci, frames=e['frames'], feat=[[r + row, n] for r, n in e['feat']])
        spk.append(s); val.append(vv); feat.append(f); row += len(f); ci += len(s)
    np.save(out / 'SPK_PREFIX.npy', np.concatenate(spk)); np.save(out / 'SPK_VALID.npy', np.concatenate(val))
    np.save(out / 'FEAT_PREFIX.npy', np.concatenate(feat)); (out / 'INDEX.json').write_text(json.dumps(index))
    m0 = json.loads((sh[0] / 'META.json').read_text())
    (out / 'META.json').write_text(json.dumps(dict(m0, shard='merged', shards=len(sh), clips=ci), indent=1))
    print('MERGED', len(sh), 'shards', ci, 'clips', row, 'rows')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'merge']); ap.add_argument('--out', required=True)
    ap.add_argument('--view'); ap.add_argument('--split', default='train')
    ap.add_argument('--audio-dir'); ap.add_argument('--words-dir'); ap.add_argument('--pstats')
    ap.add_argument('--campplus', default='/home/xr-25b7/bulk/models-ro/campplus')
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == 'run' else cmd_merge(a)


if __name__ == '__main__':
    main()
