"""Training-matched causal inputs (design v5 section 3, "训练匹配"): for every clip of a G1 view split, the prefix CAM++
speaker vector and the causal (v2) feat89 at the stream times 1 / 2 / 4 / 8 s and the whole clip (five grades), with the
same code as the streaming inference (scmg/causal_feat.py):
  speaker  grade g s: SpeakerPrefix.at(min(g s, clip) of PCM) (speech_window 10 s crop, < 1 s -> empty: valid 0);
           'full' = the view rule (whole file, >= 0.5 s), identical to the view's SPK_EMB row
  feat89   grade g s: prefix_feat89(Tg = min(50 g, T) frames): pcm[:Tg], words with end <= Tg / 50, fixed prosody statistics
           (--pstats), committed beats (final only at Tg = T); 'full' = Tg = T (the causal features once everything arrived)
  units    (11-05 训练匹配) the prosody units of the 'full' recording (causal_feat.unit_table: onset, offset, prominence,
           energy dB, f0 Hz, end of the unit's word) -> the training data loader rebuilds the prefix prosody columns of
           any stream time with causal_feat.front_prosody (bitwise = prefix_feat89, scripts/tests/test_causal_train.py)
Outputs (per shard, <out>/shard_<i>of<n>/): SPK_PREFIX.npy [N, 5, 192] float32, SPK_VALID.npy [N, 5] uint8,
FEAT_PREFIX.npy [R, 89] float16 (all grades of all clips, concatenated), UNITS.npy [U, 6] float64, INDEX.json {id:
{"clip_row": i, "frames": T, "feat": [[row, n], x5], "units": [row, n], "tend": [row, n]}}, TOKEN_END.npy [K] float32 (end of
the last view word of every e5 text-cache row of the clip, clip-relative s; needs --text-cache), META.json (rules, grades, inputs, timing);
`merge` concatenates the shards into <out>/. --view may list several views (comma-separated): the union of their split
items by id (e.g. the held-out-speaker view and the main view, whose train ids contain it).
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


def _init(campplus, pstats, tokenizer):
    from scmg import causal_feat as CF
    _W.update(CF=CF, enc=CF.speaker_encoder(campplus), pst=CF.load_pstats(pstats), tok=CF.e5_tokenizer(tokenizer) if tokenizer else None)


def one(job):
    stem, T, lang, audio_dir, words_dir, text, n_tok = job[:7]
    CF, enc, pst = _W['CF'], _W['enc'], _W['pst']
    rec = job[7] if len(job) > 7 else None
    if rec is None:
        words = CF.clip_words(Path(words_dir) / f'{stem}.json', T)
        pcm16, wave = CF.clip_audio(Path(audio_dir) / f'{stem}.wav')
    else:                                            # 11-07 EX002 recording: its items' audio / words concatenated in order
        words, pcms, waves, t0 = [], [], [], 0
        for iid, n_it in rec:
            p_, w_ = CF.clip_audio(Path(audio_dir) / f'{iid}.wav'); p_ = CF.pad_pcm(p_, n_it); w_ = p_.astype(np.float32) / 32768.
            for w in CF.clip_words(Path(words_dir) / f'{iid}.json', n_it):
                words.append(dict(w, start_time=w['start_time'] + t0 / 50., end_time=w['end_time'] + t0 / 50.))
            pcms.append(p_); waves.append(w_); t0 += n_it
        pcm16, wave = np.concatenate(pcms), np.concatenate(waves)
    sp = CF.SpeakerPrefix(wave, enc)
    spk = np.zeros((len(GRADES), 192), np.float32); val = np.zeros(len(GRADES), np.uint8); feats = []; units = None
    for j, g in enumerate(GRADES):
        if g is None:
            try: spk[j] = sp.full(); val[j] = 1
            except ValueError: pass
            Tg = T
        else:
            e, v = sp.at(min(int(round(g * CF.SR)), T * CF.HOP)); spk[j] = e; val[j] = int(v)
            Tg = min(T, int(round(g * CF.FPS)))
        if g is None:
            f, u = CF.prefix_feat89(pcm16, words, lang, T, Tg, pst, units=True); units = CF.unit_table(u, words, lang)
        else:
            f = CF.prefix_feat89(pcm16, words, lang, T, Tg, pst)
        feats.append(np.asarray(f, np.float16))
    # token end (s, clip-relative): the end of the LAST view word each e5 cache row overlaps -- a row is 'voiced' once it
    # has ended (causal_feat.token_visible: every word of its span); NaN rows = no word
    tend = np.full(n_tok, np.nan, np.float32)
    if rec is not None:                              # per item (item-relative s, the loader's causal_tend rule)
        tends = []
        for (iid, n_it), (txt, nt) in zip(rec, job[8]):
            te = np.full(nt, np.nan, np.float32); iw = CF.clip_words(Path(words_dir) / f'{iid}.json', n_it)
            if nt and iw:
                try: span = CF.token_word_span(txt, iw, lang, _W['tok'], nt); te = np.asarray([iw[j]['end_time'] for j in span[:, 1]], np.float32)
                except ValueError as e: print(f'TEND_FAILED {iid}: {e}', flush=True)
            tends.append((iid, te))
        return stem, T, spk, val, feats, units, tends
    if n_tok and words:
        try:
            span = CF.token_word_span(text, words, lang, _W['tok'], n_tok)
            tend = np.asarray([words[j]['end_time'] for j in span[:, 1]], np.float32)
        except ValueError as e:                      # item text != joined words / token count: rows stay NaN (never voiced)
            print(f'TEND_FAILED {stem}: {e}', flush=True)
    return stem, T, spk, val, feats, units, tend


def write(out, res):
    """SPK_PREFIX / SPK_VALID / FEAT_PREFIX / UNITS / TOKEN_END / INDEX.json of one() results (also used by the tests)."""
    from scmg.causal_feat import UNIT_COLS
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    index, rows, row, urows, urow, trows, trow = {}, [], 0, [], 0, [], 0
    for i, (stem, T, spk, val, feats, units, tend) in enumerate(res):
        ent = []
        for f in feats: ent.append([row, len(f)]); rows.append(f); row += len(f)
        if isinstance(tend, list):                   # 11-07 recording: per-item token ends under the item ids
            index[stem] = dict(clip_row=i, frames=T, feat=ent, units=[urow, len(units)], tend=[trow, 0], recording=[iid for iid, _ in tend])
            for iid, te in tend:
                index[iid] = dict(tend=[trow, len(te)], of=stem); trows.append(te); trow += len(te)
        else:
            index[stem] = dict(clip_row=i, frames=T, feat=ent, units=[urow, len(units)], tend=[trow, len(tend)])
            trows.append(tend); trow += len(tend)
        urows.append(units); urow += len(units)
    np.save(out / 'SPK_PREFIX.npy', np.stack([r[2] for r in res]) if res else np.zeros((0, 5, 192), np.float32))
    np.save(out / 'SPK_VALID.npy', np.stack([r[3] for r in res]) if res else np.zeros((0, 5), np.uint8))
    np.save(out / 'FEAT_PREFIX.npy', np.concatenate(rows) if rows else np.zeros((0, 89), np.float16))
    np.save(out / 'UNITS.npy', np.concatenate(urows) if urows else np.zeros((0, len(UNIT_COLS)), np.float64))
    np.save(out / 'TOKEN_END.npy', np.concatenate(trows) if trows else np.zeros(0, np.float32))
    (out / 'INDEX.json').write_text(json.dumps(index))


def cmd_run(a):
    its = {}
    for vv in a.view.split(','):
        v = Path(vv); items = json.loads((v / 'train.json').read_text()); sp = json.loads((v / 'SPLIT.json').read_text())
        keep = set(sp['train_ids'] if a.split == 'train' else sp['val_ids'])
        for it in items:
            if it['id'] in keep:
                if it['id'] in its and int(its[it['id']]['frames']) != int(it['frames']): raise ValueError(f"{it['id']}: frames differ between views")
                its[it['id']] = it
    its = sorted(its.values(), key=lambda it: it['id'])
    tix = json.loads((Path(a.text_cache) / 'TEXT_INDEX.json').read_text()) if a.text_cache else {}
    ntok = lambda it: 0 if it.get('text_null') or it['id'] not in tix else int(tix[it['id']][1])
    if a.recording:                                  # 11-07 EX002: one job per recording (view clip key 'clip'), items in offset order
        g = {}
        for it in its: g.setdefault(it['clip'], []).append(it)
        recs = sorted(g.items())
        if a.limit: recs = recs[:a.limit]
        si, sn = (int(x) for x in a.shard.split('/')); recs = recs[si::sn]
        jobs = []
        for k, lst in recs:
            lst = sorted(lst, key=lambda it: it['offset'])
            for x, y in zip(lst, lst[1:]):
                if x['offset'] + x['frames'] != y['offset']: raise ValueError(f'{k}: items not contiguous in the view')
            jobs.append((k, sum(int(it['frames']) for it in lst), lst[0]['lang'], a.audio_dir, a.words_dir, '', 0,
                         [(it['id'], int(it['frames'])) for it in lst], [(it.get('text', ''), ntok(it)) for it in lst]))
    else:
        if a.limit: its = its[:a.limit]
        si, sn = (int(x) for x in a.shard.split('/')); its = its[si::sn]
        jobs = [(it['id'], int(it['frames']), it['lang'], a.audio_dir, a.words_dir, it.get('text', ''), ntok(it)) for it in its]
    out = Path(a.out) / f'shard_{si}of{sn}'; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); res = []
    if a.workers <= 1:
        _init(a.campplus, a.pstats, a.tokenizer); res = [one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.workers, initializer=_init, initargs=(a.campplus, a.pstats, a.tokenizer)) as pool:
            for i, r in enumerate(pool.imap(one, jobs, chunksize=4)):
                res.append(r)
                if (i + 1) % 500 == 0: print(f'{i + 1}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    write(out, res)
    sec = time.time() - t0
    meta = dict(grades_s=GRADES, view=str(a.view), split=a.split, shard=a.shard, clips=len(res), audio_dir=a.audio_dir,
                words_dir=a.words_dir, pstats=a.pstats, unit_cols=list(__import__('scmg.causal_feat', fromlist=['x']).UNIT_COLS), text_cache=a.text_cache,
                tokenizer=a.tokenizer, pstats_values=_W.get('pst') or json.loads(Path(a.pstats).read_text()),
                campplus=a.campplus, workers=a.workers, seconds=round(sec, 1), s_per_clip=round(sec / max(1, len(res)), 4),
                audio_s=round(sum(T for _, T, *_ in res) / 50., 1),
                rule=__doc__.split('Outputs')[0].strip())
    (out / 'META.json').write_text(json.dumps(meta, indent=1))
    print('PRECOMPUTE_DONE', len(res), f'{sec:.1f}s', f'{meta["s_per_clip"]} s/clip', out)
    if a.workers > 1:                                # 11-07: the spawn pool (+ onnxruntime threads) hung at interpreter exit after the
        sys.stdout.flush(); sys.stderr.flush()       # outputs were written (dlc307caz5jmsiec): leave without the atexit teardown
        os._exit(0)


def cmd_merge(a):
    out = Path(a.out); sh = sorted(out.glob('shard_*of*'))
    spk, val, feat, units, tends, index, row, urow, trow, ci = [], [], [], [], [], {}, 0, 0, 0, 0
    for d in sh:
        s, vv, f = np.load(d / 'SPK_PREFIX.npy'), np.load(d / 'SPK_VALID.npy'), np.load(d / 'FEAT_PREFIX.npy')
        u = np.load(d / 'UNITS.npy'); te = np.load(d / 'TOKEN_END.npy')
        for stem, e in json.loads((d / 'INDEX.json').read_text()).items():
            if stem in index: raise ValueError(f'{stem} in two shards')
            if 'feat' not in e:                      # 11-07 recording item entry: token ends only
                index[stem] = dict(e, tend=[e['tend'][0] + trow, e['tend'][1]]); continue
            index[stem] = dict(clip_row=e['clip_row'] + ci, frames=e['frames'], feat=[[r + row, n] for r, n in e['feat']],
                               units=[e['units'][0] + urow, e['units'][1]], tend=[e['tend'][0] + trow, e['tend'][1]],
                               **({'recording': e['recording']} if 'recording' in e else {}))
        spk.append(s); val.append(vv); feat.append(f); units.append(u); tends.append(te)
        row += len(f); urow += len(u); trow += len(te); ci += len(s)
    np.save(out / 'SPK_PREFIX.npy', np.concatenate(spk)); np.save(out / 'SPK_VALID.npy', np.concatenate(val))
    np.save(out / 'FEAT_PREFIX.npy', np.concatenate(feat)); np.save(out / 'UNITS.npy', np.concatenate(units))
    np.save(out / 'TOKEN_END.npy', np.concatenate(tends))
    (out / 'INDEX.json').write_text(json.dumps(index))
    m0 = json.loads((sh[0] / 'META.json').read_text())
    for d in sh[1:]:                                   # every shard: the same inputs and rules
        m = json.loads((d / 'META.json').read_text())
        for k in ('grades_s', 'view', 'split', 'audio_dir', 'words_dir', 'pstats_values', 'campplus', 'unit_cols', 'text_cache', 'tokenizer'):
            if m.get(k) != m0.get(k): raise ValueError(f'{d.name}: META {k} differs from {sh[0].name}')
    sec = [json.loads((d / 'META.json').read_text())['seconds'] for d in sh]
    (out / 'META.json').write_text(json.dumps(dict(m0, shard='merged', shards=len(sh), clips=ci, shard_seconds_max=max(sec),
                                                   shard_seconds_sum=round(sum(sec), 1)), indent=1))
    print('MERGED', len(sh), 'shards', ci, 'clips', row, 'rows')


def cmd_slim(a):
    """<to>: the training copy of a merged precompute without FEAT_PREFIX (the data loader reads the view's mel, equal to the
    prefix mel row for row -- View.load_causal checks it on 64 clips whenever FEAT_PREFIX is present, so run that check on
    the full precompute once); INDEX / SPK_* / UNITS / TOKEN_END copied, META marked slim."""
    import shutil
    src, dst = Path(a.out), Path(a.to); dst.mkdir(parents=True, exist_ok=True)
    for f in ('INDEX.json', 'SPK_PREFIX.npy', 'SPK_VALID.npy', 'UNITS.npy', 'TOKEN_END.npy'): shutil.copy(src / f, dst / f)
    m = json.loads((src / 'META.json').read_text()); m.update(slim=f'no FEAT_PREFIX (from {src})')
    (dst / 'META.json').write_text(json.dumps(m, indent=1)); print('SLIM', dst)


def cmd_fixwords(a):
    """11-06: rewrite UNITS.npy column word_end with causal_feat.unit_word (by time) -- precomputes written before that fix
    mapped units to words by char index, which drifts after apostrophe / hyphen words. Needs --words-dir. In place
    (the old file is kept as UNITS.charmap.npy)."""
    from scmg import causal_feat as CF
    d = Path(a.out); idx = json.loads((d / 'INDEX.json').read_text()); U = np.load(d / 'UNITS.npy'); U2 = U.copy(); nd = 0
    for stem, e in idx.items():
        u0, un = e['units']
        if not un: continue
        words = CF.clip_words(Path(a.words_dir) / f'{stem}.json', e['frames'])
        U2[u0:u0 + un, 5] = [words[i]['end_time'] for i in CF.unit_word(U[u0:u0 + un, 6], words)]
        nd += int((U2[u0:u0 + un, 5] != U[u0:u0 + un, 5]).sum())
    if not (d / 'UNITS.charmap.npy').exists(): np.save(d / 'UNITS.charmap.npy', U)
    np.save(d / 'UNITS.npy', U2)
    m = json.loads((d / 'META.json').read_text()); m['units_word_map'] = f'by time (causal_feat.unit_word), {nd} of {len(U)} rows changed'
    (d / 'META.json').write_text(json.dumps(m, indent=1)); print('FIXWORDS', nd, 'of', len(U))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'merge', 'slim', 'fixwords']); ap.add_argument('--out', required=True); ap.add_argument('--to')
    ap.add_argument('--view'); ap.add_argument('--split', default='train')
    ap.add_argument('--audio-dir'); ap.add_argument('--words-dir'); ap.add_argument('--pstats')
    ap.add_argument('--campplus', default='/home/xr-25b7/bulk/models-ro/campplus')
    ap.add_argument('--text-cache', default=None, help='e5 text cache of the view (TEXT_INDEX.json): token end times')
    ap.add_argument('--tokenizer', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/multilingual-e5-base')
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--recording', action='store_true', help='11-07 EX002: one entry per recording (view clip), its items concatenated')
    a = ap.parse_args()
    {'run': cmd_run, 'merge': cmd_merge, 'slim': cmd_slim, 'fixwords': cmd_fixwords}[a.cmd](a)


if __name__ == '__main__':
    main()
