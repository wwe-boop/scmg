"""Concatenate v3.6 export parts into one MX28/train_s3 data dir (+ -audio16), with the per-item condition bank.

usage: merge_parts.py --parts P1 P2 ... --out DATA
Layout written (train_s3 / preflight contract of export_v31.main, plus cond/):
  train_q.npy [N,34] f32, train_feat.npy [N,89] f16, alignments/, train.json, SPLIT.json (train = split train, val = fixed v3.2 dev),
  labels.jsonl (empty), VOCAB.json, META.json (export_v31.build_meta), FROZEN.json,
  cond/SPK_EMB.npy [n_items,192] f32 (zeros where invalid), cond/COND.json {id: [spk_row, spk_valid, source_id, speed_id]},
  DATA-audio16/{AUDIO16.npy, BLOCKS.jsonl, REPORT.json, FROZEN.json}
"""
import argparse, collections, json, shutil, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from export_part import A, E, X, TOKENS, REAL_TOKENS, SYNTH_TOKENS, SPK_MODEL_SHA, sha  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--parts', nargs='+', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--speed-table', type=Path, help='jsonl clip_id -> speed (BEAT2 v3.5 side table); fills items whose speed is null')
    a = ap.parse_args(); t0 = time.time()
    import feat89 as F
    parts = []
    for p in a.parts:
        pj = json.loads((p / 'PART.json').read_text())
        for n, h in pj['files'].items():
            if sha(p / n) != h: raise ValueError(f'{p}/{n} changed since export')
        parts.append((p, pj, [json.loads(l) for l in (p / 'items.jsonl').read_text().splitlines() if l.strip()]))
    side = {}
    if a.speed_table:
        for l in a.speed_table.read_text().splitlines():
            if l.strip(): x = json.loads(l); side[x['clip_id']] = x['speed']
    filled = 0
    for _, _, its in parts:
        for it in its:
            if it.get('speed') is None and it.get('clip_id', it['id']) in side:
                sp = side[it.get('clip_id', it['id'])]; it['speed'] = sp; filled += 1
                it['speed_id'] = 0 if sp is None or sp.get('speed_ok_0p70') is None else (1 if sp['speed_ok_0p70'] else 2)
    print(json.dumps(dict(speed_side_table=str(a.speed_table) if a.speed_table else None, filled=filled)), flush=True)
    N = sum(pj['frames'] for _, pj, _ in parts)
    a.out.mkdir(parents=True, exist_ok=False)
    Q = np.lib.format.open_memmap(a.out / 'train_q.npy', 'w+', np.float32, (N, 34))
    Fm = np.lib.format.open_memmap(a.out / 'train_feat.npy', 'w+', np.float16, (N, 89))
    ao = Path(str(a.out) + '-audio16'); ao.mkdir(exist_ok=False)
    Au = np.lib.format.open_memmap(ao / 'AUDIO16.npy', 'w+', np.int16, (N * 320,))
    items, blocks, base, seen = [], [], 0, set()
    spk_vecs, cond = [], {}
    for p, pj, its in parts:
        q = np.load(p / 'q34.npy', mmap_mode='r'); f = np.load(p / 'feat.npy', mmap_mode='r'); au = np.load(p / 'a16.npy', mmap_mode='r')
        if len(q) != pj['frames'] or len(au) != pj['frames'] * 320: raise ValueError(f'{p}: part length')
        for i in range(0, len(q), 1 << 20):
            j = min(i + (1 << 20), len(q)); Q[base + i:base + j] = q[i:j]; Fm[base + i:base + j] = f[i:j]
        for i in range(0, len(au), 1 << 26):
            j = min(i + (1 << 26), len(au)); Au[base * 320 + i:base * 320 + j] = au[i:j]
        for it in its:
            if it['id'] in seen: raise ValueError('duplicate item across parts: ' + it['id'])
            seen.add(it['id'])
            src = p / it['dir']; dst = a.out / it['dir']; dst.mkdir(parents=True); shutil.copyfile(src / 'alignment.json', dst / 'alignment.json')
            it = dict(it, offset=it['offset'] + base, part=str(p))
            s = it['spk']; valid = bool(s.get('valid')) and s.get('path') is not None
            if valid:
                if sha(s['path']) != s['sha256']: raise ValueError(f'{it["id"]}: speaker embedding SHA')
                v = np.load(s['path']).astype(np.float32).reshape(-1)
                if v.shape != (192,) or not np.isfinite(v).all() or abs(np.linalg.norm(v) - 1) > 1e-3: raise ValueError(f'{it["id"]}: embedding not unit 192-d')
            else:
                v = np.zeros(192, np.float32)
            if it['source_token'].startswith('semsynth') and valid: raise ValueError(f'{it["id"]}: TTS source must use the none embedding')
            cond[it['id']] = [len(spk_vecs), int(valid), TOKENS.index(it['source_token']), int(it['speed_id'])]
            spk_vecs.append(v)
            blocks.append(dict(key=it['id'], kind='human' if it['source_token'] in REAL_TOKENS else 'synthetic', offset=it['offset'], frames=it['frames'],
                               samples_minus_frames320=0, absmax=it.get('absmax')))
            items.append(it)
        base += pj['frames']
    Q.flush(); Fm.flush(); Au.flush(); del Q, Fm, Au
    (a.out / 'cond').mkdir(); np.save(a.out / 'cond' / 'SPK_EMB.npy', np.stack(spk_vecs).astype(np.float32))
    E.write_json(a.out / 'cond' / 'COND.json', dict(columns=['spk_row', 'spk_valid', 'source_id', 'speed_id'], tokens=TOKENS,
                                                   speed_ids={'0': 'null', '1': 'speed_ok_0p70', '2': 'speed_not_ok'},
                                                   spk_model_sha256=SPK_MODEL_SHA, items=cond), indent=None)
    # 11-09 copy for the single EX002 test scoring run: a test-only view (train / val empty, test_ids = the 282 manifest v3.2 test rows)
    train_ids, val_ids = [], []; test_ids = [it['id'] for it in items if it['split_hint'] == 'test']
    if len(test_ids) != len(items): raise ValueError('test-only view')
    E.write_json(a.out / 'train.json', items)
    E.write_json(a.out / 'SPLIT.json', dict(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids, rule='EX002 manifest v3.2 test (23_hailing); '
                                            'scored once under EVAL_PROTOCOL v1; never used for training or selection', external_isolation_verified=True))
    (a.out / 'labels.jsonl').write_text('')
    shutil.copyfile(E.BJ50_DATA / 'VOCAB.json', a.out / 'VOCAB.json')
    src_meta = json.loads((E.BJ50_DATA / 'META.json').read_text())
    meta = E.build_meta(X.load_spec(), items, [m for _, pj, _ in parts for m in pj['manifests']], F.code_digests(), src_meta)
    meta['ex002']['bj50_data_meta'] = dict(path=str(E.BJ50_DATA / 'META.json'), sha256=sha(E.BJ50_DATA / 'META.json'), keys_copied=['feat_dim', 'mel', 'prosody', 'n_actions'])
    meta['ex002']['version'] = 'unified-ex002-3f-mx28view-v36'
    E.write_json(a.out / 'META.json', meta)
    files = {str(p.relative_to(a.out)): sha(p) for p in sorted(a.out.rglob('*')) if p.is_file() and 'alignments' not in p.parts}
    align_digest = sha_tree(a.out / 'alignments')
    hours = collections.defaultdict(float); counts = collections.Counter()
    for it in items: hours[f"{it['source_token']}|{it['split_hint']}"] += it['frames'] / 180000; counts[f"{it['source_token']}|{it['split_hint']}"] += 1
    frozen = dict(schema='d1k-frozen-v1', training_eligible=False, variant='unified-ex002-3f-v36', files=files, alignments_tree_sha256=align_digest,
                  # train_s3 audits (data_source in human_data_sources) == (item has no CE label); every v3.6 source is label-free,
                  # so all of them are listed here.  It is NOT a statement that synthetic sources are human: see source_tokens.
                  human_items_without_labels=True, human_data_sources=sorted({it['data_source'] for it in items}),
                  label_free_sources_note='key name kept for the train_s3 label audit; semsynth is synthetic (source token semsynth_*)',
                  source_tokens=TOKENS, counts=dict(items=len(items), train=len(train_ids), val=len(val_ids), rows=N, by_token_split=dict(counts)),
                  hours={k: round(v, 6) for k, v in hours.items()}, parts=[dict(path=str(p), part_sha256=sha(p / 'PART.json')) for p, _, _ in parts], speed_side_table=dict(path=str(a.speed_table), sha256=sha(a.speed_table), filled=filled) if a.speed_table else None,
                  merger=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)), feature_code=F.code_digests(), spec_sha256=sha(X.SPEC_PATH),
                  label_policy='no CE labels: train with --allow-missing-labels')
    E.write_json(a.out / 'FROZEN.json', frozen); fsha = sha(a.out / 'FROZEN.json')
    (ao / 'BLOCKS.jsonl').write_text(''.join(json.dumps(b) + '\n' for b in blocks))
    rep = dict(schema='s3data-r2-audio16', status='ASSEMBLED', mix=str(a.out), mix_frozen_sha256=fsha, rate=16000, samples_per_frame=320,
               total_frames=N, blocks=len(blocks), by_source=dict(collections.Counter(b['kind'] for b in blocks)),
               samples_minus_frames320=dict(min=0, max=0), silent_blocks=sum(1 for b in blocks if b['absmax'] == 0),
               human_from='per-part exact clip audio, cut/zero-padded to frames*320')
    E.write_json(ao / 'REPORT.json', rep)
    E.write_json(ao / 'FROZEN.json', dict(rep, files={n: sha(ao / n) for n in ('AUDIO16.npy', 'BLOCKS.jsonl', 'REPORT.json')}))
    print(json.dumps(dict(data=str(a.out), frozen=fsha, audio16_frozen=sha(ao / 'FROZEN.json'), items=len(items), rows=N, hours=frozen['hours'],
                          seconds=round(time.time() - t0, 1))))


def sha_tree(root):
    import hashlib
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file(): h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


if __name__ == '__main__':
    main()
