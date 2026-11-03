"""Collect train-only backwards corrective trajectories using the real Streamer.

Supports deterministic shards (--shard/--shards), restart, and --dry-run.
The manifest is committed only after every requested row exists and validates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scmg.common import CHUNK, CHUNK_FRAMES
from scmg.data import View, WindowDataset, clip_lm_capacity, VIEW
from scmg.evaluate import load
from scmg.infer import Streamer

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'
COND_KEYS = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid',
             'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c')


def atomic_json(path, data):
    tmp = Path(str(path) + f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(data, indent=2)); tmp.replace(path)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vae', default=B + '/trained/vae-r3/final.pt')
    p.add_argument('--view', default=VIEW); p.add_argument('--stats', default=D + '/STATS_v36_1.json')
    p.add_argument('--text-cache', default=D + '/text-e5-v36_1')
    p.add_argument('--tok-cache', default=D + '/tok-qwen25-v36_1')
    p.add_argument('--holdout', default=B + '/semantic-recognizer-v1/manifest-n200/MANIFEST.json')
    p.add_argument('--cond', choices=['null', 'reverse'], required=True)
    p.add_argument('--count', type=int, default=2000); p.add_argument('--n-gen', type=int, default=23)
    p.add_argument('--seed-chunks', type=int, default=1); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cfg-w', type=float, default=None, help='checkpoint cfg_w, otherwise established lane value 2.0')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


@torch.no_grad()
def main(a):
    if a.n_gen < 2 or a.seed_chunks < 1 or a.count < 1 or not 0 <= a.shard < a.shards:
        raise ValueError('invalid collection dimensions/shard')
    T = (a.n_gen + a.seed_chunks) * CHUNK_FRAMES
    st = json.loads(Path(a.stats).read_text())
    v = View(a.view, 'train', a.text_cache, st, mmap=True, tok_cache=a.tok_cache, holdout=a.holdout)
    weights = v.weights()
    eligible = np.array([c['frames'] >= T for c in v.clips])
    weights = weights * eligible
    if not weights.sum(): raise ValueError('no eligible train clips')
    weights /= weights.sum()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    w = a.cfg_w if a.cfg_w is not None else float(ck.get('cfg_w', 2.0))
    ck_meta = dict(path=str(Path(a.ckpt).resolve()), bytes=Path(a.ckpt).stat().st_size,
                   mtime_ns=Path(a.ckpt).stat().st_mtime_ns, step=ck.get('step'), cfg_w=w)
    del ck
    spec = dict(source='bagger_reversed', split='train', cond=a.cond, count=a.count, n_gen=a.n_gen,
                seed_chunks=a.seed_chunks, seed=a.seed, frames=T, loss_from_chunk=(T//CHUNK_FRAMES)*2//3,
                checkpoint=ck_meta, vae=a.vae, view=a.view, holdout=a.holdout, stats=a.stats,
                inference=dict(steps=10, trim_start=True, cfg_null_fix=True, window_lm_text=True, trim_unused_id_padding=True),
                seed_representation='VAE reconstruction of ground-truth seed latents; no raw-joint splice',
                eligible_clips=int((weights > 0).sum()), sampling='seed+index, original category/frame weights, holdout excluded')
    print(json.dumps(spec), flush=True)
    if a.dry_run: return
    a.out.mkdir(parents=True, exist_ok=True)
    spec_path = a.out / 'COLLECTION.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('output belongs to a different collection; use a new directory')
    atomic_json(spec_path, spec)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('BAGGER_MEMORY_FRACTION', '.48')))
    model, vae = load(a.ckpt, a.vae, 'cuda')
    qmean, qstd = v.qm, v.qs
    start_time = time.monotonic()
    for index in range(a.shard, a.count, a.shards):
        f = a.out / f'{index:06d}.npz'; meta = a.out / f'{index:06d}.json'
        if f.exists() and meta.exists(): continue
        rng = np.random.default_rng([a.seed, index])
        ci = int(rng.choice(len(weights), p=weights)); c = v.clips[ci]
        start = int(rng.integers((c['frames'] - T)//CHUNK_FRAMES + 1)) * CHUNK_FRAMES
        ds = WindowDataset(v, T, max_tokens=256, max_ids=clip_lm_capacity(v, ci))
        sample = ds.sample(ci, start, rng)
        # Keep validity/glitch information and reject corrupt windows deterministically.
        for attempt in range(100):
            if sample['mask'].min() == 1: break
            ci = int(rng.choice(len(weights), p=weights)); c = v.clips[ci]
            start = int(rng.integers((c['frames']-T)//CHUNK_FRAMES+1))*CHUNK_FRAMES
            ds = WindowDataset(v, T, max_tokens=256, max_ids=clip_lm_capacity(v, ci))
            sample = ds.sample(ci, start, rng)
        else: raise RuntimeError('could not sample a valid training window')
        # clip_lm_capacity can cover a much longer recording than this rollout.
        # Drop ONLY unused trailing padding, never overlapping utterance tokens.
        n_ids = max(160, int(sample['tidm'].sum()))
        for key in ('tids','tidm','titime','tiseg','tiend'):
            sample[key] = sample[key][:n_ids].copy()
        torch.manual_seed(a.seed + index)
        q = torch.from_numpy(sample['q']).cuda()[None]
        seed_frames = a.seed_chunks * CHUNK_FRAMES
        seed_mu = vae.encode(q[:, :seed_frames])[0]
        seed_lat = ((seed_mu-model.lat_mean)/model.lat_std)[0]
        cond = {k: sample[k] for k in COND_KEYS if k in sample}
        stream = Streamer(model, vae, sample['feat'], cond, st, H=10**9, steps=10,
                          seed=a.seed+index, standby=st['q_mean'], ctx_t=model.ctx_t, cfg_w=w,
                          trim_start=True, cfg_null_fix=True, window_lm_text=model.cfg.planner == 'voxcpm')
        stream.k = a.seed_chunks; stream.lat = list(seed_lat.clone().unbind())
        for _ in range(a.n_gen): stream.tick()
        generated_lat = torch.stack(stream.lat)
        decoded = vae.decode((generated_lat*model.lat_std+model.lat_mean)[None].float())[0]
        # Decode the WHOLE latent trajectory, including the GT-encoded seed.
        # Splicing raw GT joints here creates an artificial seed/generated seam.
        forward = decoded.cpu().numpy()*qstd+qmean
        reversed_q = forward[::-1].copy()
        re_mu = vae.encode(torch.from_numpy((reversed_q-qmean)/qstd).cuda()[None])[0][0]
        re_lat = ((re_mu-model.lat_mean)/model.lat_std).cpu().numpy()
        rev_feat = np.zeros_like(sample['feat']) if a.cond == 'null' else sample['feat'][::-1].copy()
        payload = dict(q_forward=forward, q_reversed=reversed_q, lat_reencoded=re_lat,
                       q_seed_truth=sample['q'][:seed_frames]*qstd+qmean,
                       lat_forward=generated_lat.cpu().numpy(),
                       feat_forward=sample['feat'], feat_reversed=rev_feat,
                       loss_from_chunk=np.int64(spec['loss_from_chunk']))
        payload.update({'original_'+k: sample[k] for k in COND_KEYS if k in sample})
        if not all(np.isfinite(payload[k]).all() for k in ('q_forward','q_reversed','lat_reencoded','feat_reversed')):
            raise RuntimeError('nonfinite collection trajectory')
        tmp = Path(str(f)+'.tmp')
        with tmp.open('wb') as out: np.savez_compressed(out, **payload)
        tmp.replace(f)
        row = dict(index=index, file=f.name, frames=T, loss_from_chunk=spec['loss_from_chunk'], cond=a.cond,
                   clip=c['clip'], clip_index=ci, start=start, item_ids=[it['id'] for it in c['items']],
                   source='bagger_reversed', original_source=c['source'], seed=a.seed+index,
                   source_license_records=[{k:it[k] for k in ('id','license','training_use') if k in it} for it in c['items']],
                   training_use='experimental correction only; inherits original source restrictions',
                   sha256=hashlib.sha256(f.read_bytes()).hexdigest(), seconds=sum(stream.timing))
        atomic_json(meta, row)
        if index % 10 == 0: print(json.dumps(dict(index=index, elapsed=time.monotonic()-start_time, seconds=row['seconds'])), flush=True)
    rows = []
    for i in range(a.count):
        p = a.out/f'{i:06d}.json'
        if not p.exists(): break
        row = json.loads(p.read_text()); f = a.out/row['file']
        if hashlib.sha256(f.read_bytes()).hexdigest() != row['sha256']: raise ValueError(f'bad hash {f}')
        rows.append(row)
    if len(rows) == a.count:
        atomic_json(a.out/'MANIFEST.json', dict(**spec, rows=rows, elapsed_seconds=time.monotonic()-start_time))
        inventory = [a.out/'COLLECTION.json', a.out/'MANIFEST.json']
        inventory += [a.out/f'{i:06d}{suffix}' for i in range(a.count) for suffix in ('.npz','.json')]
        (a.out/'FILES.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in inventory))
        atomic_json(a.out/'FROZEN.json', dict(count=len(rows), source='bagger_reversed', split='train',
                    scope='isolated BAgger experiment; not canonical dataset admission',
                    physical_gate='not certified for execution or canonical data',
                    files_sha256=hashlib.sha256((a.out/'FILES.sha256').read_bytes()).hexdigest()))
        print('COLLECTION_DONE', len(rows), flush=True)


if __name__ == '__main__': main(parse())
