"""Opt-in BAgger dataset and masks; no I/O or RNG changes on the disabled path."""
import json
from pathlib import Path
import numpy as np
import torch
from .common import CHUNK_FRAMES, J, FEAT, TEXT_DIM, SPK, N_SOURCE

SOURCE_ID = N_SOURCE - 1


def expand_source(model):
    """Append a zero row without consuming RNG, after legacy checkpoint load."""
    w = model.source.weight
    if len(w) == N_SOURCE: return
    if len(w) != SOURCE_ID: raise ValueError(f'unexpected source rows: {len(w)}')
    model.source = torch.nn.Embedding.from_pretrained(torch.cat([w.detach(), w.new_zeros(1, w.shape[1])]), freeze=False)
    model.cfg.n_source = N_SOURCE


def target_chunks(batch, K):
    if 'loss_from_chunk' not in batch: return None, None
    start = batch['loss_from_chunk'].long()
    if ((start < -1) | (start >= K)).any(): raise ValueError('invalid loss_from_chunk for window')
    return torch.arange(K, device=start.device)[None] >= start[:, None], start >= 0


class BaggerPool:
    """Each collection is a manifest plus independently resumable NPZ trajectories.

    The complete reversed clip has a 2/3 clean prefix. For 256-frame training,
    use its last 16 chunks: 8 prefix + 8 target chunks for a 24-chunk rollout.
    This preserves the entire correction tail, including the real seed chunk.
    Stored joints are radians, features are already normalised. The VAE is run
    again on the actual cropped training window by step_losses.
    """
    def __init__(self, roots, ds):
        if not roots: raise ValueError('bagger_share > 0 requires --bagger-dir')
        self.ds, self.rows = ds, []
        if ds.v.tts is not None: raise ValueError('BAgger does not support TTS hidden conditioning')
        if ds.T % CHUNK_FRAMES: raise ValueError('BAgger windows must be chunk aligned')
        for root in str(roots).split(':'):
            p = Path(root)
            meta = json.loads((p / 'MANIFEST.json').read_text())
            if meta['source'] != 'bagger_reversed' or meta['split'] != 'train':
                raise ValueError('only training BAgger collections may be aggregated')
            if meta.get('seed_representation') != 'VAE reconstruction of ground-truth seed latents; no raw-joint splice':
                raise ValueError('collection must use unified VAE decoding; regenerate legacy raw-spliced seed data')
            for row in meta['rows']:
                if row['frames'] < ds.T: raise ValueError('BAgger clip shorter than training window')
                start = row['frames'] - ds.T
                local = row['loss_from_chunk'] - start // CHUNK_FRAMES
                if not 0 < local < ds.T // CHUNK_FRAMES:
                    raise ValueError('window must contain both a clean prefix and supervised tail')
                self.rows.append((p / row['file'], row, start, local))
        if not self.rows: raise ValueError('empty BAgger collection')

    def sample(self, rng):
        d = self.ds; v = d.v; T, K, I = d.T, d.K, d.KI
        p, row, start, local = self.rows[int(rng.integers(len(self.rows)))]
        with np.load(p, allow_pickle=False) as z:
            q = (z['q_reversed'][start:start+T].astype(np.float32) - v.qm) / v.qs
            f = z['feat_reversed'][start:start+T].astype(np.float32)
            style = {key: z['original_'+key].copy() for key in ('spk','spk_valid','ref','ref_valid')} if row['cond']=='reverse' else {}
        s = dict(q=q, mask=np.ones(T, np.float32), feat=f,
                 text=np.zeros((K, TEXT_DIM), np.float32), ttime=np.full(K, np.nan, np.float32),
                 tmask=np.zeros(K, bool), tseg=np.full(K, np.nan, np.float32),
                 tids=np.zeros(I, np.int64), tidm=np.zeros(I, bool),
                 titime=np.full(I, np.nan, np.float32), tiseg=np.full(I, np.nan, np.float32),
                 tiend=np.full(I, np.nan, np.float32), spk=np.zeros(SPK, np.float32), spk_valid=np.float32(0),
                 source=np.int64(SOURCE_ID), ref=np.zeros((d.R, J), np.float32), ref_valid=np.float32(0),
                 clip=np.int64(-1), start=np.int64(start), loss_from_chunk=np.int64(local),
                 bagger_null=np.bool_(row['cond'] == 'null'))
        if style:
            if style['ref'].shape != s['ref'].shape: raise ValueError('reference length differs from collection')
            s.update(style)
        if v.text_c is not None: s['text_c'] = np.zeros_like(s['text'])
        if v.tok is not None and 'ilab' in v.tok: s['tilab'] = np.full(I, -1, np.int64)
        if v.intent is not None:
            s['intent'] = np.full(T, -1, np.int16); s['gest'] = np.full(T, -1, np.int16)
        if v.tts is not None:
            raise ValueError('BAgger TTS-hidden conditioning is not supported; use audio features')
        return s
