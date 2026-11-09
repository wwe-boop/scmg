"""Checkpoint-driven token-cache selection and fail-closed metadata validation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .common import P

DATA = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
MODELS = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models'
CACHES = {'qwen25': DATA + '/tok-qwen25-v36_1', 'qwen3': MODELS + '/tok-qwen3-0.6b-v36_1',
          'minicpm4': MODELS + '/tok-minicpm4-v36_1', 'voxcpm15': MODELS + '/tok-voxcpm15-v36_1'}


def lm_family(name):
    n = str(name).rstrip('/').split('/')[-1].lower()
    if not n or n in ('qwen2.5-0.5b', 'qwen2.5-0.5b-instruct'): return 'qwen25'
    if n in ('qwen3-0.6b', 'qwen3-0.6b-base'): return 'qwen3'
    if n == 'minicpm4-0.5b': return 'minicpm4'
    if n == 'voxcpm1.5': return 'voxcpm15'
    raise ValueError(f'unknown tokenizer identity: {name!r}; use a cache with matching PREP.model metadata')


def checkpoint_config(checkpoint, entry, overrides=None):
    import torch
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = SimpleNamespace(**{**ck['base_cfg'], **ck.get('extra_cfg', {}), **(overrides or {})})
    reject_ttsh(cfg, entry)
    return cfg


def reject_ttsh(cfg, entry):
    if getattr(cfg, 'tts_h', False):
        raise ValueError(f'{entry} does not support TTSH checkpoints (tts_h=True); use scmg.evaluate --tts-hidden <directory>')


def cache_metadata(root):
    root = Path(P(str(root)))
    prep = json.loads((root / 'PREP.json').read_text())
    raw_index = (root / 'TOK_INDEX.json').read_bytes()
    index = json.loads(raw_index)
    ids = np.load(root / 'TOK_IDS.npy', mmap_mode='r')
    if not isinstance(index, dict) or not index or ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) == 0:
        raise ValueError(f'invalid token cache arrays / TOK_INDEX: {root}')
    for key, pair in index.items():
        if not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair):
            raise ValueError(f'invalid TOK_INDEX row {key}')
    cursor = 0
    for key, pair in sorted(index.items(), key=lambda kv: kv[1][0]):
        start, n = pair
        if start != cursor or n < 0 or start + n > len(ids): raise ValueError(f'invalid TOK_INDEX range for {key}')
        cursor += n
    if cursor != len(ids) or prep.get('items') != len(index) or prep.get('tokens') != len(ids):
        raise ValueError(f'TOK_INDEX / PREP counts do not match TOK_IDS: {root}')
    if np.any(ids < 0): raise ValueError('negative token ids')
    meta = dict(model=prep['model'], voxcpm_chars=prep.get('voxcpm_chars', False),
                index_sha256=hashlib.sha256(raw_index).hexdigest(),
                ids_sha256=hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest())
    for key in ('index_sha256', 'ids_sha256'):
        if key in prep and prep[key] != meta[key]: raise ValueError(f'token cache {key} mismatch')
    return meta


def resolve_token_cache(cfg, override=None):
    if getattr(cfg, 'planner', 'chunk') != 'voxcpm':
        if override: cache_metadata(override)
        return override
    name = getattr(cfg, 'vox_lm', '')
    try: family = lm_family(name)
    except ValueError:
        if not override: raise
        family = None
    root = override or CACHES[family]
    meta = cache_metadata(root)
    matches = lm_family(meta['model']) == family if family else Path(meta['model']).name == Path(name).name
    if not matches or bool(meta['voxcpm_chars']) != (family == 'voxcpm15'):
        raise ValueError(f'token cache mismatch: checkpoint vox_lm={name!r}, PREP={meta}; cache={root}')
    return str(Path(P(root)))


def validate_manifest_cache(manifest, cache):
    actual = cache_metadata(cache)
    expected = manifest.get('token_cache_metadata')
    if expected is None:
        # semantic-eval-manifest-v1 was produced by a hard-coded Qwen2.5 cache.
        if manifest.get('version') != 'semantic-eval-manifest-v1' or lm_family(actual['model']) != 'qwen25':
            raise ValueError('legacy manifest token cache mismatch: only Qwen2.5 provenance is known; rebuild manifest')
    else:
        for key in ('model', 'voxcpm_chars', 'index_sha256', 'ids_sha256'):
            equal = Path(expected[key]).name == Path(actual[key]).name if key == 'model' else expected[key] == actual[key]
            if not equal: raise ValueError(f'manifest token cache mismatch: {key}')


def validate_manifest_tokens(saved, expected, ci):
    # Legacy manifests have no tokenizer metadata: validate their actual token sequence against TOK_INDEX rows.
    for key in ('tids', 'tidm'):
        if key not in saved or not np.array_equal(saved[key], expected[key]):
            raise ValueError(f'manifest token cache mismatch at ci={ci}, {key}; rebuild manifest with the checkpoint tokenizer')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('checkpoint'); ap.add_argument('--entry', default='ft_lane.sh')
    ap.add_argument('--cfg', default='{}')
    a = ap.parse_args(); checkpoint_config(a.checkpoint, a.entry, json.loads(a.cfg))
