"""Frozen LLM token-feature cache: per-token hidden states of a decoder LLM (default Qwen3-0.6B, penultimate layer).

The cache is drop-in compatible with scmg/prep_text.py: TEXT_FEAT.npy [Ntok, D] fp16, TEXT_TIME.npy [Ntok] float32
(token onset in s from the utterance start, NaN when no aligned character falls in the token), TEXT_INDEX.json
{item_id: [first_row, n_tokens]}, PREP.json.  The tokenisation is the LLM's own byte-level BPE (Chinese and English
both work); token -> character spans are recovered from the *bytes* of every BPE piece, then mapped onto the
per-character forced alignment (alignments/<item>/alignment.json, index = char index in `text`).

Why no `--causal` flag: a decoder LLM is prefix-causal by construction.  The hidden state at token j in a full-sentence
forward depends only on tokens 0..j, so the same cache serves offline and token-streaming deployment.  We verify this
empirically on a random sample (`--verify-n`, default 20): every prefix is forwarded separately and compared with the
corresponding prefix of the full-sentence hidden states.  The resulting errors are written to PREP.json.

    python -m scmg.prep_text_llm --out <dir> [--view ...] [--model ...] [--layer -2]
    python -m scmg.prep_text_llm --out <dir>/manifest-n200 --manifest <manifest dir or MANIFEST.json>  # semantic subset
"""
import argparse, json, math, time, unicodedata
from bisect import bisect_left, bisect_right
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .data import VIEW

MODEL = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/tts/qwen3_text/offical/Qwen3-0.6B'


# --------------------------------------------------------------------------- tokenisation / byte -> character spans
def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode(), the byte<->unicode table used by Qwen's byte-level BPE."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    return {v: k for k, v in bytes_to_unicode().items()}


def _piece_bytes(token, dec):
    try:
        return bytes(dec[c] for c in token)
    except KeyError:                    # added/special token: not part of the text
        return None


def token_spans(text, ids, tok, dec):
    """Character spans [c0, c1) of every token of `ids` (which encode `text`).

    Byte-level BPE pieces may split one UTF-8 character across several tokens (common for rare CJK characters), so
    offsets must be computed on the concatenated bytes, not with the tokenizer's offset mapping (which can overlap).
    Returns (spans, n_fallback); a span is None for a special token.  If the byte reconstruction does not reproduce
    `text` exactly (text normalisation), fall back to the tokenizer offset mapping.
    """
    toks = tok.convert_ids_to_tokens(ids)
    pieces = [_piece_bytes(t, dec) for t in toks]
    if any(p is None for p in pieces):
        spans = [None] * len(pieces)
    else:
        raw = b''.join(pieces)
        try:
            raw_text = raw.decode('utf-8')
        except UnicodeDecodeError:
            raw_text = None
        if raw_text == text:
            bounds = [0]
            for ch in text:
                bounds.append(bounds[-1] + len(ch.encode('utf-8')))
            spans, cur = [], 0
            for p in pieces:
                b0, b1 = cur, cur + len(p)
                cur = b1
                if b1 <= b0:
                    spans.append(None); continue
                c0 = bisect_right(bounds, b0) - 1          # character containing the first byte
                c1 = bisect_left(bounds, b1)               # first character starting at/after the last byte
                spans.append((c0, max(c0 + 1, c1)))
            return spans, 0
    # fallback: tokenizer character offsets (also correct for ASCII / most complete CJK tokens)
    off = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    if len(off) != len(ids):
        raise ValueError(f'tokenizer offset mapping length {len(off)} != ids {len(ids)}')
    return [None if a == b else (a, b) for a, b in off], 1


# --------------------------------------------------------------------------- model
def load_model(path, dtype=torch.float16, device='cuda', mem_fraction=0.0):
    if mem_fraction:
        dev = torch.device(device)
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction),
                                                   device=dev.index if dev.index is not None else torch.cuda.current_device())
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    model = AutoModel.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True,
                                      attn_implementation='sdpa').to(device).eval()
    return tok, model


def resolve_layer(hidden_states, layer):
    n = len(hidden_states)                 # embeddings + one per transformer layer
    i = layer if layer >= 0 else n + layer
    if not 0 <= i < n:
        raise ValueError(f'--layer {layer} out of range for {n} hidden-state entries')
    return i


@torch.no_grad()
def hidden(model, token_lists, layer, device='cuda', pad_id=0):
    """Hidden states [sum n, D] of each token list (right padding, no special tokens)."""
    L = max(len(x) for x in token_lists)
    x = torch.full((len(token_lists), L), pad_id, dtype=torch.long, device=device)
    m = torch.zeros((len(token_lists), L), dtype=torch.long, device=device)
    for r, ids in enumerate(token_lists):
        x[r, :len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=device)
        m[r, :len(ids)] = 1
    hs = model(input_ids=x, attention_mask=m, use_cache=False, output_hidden_states=True).hidden_states
    h = hs[layer].float().cpu()
    return [h[r, :len(ids)].numpy() for r, ids in enumerate(token_lists)]


@torch.no_grad()
def verify_causal(model, tok, items, layer, n=20, seed=0, device='cuda', tol_rel=5e-3):
    """Forward every prefix separately; compare its last hidden state with the full-sentence state at that position.

    Qwen3 has a few 'massive activation' channels (|h| ~ 6e3), so a raw max absolute error of a fraction of an fp16 ULP
    is not informative.  The primary fp16 tolerance check is therefore the relative L2 error of the whole hidden vector
    (tol_rel = 5e-3 ~ 5 fp16 eps); we also record the raw abs error and the abs error normalised by the vector scale.
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(items), min(n, len(items)), replace=False) if items else []
    max_abs = max_abs_scaled = max_rel_norm = 0.0
    p99s = []; n_prefix = 0; n_within = 0; per = []
    for k in pick:
        text = items[int(k)]['text']
        ids = tok(text, add_special_tokens=False)['input_ids']
        if len(ids) < 2:
            continue
        full = hidden(model, [ids], layer, device)[0]                      # [n, D]
        pmax = pscaled = preln = 0.0; pwithin = 0; pn = 0
        for j in range(len(ids)):
            p = hidden(model, [ids[:j + 1]], layer, device)[0][-1]          # [D]
            f = full[j].astype(np.float64); d = np.abs(p.astype(np.float64) - f)
            scale = max(float(np.abs(f).max()), 1e-9)
            rel_norm = float(np.linalg.norm(d) / max(float(np.linalg.norm(f)), 1e-9))
            pmax = max(pmax, float(d.max())); pscaled = max(pscaled, float(d.max()) / scale)
            preln = max(preln, rel_norm); p99s.append(float(np.percentile(d, 99)))
            max_abs = max(max_abs, float(d.max())); max_abs_scaled = max(max_abs_scaled, float(d.max()) / scale)
            max_rel_norm = max(max_rel_norm, rel_norm)
            pwithin += int(rel_norm <= tol_rel); pn += 1
        n_prefix += pn; n_within += pwithin
        per.append(dict(id=items[int(k)]['id'], tokens=int(len(ids)), max_abs=pmax,
                        max_abs_over_scale=pscaled, max_rel_norm=preln, frac_within_rel_tol=float(pwithin / max(pn, 1))))
        print(f'  causal {len(per)}/{len(pick)} id={items[int(k)]["id"]} tokens={len(ids)} '
              f'max_abs={pmax:.3e} max_rel_norm={preln:.3e}', flush=True)
    return dict(n_items=len(per), n_prefixes=int(n_prefix), tol_rel_norm=float(tol_rel),
                max_abs=float(max_abs), max_abs_over_scale=float(max_abs_scaled), p99_abs=float(np.percentile(p99s, 99)),
                max_rel_norm=float(max_rel_norm), frac_within_rel_tol=float(n_within / max(n_prefix, 1)),
                seed=int(seed), per_item=per)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=VIEW); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=MODEL); ap.add_argument('--layer', type=int, default=-2,
                    help='hidden_states index: -1 = last layer, -2 = penultimate (default); positive = absolute index')
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-length', type=int, default=256); ap.add_argument('--dtype', default='float16',
                    choices=('float16', 'bfloat16'))
    ap.add_argument('--device', default='cuda'); ap.add_argument('--mem-fraction', type=float, default=0.0,
                    help='cap this process to this fraction of the GPU (0 = no cap; use on a shared server)')
    ap.add_argument('--verify-n', type=int, default=20, help='random utterances whose prefixes are re-forwarded (0 = skip)')
    ap.add_argument('--verify-seed', type=int, default=0)
    ap.add_argument('--manifest', default=None,
                    help='MANIFEST.json (or its directory): encode only its item_ids (semantic-eval subset)')
    a = ap.parse_args()

    v = Path(a.view); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads((v / 'train.json').read_text()) if not it.get('text_null') and it.get('text')]
    manifest = None
    if a.manifest:
        mp = Path(a.manifest)
        if mp.is_dir(): mp = mp / 'MANIFEST.json'
        manifest = json.loads(mp.read_text())
        wanted = {i for e in manifest.get('entries', []) for i in e.get('item_ids', [])}
        have = {it['id'] for it in items}
        missing = sorted(wanted - have)
        if missing:
            raise ValueError(f'{len(missing)} manifest item_ids are not in {v}/train.json, e.g. {missing[:5]}')
        items = [it for it in items if it['id'] in wanted]
        print(f'manifest subset: {len(items)} items, {len(wanted)} ids', flush=True)
    if a.limit:
        items = items[:a.limit]
    if not items:
        raise ValueError('no items to encode')

    dtype = getattr(torch, a.dtype)
    tok, model = load_model(a.model, dtype, a.device, a.mem_fraction)
    with torch.no_grad():
        probe_hs = model(input_ids=torch.zeros((1, 1), dtype=torch.long, device=a.device),
                                 output_hidden_states=True).hidden_states
    n_hs = len(probe_hs)
    layer_i = resolve_layer(probe_hs, a.layer)
    n_hidden = model.config.hidden_size
    print(f'model {a.model} hidden={n_hidden} layer={a.layer} (resolved {layer_i}) n_items={len(items)}', flush=True)

    dec = _byte_decoder()
    feats, times, index, row = [], [], {}, 0
    bad_align = 0; late = 0; truncated = 0; n_fallback = 0; n_special = 0
    t0 = time.time()
    for b0 in range(0, len(items), a.batch):
        batch = items[b0:b0 + a.batch]
        texts = [it['text'] for it in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=a.max_length, return_tensors='pt',
                  add_special_tokens=False)
        ids = enc['input_ids'].numpy(); am = enc['attention_mask'].numpy()
        truncated += sum(int(m.sum()) >= a.max_length for m in am)
        hs = hidden(model, [ids[i, :int(am[i].sum())].tolist() for i in range(len(batch))], layer_i, a.device)
        for i, it in enumerate(batch):
            n = int(am[i].sum()); seq = ids[i, :n].tolist()
            spans, fb = token_spans(it['text'], seq, tok, dec)
            n_fallback += fb
            sel = [j for j in range(n) if seq[j] not in tok.all_special_ids and spans[j] is not None]
            n_special += n - len(sel)
            onset = {}
            try:
                for e in json.loads((v / it['dir'] / 'alignment.json').read_text()):
                    onset[int(e['index'])] = float(e['onset_s'])
            except Exception:
                bad_align += 1
            if onset and max(onset.values()) > it['frames'] / 50. + 0.5:
                late += 1
            tt = []
            for j in sel:
                s, e = spans[j]
                ts = [onset[c] for c in range(s, min(e, len(it['text']))) if c in onset]
                tt.append(min(ts) if ts else np.nan)
            feats.append(hs[i][sel].astype(np.float16)); times.append(np.asarray(tt, np.float32))
            index[it['id']] = [row, len(sel)]; row += len(sel)
        if (b0 // a.batch) % 20 == 0:
            print(f'{b0 + len(batch)}/{len(items)} tokens={row} bad_align={bad_align} late={late} '
                  f'fallback={n_fallback} {time.time() - t0:.0f}s', flush=True)

    feat = np.concatenate(feats); tarr = np.concatenate(times)
    np.save(out / 'TEXT_FEAT.npy', feat); np.save(out / 'TEXT_TIME.npy', tarr)
    (out / 'TEXT_INDEX.json').write_text(json.dumps(index))
    ver = None
    if a.verify_n:
        print(f'causal verification: {a.verify_n} random utterances, every prefix re-forwarded', flush=True)
        ver = verify_causal(model, tok, items, layer_i, a.verify_n, a.verify_seed, a.device)
        print(f'  max_abs={ver["max_abs"]:.3e} max_abs_over_scale={ver["max_abs_over_scale"]:.3e} '
              f'max_rel_norm={ver["max_rel_norm"]:.3e} within_rel_tol={ver["frac_within_rel_tol"]:.4f} '
              f'prefixes={ver["n_prefixes"]}', flush=True)
    prep = dict(model=a.model, encoder='llm', layer=a.layer, layer_resolved=layer_i, layer_from_end=layer_i - n_hs,
                hidden_size=int(n_hidden), dtype=a.dtype, view=str(v), items=len(items), tokens=int(row),
                causal=True, native_causal=True, manifest=str(a.manifest) if a.manifest else None,
                tokenization=dict(add_special_tokens=False, padding_side='right', max_length=a.max_length,
                                  byte_offsets=True),
                timed_frac=float(np.isfinite(tarr).mean()), bad_align=bad_align, late_alignment=late,
                truncated=truncated, offset_fallback=n_fallback, dropped_special_tokens=n_special)
    if ver is not None:
        prep['causal_verification'] = ver
    (out / 'PREP.json').write_text(json.dumps(prep, indent=1, ensure_ascii=False))
    print(f'done {row} tokens, dim {feat.shape[1]}, timed {prep["timed_frac"]:.4f}, fallback {n_fallback}', flush=True)


if __name__ == '__main__':
    main()
