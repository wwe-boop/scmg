"""Frozen semantic encoder cache: multilingual-e5-base last-layer token features for every utterance with text.

Output dir: TEXT_FEAT.npy [Ntok, 768] fp16, TEXT_TIME.npy [Ntok] float32 (token onset in s from the utterance start,
NaN when no aligned character falls in the token), TEXT_INDEX.json {item_id: [first_row, n_tokens]}, PREP.json.
Token times come from the per-character forced alignment (alignments/<item>/alignment.json: index = char index in
`text`, onset_s relative to the utterance start; checked below against the utterance length).
    python -m scmg.prep_text --out <dir> [--view ...] [--model ...]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .data import VIEW

MODEL = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/multilingual-e5-base'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=VIEW); ap.add_argument('--model', default=MODEL); ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=256); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    v = Path(a.view); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads((v / 'train.json').read_text()) if not it.get('text_null') and it.get('text')]
    if a.limit: items = items[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    feats, times, index, row = [], [], {}, 0
    bad_align = 0; late = 0; t0 = time.time()
    for b0 in range(0, len(items), a.batch):
        batch = items[b0:b0 + a.batch]
        enc = tok([it['text'] for it in batch], return_tensors='pt', padding=True, truncation=True, max_length=160,
                  return_offsets_mapping=True)
        offs = enc.pop('offset_mapping').numpy()
        with torch.no_grad():
            h = model(**{k: x.cuda() for k, x in enc.items()}).last_hidden_state.float().cpu().numpy()
        am = enc['attention_mask'].numpy(); ids = enc['input_ids'].numpy()
        for i, it in enumerate(batch):
            onset = {}
            ap_ = v / it['dir'] / 'alignment.json'
            try:
                for e in json.loads(ap_.read_text()): onset[int(e['index'])] = float(e['onset_s'])
            except Exception:
                bad_align += 1
            if onset and max(onset.values()) > it['frames'] / 50. + 0.5: late += 1
            sel = [j for j in range(len(ids[i])) if am[i, j] and ids[i, j] not in tok.all_special_ids]
            tt = []
            for j in sel:
                s, e = offs[i, j]
                ts = [onset[c] for c in range(s, e) if c in onset]
                tt.append(min(ts) if ts else np.nan)
            feats.append(h[i, sel].astype(np.float16)); times.append(np.asarray(tt, np.float32))
            index[it['id']] = [row, len(sel)]; row += len(sel)
        if (b0 // a.batch) % 20 == 0:
            print(f'{b0 + len(batch)}/{len(items)} tokens={row} bad_align={bad_align} late={late} {time.time() - t0:.0f}s', flush=True)
    np.save(out / 'TEXT_FEAT.npy', np.concatenate(feats)); np.save(out / 'TEXT_TIME.npy', np.concatenate(times))
    (out / 'TEXT_INDEX.json').write_text(json.dumps(index))
    tt = np.concatenate(times)
    (out / 'PREP.json').write_text(json.dumps(dict(model=a.model, view=str(v), items=len(items), tokens=row,
                                                   timed_frac=float(np.isfinite(tt).mean()), bad_align=bad_align,
                                                   late_alignment=late), indent=1))
    print('done', row, 'tokens; timed', float(np.isfinite(tt).mean()))


if __name__ == '__main__':
    main()
