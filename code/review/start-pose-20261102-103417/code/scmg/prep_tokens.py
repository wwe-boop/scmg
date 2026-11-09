"""Token ids of every utterance for the VoxCPM-style planner (text goes INTO the LM sequence, no timing needed).
Output: TOK_IDS.npy int32 [Ntok], TOK_INDEX.json {item_id: [first, n]}, PREP.json.
    python -m scmg.prep_tokens --out <dir> [--model <Qwen2.5-0.5B dir>]"""
import argparse, json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .data import VIEW

MODEL = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/Qwen2.5-0.5B'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=VIEW); ap.add_argument('--model', default=MODEL); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads((Path(a.view) / 'train.json').read_text()) if not it.get('text_null') and it.get('text')]
    tok = AutoTokenizer.from_pretrained(a.model)
    ids, index, row = [], {}, 0
    for b in range(0, len(items), 2048):
        batch = items[b:b + 2048]
        enc = tok([it['text'] for it in batch], add_special_tokens=False)['input_ids']
        for it, e in zip(batch, enc):
            e = e[:200]; ids.append(np.asarray(e, np.int32)); index[it['id']] = [row, len(e)]; row += len(e)
    np.save(out / 'TOK_IDS.npy', np.concatenate(ids)); (out / 'TOK_INDEX.json').write_text(json.dumps(index))
    (out / 'PREP.json').write_text(json.dumps(dict(model=a.model, items=len(items), tokens=row), indent=1))
    print('tokens', row, 'items', len(items))


if __name__ == '__main__':
    main()
