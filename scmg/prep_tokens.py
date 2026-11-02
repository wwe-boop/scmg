"""Token ids of every utterance for the VoxCPM-style planner (text goes INTO the LM sequence, no timing needed).
Output: TOK_IDS.npy int32 [Ntok], TOK_INDEX.json {item_id: [first, n]}, PREP.json.
    python -m scmg.prep_tokens --out <dir> [--model <Qwen2.5-0.5B dir>]"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .data import VIEW

MODEL = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/Qwen2.5-0.5B'


def voxcpm_token_ids(tok, texts):
    """OpenBMB/VoxCPM model/utils.py:mask_multichar_chinese_tokens (Apache-2.0).

    Match the official wrapper exactly, including the SentencePiece prefix rule.
    Non-Chinese and mixed tokens are preserved; no BOS/EOS is added.
    """
    multi = {t for t in tok.vocab if len(t) >= 2 and all('\u4e00' <= c <= '\u9fff' for c in t)}
    result = []
    for text in texts:
        pieces = []
        for token in tok.tokenize(text):
            clean = token.replace('▁', '')
            pieces.extend(list(clean) if clean in multi else [token])
        result.append(tok.convert_tokens_to_ids(pieces))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=VIEW); ap.add_argument('--model', default=MODEL); ap.add_argument('--out', required=True)
    ap.add_argument('--voxcpm-chars', action='store_true', help='official VoxCPM Chinese multichar token splitting')
    ap.add_argument('--trust-remote-code', action='store_true', help='load pinned local tokenizer custom code')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads((Path(a.view) / 'train.json').read_text()) if not it.get('text_null') and it.get('text')]
    tok = AutoTokenizer.from_pretrained(a.model, **({'trust_remote_code': True} if a.trust_remote_code else {}))
    ids, index, row = [], {}, 0
    for b in range(0, len(items), 2048):
        batch = items[b:b + 2048]
        texts = [it['text'] for it in batch]
        enc = voxcpm_token_ids(tok, texts) if a.voxcpm_chars else tok(texts, add_special_tokens=False)['input_ids']
        for it, e in zip(batch, enc):
            e = e[:200]; ids.append(np.asarray(e, np.int32)); index[it['id']] = [row, len(e)]; row += len(e)
    np.save(out / 'TOK_IDS.npy', np.concatenate(ids)); (out / 'TOK_INDEX.json').write_text(json.dumps(index))
    (out / 'PREP.json').write_text(json.dumps(dict(model=a.model, items=len(items), tokens=row, voxcpm_chars=a.voxcpm_chars,
        trust_remote_code=a.trust_remote_code, max_ids=200,
        input_sha256=hashlib.sha256((Path(a.view) / 'train.json').read_bytes()).hexdigest()), indent=1))
    print('tokens', row, 'items', len(items))


if __name__ == '__main__':
    main()
