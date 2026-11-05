"""LLM semantic-intent gate (11-06 round 2 (c)): can an instruction LLM, reading the transcript, mark WHERE a speaker makes a
semantic gesture (deictic / iconic / metaphoric) well enough to drive tokens? Text only; zero-shot (no labels are shown to the
model, so nothing is fitted on dev). Look-ahead = the whole utterance text (upper bound of the anticipatory protocol: if this
fails the gate, a shorter TTS look-ahead fails too).
Gate (written before running): precision >= 0.30 = fraction of LLM-marked words whose onset lies within 0.5 s of a BEAT2
annotated deictic / iconic / metaphoric token time (keyword-anchored, g1_ctl_tokens.py), at the LLM's own marking density;
reported with the chance rate of the same words (any word within 0.5 s) and class precision.
Output PRED.json (CTL token format, mode pred in infer_test.py) + GATE.json.
    python scripts/semantic/g1_llm_intent.py --ctl <CTL dir> --view <hos view> --model <Qwen3-4B-Instruct dir> --out <dir> [--n 400]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_lexicon import words, load_align

CLS = {'deictic': 1, 'iconic': 2, 'metaphoric': 3}
PROMPT = """You are an expert in human co-speech gesture. Below is a transcript of a person speaking spontaneously; every word has an index.
Mark the words where the speaker most likely makes a SEMANTIC gesture with the hands, i.e. a gesture whose form carries meaning:
- deictic: pointing at a person, place, object or direction (e.g. "me", "you", "there", "this", "left")
- iconic: depicting a concrete shape, size, motion or action (e.g. "big", "round", "climb", "throw")
- metaphoric: presenting an abstract idea as if it had a shape or place (e.g. "idea", "the future", "on one hand", "balance")
Ordinary rhythmic beat gestures do NOT count. People make roughly one semantic gesture every 3 to 5 seconds of speech; this
utterance lasts {dur:.1f} s, so mark about {k} words (fewer if nothing is gesture-worthy). Mark the single word the gesture is
about (its lexical affiliate).
Transcript: {text}
Answer with JSON only: a list of [word_index, "deictic" | "iconic" | "metaphoric"]."""


def main():
    ap = argparse.ArgumentParser()
    for k in ('ctl', 'view', 'model', 'out'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--n', type=int, default=400); ap.add_argument('--density', type=float, default=4.0, help='seconds per gesture hint')
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model); m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text()); v = Path(a.view)
    items = {it['id']: it for it in json.loads((v / 'train.json').read_text())}
    stems = sorted(s for s, ts in raw.items() if ts and ts[0]['split'] == 'dev')[:a.n]
    out, st = {}, dict(marked=0, hit=0, cls_hit=0, chance_hit=0, words=0, parse_fail=0)
    for i, s in enumerate(stems):
        it = items[s]; ws = words(it['text'], load_align(v, s), it['lang']); dur = it['frames'] / 50.
        if not ws: continue
        k = max(1, int(round(dur / a.density)))
        text = ' '.join(f'{j}:{w[0]}' for j, w in enumerate(ws))
        msg = [{'role': 'user', 'content': PROMPT.format(dur=dur, k=k, text=text)}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors='pt').to('cuda')
        with torch.no_grad(): g = m.generate(ids, max_new_tokens=200, do_sample=False)
        ans = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        try: marks = json.loads(re.search(r'\[.*\]', ans, re.S).group(0))
        except Exception: st['parse_fail'] += 1; marks = []
        ann = [(t['t'], t['type']) for t in raw[s] if t['type'] in (1, 2, 3)]
        toks = []
        for mk in marks:
            try: j, c = int(mk[0]), CLS[str(mk[1]).lower()]
            except Exception: continue
            if not 0 <= j < len(ws): continue
            t0 = ws[j][2]; near = [cc for (ta, cc) in ann if abs(ta - t0) <= 0.5]
            st['marked'] += 1; st['hit'] += bool(near); st['cls_hit'] += c in near
            toks.append(dict(type=c, cls=[n for n, x in CLS.items() if x == c][0], t=round(t0, 3), a=round(t0, 3), b=round(t0 + 1.0, 3),
                             split='dev', word=ws[j][0], src='llm'))
        for w in ws: st['words'] += 1; st['chance_hit'] += any(abs(ta - w[2]) <= 0.5 for ta, _ in ann)
        out[s] = sorted(toks, key=lambda t: t['t'])
        if i % 50 == 0: print(i, len(stems), s, ans[:120].replace('\n', ' '), flush=True)
    gate = dict(precision=st['hit'] / max(1, st['marked']), class_precision=st['cls_hit'] / max(1, st['marked']),
                chance_rate=st['chance_hit'] / max(1, st['words']), threshold=0.30, **st, n_clips=len(stems),
                n_dev_annotations=sum(1 for s in stems for t in raw[s] if t['type'] in (1, 2, 3)), model=a.model, density_hint_s=a.density)
    gate['pass'] = gate['precision'] >= 0.30
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / 'PRED.json').write_text(json.dumps(out)); (o / 'GATE.json').write_text(json.dumps(gate, indent=1)); print('GATE', json.dumps(gate))


if __name__ == '__main__':
    main()
