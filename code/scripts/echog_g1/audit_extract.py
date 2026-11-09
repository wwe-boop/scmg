"""Audit of build_view.py extract results: counts, text/alignment consistency, audio format, speaker validity.
    audit_extract.py <work dir> [out json]"""
import collections, glob, json, sys
from pathlib import Path
import numpy as np

w = Path(sys.argv[1])
rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(w / 'items' / '*.npz.json')))]
bad_align = [r['stem'] for r in rs if any(r['text'][a['index']] != a['char'] for a in r['alignment'])]
late = [r['stem'] for r in rs if r['alignment'] and max(a['onset_s'] for a in r['alignment']) > r['n50'] / 50]
out = dict(stems=len(rs), lang=dict(collections.Counter(r['lang'] for r in rs)), text_null=sum(r['text_null'] for r in rs),
           feat_errors=collections.Counter(r['feat_error'] for r in rs if r['feat_error']).most_common(10),
           spk_invalid=sum(not r['spk_valid'] for r in rs), spk_errors=collections.Counter(r['spk_error'] for r in rs if r['spk_error']).most_common(5),
           audio_subtype=dict(collections.Counter(r['audio_subtype'] for r in rs)), audio_clipped=sum(r['audio_clipped'] for r in rs),
           audio_absmax_max=max(r['audio_absmax'] for r in rs),
           audio_shorter_than_motion=sum(r['audio_samples'] < r['n50'] * 320 for r in rs),
           audio_over_motion_s=dict(zip(('p0', 'p50', 'p100'), np.percentile([r['audio_seconds'] - r['n50'] / 50 for r in rs], [0, 50, 100]).round(3).tolist())),
           align_char_mismatch=len(bad_align), onset_past_end=len(late),
           words_total=sum(r['words'] for r in rs), chars_aligned=sum(len(r['alignment']) for r in rs),
           text_equals_transcript=sum(r['text'].replace(' ', '') == r['transcript'].replace(' ', '') for r in rs),
           frames50=sum(r['n50'] for r in rs), frames30=sum(r['n30'] for r in rs),
           n50_rule_ok=all(r['n50'] == int(round(r['n30'] * 5 / 3)) for r in rs))
print(json.dumps(out, indent=1, ensure_ascii=False))
if len(sys.argv) > 2: Path(sys.argv[2]).write_text(json.dumps(out, indent=1, ensure_ascii=False))
