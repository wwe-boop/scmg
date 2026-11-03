"""Collect every eval-*/RESULTS.json under a run root into one Markdown table.
    python -m scmg.summarize <run root>"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows, cl = [], []
for f in sorted(root.glob('*/eval-*/RESULTS.json')):
    lane, ck = f.parent.parent.name, f.parent.name[5:]
    r = json.loads(f.read_text())
    for h, v in r.items():
        if h == 'closed_loop':
            for k, x in v.items(): cl.append((lane, ck, k, x))
        else:
            rows.append((lane, ck, h, v))
print('| lane | ckpt | H | clips | FKD | beat (gen/gt) | boundary (gen/gt) | speed ratio | diversity | tick p95 ms |')
print('|---|---|---|---|---|---|---|---|---|---|')
for lane, ck, h, v in rows:
    fkd = 'NA' if v.get('FKD') is None else f"{v['FKD']:.2f}"
    print(f"| {lane} | {ck} | {h} | {v['clips']} | {fkd} | {v['beat_align']:.2f}/{v['beat_align_gt']:.2f} | "
          f"{v['boundary']:.2f}/{v['boundary_gt']:.2f} | {v['speed_ratio']:.2f} | {v['diversity']:.3f} | {v['tick_ms_p95']:.0f} |")
if cl:
    print('\n| lane | ckpt | feedback | clips | fight rms | snap acc p99 | recover s | track rms after |')
    print('|---|---|---|---|---|---|---|---|')
    for lane, ck, k, x in cl:
        print(f"| {lane} | {ck} | {k} | {x['clips']} | {x['fight_rms']:.3f} | {x['snap_acc_p99']:.1f} | {x['recover_s']:.2f} | {x['track_rms_after']:.3f} |")
