"""EX002 test-split export for the single frozen-protocol scoring run (EVAL_PROTOCOL v1, TEST_USAGE entry 2026-11-09 09:27).
Runs USERS/.../export_part.py unchanged, except export_source.process, which refuses split 'test' (guard against test rows in
training exports): here rows with split 'test' and source beat2_en / beat2_zh go through the same path as dev rows (E.process_row
+ the audio-match check). Every other split is refused, so this wrapper cannot produce a training part. Output must be outside views/.
    python test_export_1109.py <export_part.py args> --splits test --out <dir under CKPT/.../ex002-test-1109>"""
import sys, runpy
C = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-ex002-data-v3_6-20261101/train'
sys.path.insert(0, C); sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025')
sys.path.insert(2, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/speaker-style-emb-r2-20261030/code')
assert sys.argv[sys.argv.index('--splits') + 1] == 'test' and '/ex002-test-1109/' in sys.argv[sys.argv.index('--out') + 1] + '/'
import export_source as S
import export_v31 as E


def process(r, cache, F, sf, info):
    if r['split'] != 'test' or r['source'] not in ('beat2_en', 'beat2_zh'): raise ValueError('only beat2_en / beat2_zh test rows')
    before = len(info); out = S.old(r, cache, F, sf, info)          # the unwrapped export_v31.process_row (export_source rebinds E.process_row to its guard)
    if any(x.get('kind') == 'manifest_audio_differs_from_recording_cut' for x in info[before:]):
        raise ValueError('v3.5 audio must match source; no silent replacement')
    return out


S.process = process; E.process_row = process
sys.argv[0] = C + '/export_part.py'
runpy.run_path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-test-1109/export_part_test.py', run_name='__main__')   # export_part.py with the exported-split line set to test
