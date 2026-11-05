"""Rebuild /mnt/bulk/zehan/renders/index.html with the Isaac Sim EX002 groups on top; the earlier sections (EX002 MuJoCo
version, G1 ECHO-G official MuJoCo renders) are kept verbatim from the previous index (backed up next to it once).
  python3 scripts/make_render_index.py [renders root]
"""
import html, json, shutil, sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/bulk/zehan/renders')
DEV, TTS, MJ = 'ex002-deploy-isaac-1105', 'ex002-deploy-tts-zh-1105', 'ex002-deploy-1105'
old = (R / 'index.html').read_text()
bak = R / 'index.html.bak-before-isaac'
if not bak.exists():
    shutil.copy2(R / 'index.html', bak)
src = bak.read_text()
keep = src[src.index('<h2 id="ex002">'):]                       # old EX002 MuJoCo + G1 sections, unchanged
keep = keep.replace('<h2 id="ex002">EX002 部署候选：流式口径（引擎）输出，中文开发集</h2>',
                    '<h2 id="ex002">（旧版，MuJoCo 渲染）EX002 部署候选：流式口径（引擎）输出，开发集</h2>', 1)
e = html.escape

dv = json.loads((R / DEV / 'INDEX.json').read_text())
ts = json.loads((R / TTS / 'INDEX.json').read_text())
fk = json.loads((R / DEV / 'FK_CHECK.json').read_text()) if (R / DEV / 'FK_CHECK.json').exists() else None
out = ['<!doctype html><meta charset="utf-8"><title>SCMG 渲染 11-05</title><style>body{font-family:sans-serif;margin:20px;max-width:1950px}'
       'video{width:100%;max-width:1900px;background:#000}.c{margin:18px 0 34px}small{color:#555}code{background:#f2f2f2;padding:0 3px}</style>',
       '<h1>SCMG 渲染对比（11-05）</h1><p><a href="#tts">EX002 自家 TTS 中文讲解句（Isaac Sim，6 句）</a> · '
       '<a href="#isaac">EX002 开发集（Isaac Sim，8 段）</a> · <a href="#ex002">EX002 开发集旧版（MuJoCo）</a> · <a href="#g1">G1 对 ECHO-G（7 段）</a></p>']
cfg = ts['infer']
out.append('<h2 id="tts">EX002 部署候选：自家 TTS 合成中文语音 → 部署流式路径（Isaac Sim 渲染）</h2>'
           '<p>左：修复前 sibag-s2 ｜ 右：修复后 sibag-coll-s1（每个模型 正面 + 3/4 侧）。两模型输入完全相同：同一段 TTS 音频（speech_server 共享 TTS，'
           'robot_service_v1，24 kHz → 16 kHz）、同一句文本与字时间锚点；推理走 <code>scmg/deploy_stream.py</code> 的部署 worker 原样逻辑'
           f'（CFG {cfg["cfg_w"]}、{cfg["steps"]} 步、H=2 即 160 ms 前瞻、text x2 15 tok/s lag 0.15 s、deploy_timing、因果 e5、说话人为空、source {cfg["source"]}、种子 0）。'
           f'50 fps，Isaac Sim 5.1 RTX。每句的 wav、两个模型的 34 关节轨迹 npy、句子文本在 <a href="{TTS}/">{TTS}/</a>。</p>')
for c in ts['clips']:
    out.append(f'<div class="c"><b>{e(c["id"])}</b> · {c["seconds"]:.1f} s · 「{e(c["text"])}」 · <small><a href="{TTS}/{c["wav"]}">wav</a> · '
               + ' · '.join(f'<a href="{TTS}/{p}">{e(l)} 轨迹</a>' for l, p in c['traj'].items())
               + f'</small><br><video controls preload="metadata" poster="{TTS}/{c["poster"]}" src="{TTS}/{c["video"]}"></video></div>')
fk_s = (f'34→37 维转换核对：MuJoCo 正运动学下原始 34 维与转换后轨迹最大差 {fk["raw_vs_npz_mm"]:.4f} mm（拇指 mimic 关节以下的体除外），'
        f'Isaac 与 MuJoCo 正运动学最大差 {fk["isaac_vs_mujoco_mm"]:.4f} mm。') if fk else ''
out.append('<h2 id="isaac">EX002 部署候选：流式口径（引擎）输出，开发集（Isaac Sim 渲染）</h2>'
           '<p>从左到右：真人原动作 ｜ 修复前 sibag-s2 ｜ 修复后 sibag-coll-s1（每个 正面 + 3/4 侧）。与下方 MuJoCo 旧版同一批轨迹'
           '（eval/final/engine/eval-s0 的 gt_ / gen_H2_，H=2 即 160 ms 前瞻）、同一段原始语音；50 fps，Isaac Sim 5.1 RTX。'
           f'升降柱固定 0.32 m，拇指 x_1_4 按 mimic 由 x_1_3 还原。{fk_s}</p>')
for c in dv['clips']:
    out.append(f'<div class="c"><b>clip {c["ci"]}</b> · {c["seconds"]} s · <small>{e(c["clip"])}</small><br><small>{e(c["text"])}</small><br>'
               f'<video controls preload="metadata" poster="{DEV}/{c["poster"]}" src="{DEV}/{c["video"]}"></video></div>')
out.append(keep)
(R / 'index.html').write_text('\n'.join(out))
print('wrote', R / 'index.html', len(dv['clips']), len(ts['clips']))
