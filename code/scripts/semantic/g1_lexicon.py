"""Text-derived semantic labels for the G1 (ECHO-G / BEAT2) semantic-recognition metric (semantic-baseline-1105).

Label of a word = its lexical semantic category (fixed lexicon below, decided BEFORE looking at any motion), timed by the
G1 view's character-level forced alignment (echog-g1-v1/alignments/<stem>/alignment.json: index into item text, onset_s /
offset_s relative to the clip start).  English: words = whitespace tokens (lower-case, the view text has no punctuation
except apostrophes / digits); Chinese: longest-match of the zh lexicon on the character string (the text has no spaces).

Categories (one per word; first match wins, in this order):
  we      collective / inclusive deixis            self    speaker deixis (I, me, my ...)
  you     addressee deixis                         third   third-person / other people
  neg     negation                                 number  numbers, enumeration
  big     large size / quantity / totality         small   small size / quantity / restriction
  up      upward / high / increase                 down    downward / low / decrease
  this    proximal deixis (this / here / these)    there   distal deixis (there / those)
  quest   wh-questions                             mental  cognition verbs (think / know / feel / remember)
'none' = a word in no category (used only as the detection background).
"""
import json
from pathlib import Path

EN = {
    'we': "we us our ours ourselves we're we've we'll we'd everyone everybody".split(),
    'self': "i me my mine myself i'm i've i'll i'd".split(),
    'you': "you your yours yourself yourselves you're you've you'll you'd".split(),
    'third': "they them their theirs themselves they're they've he him his she her hers people someone somebody "
             "person".split(),
    'neg': "no not never nothing nobody none neither nor don't didn't doesn't can't cannot won't isn't wasn't aren't "
           "weren't couldn't wouldn't shouldn't haven't hasn't hadn't".split(),
    'number': "two three four five six seven eight nine ten twenty thirty hundred thousand million first second third "
              "both half once twice".split(),
    'big': "big huge large great giant massive enormous whole entire everything all every lot lots many much tons wide "
           "always totally".split(),
    'small': "little small tiny bit few less least only short slightly narrow".split(),
    'up': "up high higher highest rise rising raise above top over increase grow growing".split(),
    'down': "down low lower lowest below bottom under fall falling drop decrease".split(),
    'this': "this here these".split(),
    'there': "there those".split(),
    'quest': "what why how who where which".split(),
    'mental': "think thought know knew feel felt remember believe guess wonder imagine".split(),
}
ZH = {
    'we': '我们 咱们 大家'.split(),
    'self': '我 自己'.split(),
    'you': '你们 你 您'.split(),
    'third': '他们 她们 他 她 别人 人们'.split(),
    'neg': '不 没 别'.split(),
    'number': '两 三 四 五 六 七 八 九 十 百 千 万 第'.split(),
    'big': '很多 所有 全部 整个 巨大 大'.split(),
    'small': '一点 小 少 只'.split(),
    'up': '上 高 升'.split(),
    'down': '下 低 降'.split(),
    'this': '这'.split(),
    'there': '那'.split(),
    'quest': '为什么 什么 怎么 哪 谁'.split(),
    'mental': '觉得 认为 知道 记得 想'.split(),
}
CLASSES = ['none'] + list(EN)
EN_MAP = {w: c for c, ws in EN.items() for w in ws}
ZH_ITEMS = sorted(((w, c) for c, ws in ZH.items() for w in ws), key=lambda x: -len(x[0]))


def words(text, align, lang):
    """-> list of (word, category, onset_s, offset_s); words without any aligned character are skipped."""
    t = {a['index']: (a['onset_s'], a['offset_s']) for a in align}
    out = []
    if lang == 'zh':
        i = 0
        while i < len(text):
            hit = next(((w, c) for w, c in ZH_ITEMS if text.startswith(w, i)), None)
            w, c = hit if hit else (text[i], 'none')
            ts = [t[j] for j in range(i, i + len(w)) if j in t]
            if ts: out.append((w, c, min(x[0] for x in ts), max(x[1] for x in ts)))
            i += len(w)
        return out
    i = 0
    for w in text.split(' '):
        if w:
            ts = [t[j] for j in range(i, i + len(w)) if j in t]
            if ts: out.append((w, EN_MAP.get(w.lower(), 'none'), min(x[0] for x in ts), max(x[1] for x in ts)))
        i += len(w) + 1
    return out


def load_align(view_dir, stem):
    return json.loads((Path(view_dir) / 'alignments' / stem / 'alignment.json').read_text())
