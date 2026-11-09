import json, base64, numpy as np, wave, sys
SID, TI, TOT = 's3', 4.32, 6.50
O = '/home/xr-25b7/bulk/fig-1109/s3/'; R = '/mnt/bulk/zehan/renders/final-1109/'
import os; os.makedirs(O + 'tts/s3int', exist_ok=True); os.makedirs(O + 'audio', exist_ok=True)
rows = [json.loads(l) for l in open(R + f'tts/{SID}/stream.jsonl')]
cut, end = int(TI * 24000), int(TOT * 24000); out = []; last = 0
for r in rows:
    if r['type'] == 'audio':
        p = np.frombuffer(base64.b64decode(r['pcm']), np.float32).copy(); p[max(0, cut - r['start']):] = 0
        out.append(dict(r, pcm=base64.b64encode(p.tobytes()).decode())); last = r['end']
    elif r['type'] == 'anchor' and r['anchor']['output_sample_end'] <= cut: out.append(r)
s = last
while s < end:
    e = min(s + 1920, end); out.append(dict(type='audio', start=s, end=e, pcm=base64.b64encode(np.zeros(e - s, np.float32).tobytes()).decode())); s = e
out.append(dict(type='audio_done', samples=end))
open(O + 'tts/s3int/stream.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in out)); open(O + 'tts/s3int/text.txt', 'w').write(open(R + f'tts/{SID}/text.txt').read())
w = wave.open(R + f'work/g1/data/audio/chinese_tts_final1109_{SID}_utt_0000.wav'); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
y = np.zeros(int(TOT * sr), np.int16); n = min(len(x), int(TI * sr)); y[:n] = x[:n]
o = wave.open(O + 'audio/s3int.wav', 'wb'); o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes(y.tobytes()); o.close()
W = json.load(open(R + f'work/g1/data/annotations/words/chinese_tts_final1109_{SID}_utt_0000.json')); ws = W.get('words', W)
kept = [dict(text=x['text'], start=round(x['start'], 3), end=round(x['end'], 3)) for x in ws if x['end'] <= TI + 1e-6]
open(O + 'utterances.jsonl', 'w').write(json.dumps(dict(stem='s3int', audio_path='audio/s3int.wav', transcript=''.join(k['text'] for k in kept), words=kept), ensure_ascii=False) + '\n')
json.dump(dict(interrupts=[dict(ci=3, clip='final1109_s3', item='s3', frac=0.0, t_i=TI, f_i=int(round(TI * 50)))], rule=f'fig-1109: s3 interrupted at {TI} s'), open(O + 'INT.json', 'w'))
print('ok', ''.join(k['text'] for k in kept), int(round(TI * 50)))
