"""Record shared-TTS (speech_server 63261) streams for offline replay through scmg/deploy_stream.py.

Runs ON speech_server (CPU only; the TTS itself is the shared service, never started/stopped here):
  /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture/.venv-sim/bin/python tts_record.py <sentences.json> <out dir>
sentences.json = [{"id": "s1", "text": "..."}, ...]
Same session config as edge-live remote/edge_live_remote.py (custom_voice, robot_service_v1, chinese, full_text,
pcm_f32 24 kHz, firehose, text events on). Per sentence writes <out>/<id>/stream.jsonl with the messages in arrival
order ({"type":"anchor","anchor":{native_cursor_v3}} / {"type":"audio","start","end","pcm":<b64 f32le>} /
{"type":"audio_done","samples"}) -- exactly the rows the deploy worker receives -- plus tts24.wav (s16) and text.txt.
Sentences are synthesized one at a time (the 8 TTS slots are shared with all users).
"""
import base64, dataclasses, json, sys, time, uuid, wave
from pathlib import Path

import numpy as np

sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/Qwen3TTS-0818-native/client/src')
from qwen3tts import TTSClient  # noqa: E402
from qwen3tts.progress import TextProgressAnchor  # noqa: E402
from qwen3tts_protocol import AudioFormat, OutputPolicy, SessionStartRequest, SynthesisConfig, VADPolicy  # noqa: E402

TTS = '127.0.0.1:63261'
SPEAKER = 'robot_service_v1'


def record(client, text, out):
    out.mkdir(parents=True, exist_ok=True)
    policy = OutputPolicy(emit_text_events=True, config={'delivery': 'firehose'}, vad=VADPolicy(enabled=False, strategy='disabled'))
    cfg = SynthesisConfig(task_type='custom_voice', language='chinese', speaker=SPEAKER, input_mode='full_text',
                          audio=AudioFormat('pcm_f32', 24000, 1), output_policy=policy)
    rows, parts, count, done, t0 = [], [], 0, False, time.monotonic()
    s = client.open_stream(SessionStartRequest(session_id='scmg-rec-' + uuid.uuid4().hex, config=cfg, output_policy=policy))
    try:
        s.send_text(text, seq_no=0); s.end()
        for msg in s.iter_messages(post_send_idle_timeout=60):
            a = TextProgressAnchor.from_meta(dict(getattr(msg, 'meta', {}) or {}))
            if a:
                rows.append(dict(type='anchor', anchor=dataclasses.asdict(a)))
            if hasattr(msg, 'pcm_bytes'):
                n = len(msg.pcm_bytes) // 4
                if not n:
                    continue
                rows.append(dict(type='audio', start=count, end=count + n, pcm=base64.b64encode(msg.pcm_bytes).decode()))
                parts.append(np.frombuffer(msg.pcm_bytes, '<f4')); count += n
            elif str(msg.type) == 'done':
                done = True
            elif str(msg.type) == 'error':
                raise RuntimeError('TTS error event ' + str(getattr(msg, 'message', ''))[:200])
    finally:
        try: s.close('done')
        except Exception: pass
    if not done or not count:
        raise RuntimeError('TTS ended before completion')
    rows.append(dict(type='audio_done', samples=count))
    with open(out / 'stream.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype('<i2')
    with wave.open(str(out / 'tts24.wav'), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm.tobytes())
    (out / 'text.txt').write_text(text + '\n')
    return dict(samples=count, seconds=round(count / 24000, 2), anchors=sum(r['type'] == 'anchor' for r in rows),
                audio_msgs=sum(r['type'] == 'audio' for r in rows), wall_s=round(time.monotonic() - t0, 2))


def main():
    sents = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
    client = TTSClient.connect(TTS, transport='engine-grpc', timeout=180)
    for it in sents:
        for attempt in range(3):
            try:
                r = record(client, it['text'], out / it['id']); break
            except Exception as e:
                if attempt == 2: raise
                print('retry', it['id'], str(e)[:200], flush=True); time.sleep(2 * (attempt + 1))
        print(json.dumps(dict(id=it['id'], **r), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
