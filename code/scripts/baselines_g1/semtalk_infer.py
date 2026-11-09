"""SemTalk (ICCV 2025, official code github.com/Xiangyue-Zhang/SemTalk @7ba3520, Speaker-2 paper-protocol weights
HF X-Zhang/SemTalk weights_semtalk_no_smplx.zip sha256 3686d826...) on ECHO-G val_common clips -> BEAT2-layout SMPL-X npz
(Y-up, 30 fps), ready for the same GMR -> G1 converter as EMAGE / GestureLSM (gmr_convert.py).

Conditions are only the clip's own audio and word-timed transcript (ECHO-G annotations/words, the same source the
GestureLSM baseline used); the ground-truth motion of the clip is never read (only its frame count, to trim the output).
The decoding loop is the official test loop (semtalk_sparse_trainer.CustomTrainer._g_test: Base + Sparse models, gate
selects semantic codes, RVQ decoding, VAEConvZero global motion) and the features are built with the official
beat2_add_hubert / audio_to_frame_tokens recipes (HuBERT-large on the librosa-default-rate waveform, rhythm/pause 'beat'
features, frame-level word ids from the BEAT2 vocab with "" for silences, per-64-frame-window sentence text and
emotion2vec_plus_large label, both CLIP ViT-B/32 encoded).  Speaker id = scott (idmapping(2) = 1), the only id the
Speaker-2 model was trained with.

Seed pose: SemTalk conditions every window on 4 known frames (pre_frames=4).  Its official test() takes them from the
test clip's GT; its official audio-only inference() takes them from one fixed BEAT2 test clip of speaker 2.  We follow
inference(): the first 4 frames of the repository's demo/2_scott_0_1_1.npz (speaker 2, never a test speaker), the same
for every clip; later windows use the model's own last 4 frames (official).  No GT of the evaluated clip is used.

Length: features are built on the waveform zero-padded so that 4 + 60*k >= n frames (k windows), the generated motion is
trimmed to the GT frame count n (EMAGE / GestureLSM convention).  Decoding is argmax (deterministic), so there is a
single sample per clip.

  semtalk_infer.py --stems <txt> --out <dir> [--shard i/n] [--limit N]
"""
from __future__ import annotations

import argparse, json, os, sys, time, types, zlib
from itertools import groupby
from pathlib import Path

import numpy as np

ST = Path(os.environ.get('SEMTALK_ROOT', '/home/xr-25b7/bulk/semtalk-1105'))
REPO = ST / 'repo'
EG = Path(os.environ.get('ECHOG_DS', '/mnt/bulk/zehan/echog-dataset/ds'))

sys.path.insert(0, str(REPO))
os.environ.setdefault('SEMTALK_PRETRAINED_VQ_DIR', str(ST / 'weights/weights/pretrained_vq'))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import librosa  # noqa: E402
from numpy.lib import stride_tricks  # noqa: E402

import pandas  # noqa: E402,F401  (import before the pyarrow stub below so pandas sees no pyarrow)
for _m in ('whisperx', 'fasttext', 'pyarrow'):  # module-level imports of unused code paths (ASR / vocab building)
    sys.modules.setdefault(_m, types.ModuleType(_m))
from dataloaders.build_vocab import Vocab  # noqa: E402  BEAT2 vocab.pkl was pickled from a __main__ script
import __main__  # noqa: E402
__main__.Vocab = Vocab
del sys.modules['pyarrow']  # the stub only served build_vocab's unused import; sklearn & co. must not see it


# ---------------------------------------------------------------- official feature recipes (verbatim logic)
def extract_rhythm_pause_features(audio_each_file, sr, target_sr=16000, frame_length=1024, hop_length=512, target_fps=30):
    """beat2_add_hubert.extract_rhythm_pause_features, taking the waveform loaded with sr=None instead of a path."""
    audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=target_sr)
    shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
    strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
    rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
    amplitude_envelope = amplitude_envelope[:len(audio_each_file) // hop_length]
    energy = np.array([np.sum(np.abs(audio_each_file[i:i + frame_length] ** 2))
                       for i in range(0, len(audio_each_file), hop_length)])
    energy = energy[:len(amplitude_envelope)]
    audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=target_sr, hop_length=hop_length, units='frames')
    onset_array = np.zeros(len(amplitude_envelope), dtype=float)
    valid_onsets = audio_onset_f[audio_onset_f < len(onset_array)]
    onset_array[valid_onsets] = 1.0
    features = np.stack([amplitude_envelope, energy, onset_array], axis=1)
    duration = len(audio_each_file) / target_sr
    num_frames = int(duration * target_fps)
    out = np.zeros((num_frames, features.shape[1]))
    for i in range(features.shape[1]):
        out[:, i] = np.interp(np.linspace(0, len(features) - 1, num_frames), np.arange(len(features)), features[:, i])
    return out


@torch.no_grad()
def hubert_long(hubert_model, processor, speech, device):
    """audio_to_frame_tokens.get_hubert_from_16k_speech_long."""
    input_values_all = processor(speech, return_tensors='pt', sampling_rate=16000).input_values.squeeze(0).to(device)
    kernel, stride = 400, 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res = []
    for i in range(num_iter):
        s0 = 0 if i == 0 else clip_length * i
        e0 = (clip_length - stride + kernel) if i == 0 else s0 + (clip_length - stride + kernel)
        res.append(hubert_model(input_values_all[:, s0:e0]).last_hidden_state[0])
    iv = input_values_all[:, clip_length * num_iter:] if num_iter > 0 else input_values_all
    if iv.shape[1] >= kernel:
        res.append(hubert_model(iv).last_hidden_state[0])
    ret = torch.cat(res, 0).cpu()
    assert abs(ret.shape[0] - expected_T) <= 1
    return F.pad(ret, (0, 0, 0, expected_T - ret.shape[0])) if ret.shape[0] < expected_T else ret[:expected_T]


def process_annotated_sentence(words):
    filtered = [w for w in words if w.strip()]
    return ' '.join(k for k, _ in groupby(filtered))


def word_tracks(words, n_frames, duration, lang):
    """beat2_add_hubert word loop on a BEAT2-style TextGrid built from ECHO-G word times ("" fills the gaps)."""
    iv, cur = [], 0.0
    for w in sorted(words, key=lambda x: x['start']):
        s, e = max(0.0, float(w['start'])), min(duration, float(w['end']))
        if s > cur + 1e-9:
            iv.append(('', cur, s))
        if e > s:
            iv.append((str(w['text']).lower(), s, e)); cur = max(cur, e)
    if cur < duration:
        iv.append(('', cur, duration))
    word_ids, word_sent = [], []
    for i in range(n_frames):
        t = i / 30.0
        for (wn, ws, we) in iv:
            if ws <= t <= we:
                word_ids.append(lang.PAD_token if wn == ' ' else lang.get_word_index(wn)); word_sent.append(wn); break
        else:
            word_ids.append(lang.UNK_token)
    return np.array(word_ids, np.int64), word_sent


# ---------------------------------------------------------------- model
def build(device):
    from utils import config, other_tools
    from utils.semtalk_components import build_semtalk_joint_context, load_pretrained_vq_suite
    W = ST / 'weights/weights'
    sys.argv = ['semtalk', '--config', str(REPO / 'configs/semtalk_sparse.yaml'),
                '--data_path', str(ST / 'beat2stub') + '/', '--data_path_1', str(W) + '/',
                '--load_ckpt', str(W / 'best_semtalk_sparse.bin'), '--base_ckpt', str(W / 'best_semtalk_base.bin')]
    args = config.parse_args()
    assert list(args.training_speakers) == [2], args.training_speakers
    mod = __import__('models.semtalk', fromlist=['x'])
    base = mod.semtalk_base(args).to(device)
    other_tools.load_checkpoints(base, args.base_ckpt, 'semtalk_base')
    sparse = getattr(mod, args.g_name)(args).to(device)
    other_tools.load_checkpoints(sparse, args.load_ckpt, 'semtalk_sparse')
    vq = load_pretrained_vq_suite(args, device, args.e_name, include_global_motion=True)
    ctx = build_semtalk_joint_context(args.ori_joints)
    return args, base.eval(), sparse.eval(), vq, ctx


def seed_latent(args, ctx, device):
    """First 4 frames of demo/2_scott_0_1_1.npz -> latent_all layout [6D pose 330 | trans 3 | foot contact 4]."""
    import smplx
    from utils import rotation_conversions as rc
    d = np.load(REPO / 'demo/2_scott_0_1_1.npz', allow_pickle=True)
    k = args.pre_frames + 1                                   # one extra frame for the contact velocity
    poses = torch.from_numpy(d['poses'][:k].astype(np.float32))
    trans = torch.from_numpy(d['trans'][:k].astype(np.float32))
    exps = torch.from_numpy(d['expressions'][:k].astype(np.float32))
    betas = torch.from_numpy(np.tile(d['betas'].reshape(1, 300), (k, 1)).astype(np.float32))
    bm = smplx.create(str(ST / 'weights/weights/smplx_models'), model_type='smplx', gender='NEUTRAL_2020',
                      use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext='npz', use_pca=False)
    with torch.no_grad():
        j = bm(betas=betas, transl=trans, expression=exps, jaw_pose=poses[:, 66:69], global_orient=poses[:, :3],
               body_pose=poses[:, 3:66], left_hand_pose=poses[:, 75:120], right_hand_pose=poses[:, 120:165],
               leye_pose=poses[:, 69:72], reye_pose=poses[:, 72:75], return_joints=True)['joints'][:, (7, 8, 10, 11)]
    feetv = torch.zeros(4, k); feetv[:, :-1] = (j[1:] - j[:-1]).norm(dim=-1).T
    contact = (feetv < 0.01).float().T                       # [k, 4]
    p6 = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(poses.reshape(k, 55, 3))).reshape(k, 330)
    lat = torch.cat([p6, trans, contact], -1)[:args.pre_frames]
    return lat[None].to(device), trans[0].to(device)


@torch.no_grad()
def generate(args, base, sparse, vq, ctx, feats, seed_lat, trans0, device, mark=None):
    """semtalk_sparse_trainer._g_test decoding with the seed above (GT never enters)."""
    from utils import rotation_conversions as rc, other_tools
    j = ctx['joints']; mu, ml, mh = ctx['masks']['upper'], ctx['masks']['lower'], ctx['masks']['hands']
    pre, L = args.pre_frames, args.pose_length; rl = L - pre
    in_word, beat, hubert = feats['word'], feats['beat'], feats['hubert']
    feat_clip, emo_clip = feats['feat_clip'], feats['emo_clip']
    k = feats['k']; bs = 1
    tar_id = torch.ones(bs, L, 1, dtype=torch.long, device=device)   # idmapping(2) = 1 (scott)
    lsm = torch.nn.LogSoftmax(dim=2)
    V = args.vae_codebook_size

    def inv(x, mask, n):
        out = torch.zeros((n, 165), device=device); out[:, torch.where(torch.from_numpy(mask).to(device) == 1)[0]] = x
        return out

    def idx(logits):
        return torch.max(lsm(logits).reshape(-1, V, 6).reshape(-1, 16, V, 6), dim=2)[1]

    all_f, all_u, all_l, all_h = [], [], [], []
    latent_last = None
    for i in range(k):
        sl = slice(i * rl, (i + 1) * rl + pre)
        mask_val = torch.ones(bs, L, args.pose_dims + 3 + 4, device=device); mask_val[:, :pre] = 0.0
        lat = torch.zeros(bs, L, args.pose_dims + 3 + 4, device=device)
        lat[:, :pre] = seed_lat if i == 0 else latent_last[:, -pre:]
        bo = base(in_audio=beat[:, sl], in_word=in_word[:, sl], mask=mask_val, in_motion=lat, in_id=tar_id,
                  hubert=hubert[:, sl], use_attentions=True)
        so = sparse(in_word=in_word[:, sl], feat_clip_text=feat_clip[:, i], emotion=emo_clip[:, i], mask=mask_val,
                    in_motion=lat, in_id=tar_id, use_attentions=True, hubert=hubert[:, sl])
        gate = torch.argmax(torch.softmax(so['gate'], dim=-1), dim=-1).unsqueeze(-1)
        iu, il, ih, iface = idx(bo['cls_upper']), idx(bo['cls_lower']), idx(bo['cls_hands']), idx(bo['cls_face'])
        iu = torch.where(gate != 0, idx(so['cls_upper']), iu)
        ih = torch.where(gate != 0, idx(so['cls_hands']), ih)
        il = torch.where(gate != 0, idx(so['cls_lower']), il)
        s0 = 0 if i == 0 else 1
        all_f.append(iface[:, s0:]); all_u.append(iu[:, s0:]); all_l.append(il[:, s0:]); all_h.append(ih[:, s0:])
        ru, rlo, rh = vq['upper'].decode(iu), vq['lower'].decode(il), vq['hands'].decode(ih)
        n = rlo.shape[1]
        pu = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(ru.reshape(bs, n, 13, 6))).reshape(bs * n, 39)
        pl = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rlo[:, :, :54].reshape(bs, n, 9, 6))).reshape(bs * n, 27)
        ph = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rh.reshape(bs, n, 30, 6))).reshape(bs * n, 90)
        rp = inv(pu, mu, bs * n) + inv(pl, ml, bs * n) + inv(ph, mh, bs * n)
        rp = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(rp.reshape(bs, n, j, 3))).reshape(bs, n, j * 6)
        tv = rlo[:, :, 54:57]
        rt = torch.cat([other_tools.velocity2position(tv[:, :, 0:1], 1 / args.pose_fps, trans0[None, 0:1]), tv[:, :, 1:2],
                        other_tools.velocity2position(tv[:, :, 2:3], 1 / args.pose_fps, trans0[None, 2:3])], -1)
        latent_last = torch.cat([rp, rt, rlo[:, :, 57:61]], -1)
    if mark: mark('generate')                                 # window loop (incl. its per-window feedback decode)
    iface, iu, il, ih = (torch.cat(x, 1) for x in (all_f, all_u, all_l, all_h))
    ru, rlo, rh, rf = vq['upper'].decode(iu), vq['lower'].decode(il), vq['hands'].decode(ih), vq['face'].decode(iface)
    n = rf.shape[1]
    pu = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(ru.reshape(bs, n, 13, 6))).reshape(bs * n, 39)
    lm = rc.rotation_6d_to_matrix(rlo[:, :, :54].reshape(bs, n, 9, 6))
    l2g = rc.matrix_to_rotation_6d(lm.clone()).reshape(bs, n, 54)
    pl = rc.matrix_to_axis_angle(lm).reshape(bs * n, 27)
    ph = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rh.reshape(bs, n, 30, 6))).reshape(bs * n, 90)
    pj = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rf[:, :, :6].reshape(bs * n, 6))).reshape(bs * n, 3)
    rp = inv(pu, mu, bs * n) + inv(pl, ml, bs * n) + inv(ph, mh, bs * n)
    rp[:, 66:69] = pj
    tg = rlo; tg[:, :, 54:57] = 0.0; tg[:, :, :54] = l2g
    tv = vq['global_motion'](tg)['rec_pose'][:, :, 54:57]
    rt = torch.cat([other_tools.velocity2position(tv[:, :, 0:1], 1 / args.pose_fps, trans0[None, 0:1]), tv[:, :, 1:2],
                    other_tools.velocity2position(tv[:, :, 2:3], 1 / args.pose_fps, trans0[None, 2:3])], -1)
    return rp.cpu().numpy().reshape(n, 165), rt.cpu().numpy().reshape(n, 3), rf[:, :, 6:].cpu().numpy().reshape(n, 100)


# ---------------------------------------------------------------- features for one clip
class Feat:
    def __init__(self, device):
        from transformers import Wav2Vec2Processor, HubertModel
        import clip, pickle
        from funasr import AutoModel
        self.device = device
        self.proc = Wav2Vec2Processor.from_pretrained(str(ST / 'hf/hubert-large-ls960-ft'))
        self.hubert = HubertModel.from_pretrained(str(ST / 'hf/hubert-large-ls960-ft')).to(device).eval()
        self.emo = AutoModel(model=str(ST / 'hf/emotion2vec_plus_large'), disable_update=True, device=device)
        self.clip, _ = clip.load(str(ST / 'hf/ViT-B-32.pt'), device=device, jit=False)
        clip.model.convert_weights(self.clip); self.clip.eval(); self._tok = clip.tokenize
        with open(ST / 'beat2stub/weights/vocab.pkl', 'rb') as f:
            self.lang = pickle.load(f)

    def text(self, s):
        return self.clip.encode_text(self._tok(s, truncate=True).to(self.device)).float()

    def emo_label(self, a16, i0, i1):
        """beat2_add_hubert.get_max_score_label (16 kHz waveform, frames at 30 fps)."""
        seg = a16[int(i0 / 30 * 16000):int(i1 / 30 * 16000)]
        r = self.emo.generate(seg, granularity='utterance', extract_embedding=False, disable_pbar=True)
        return max(zip(r[0]['labels'], r[0]['scores']), key=lambda x: x[1])[0]

    @torch.no_grad()
    def __call__(self, wav, words, n, pre=4, rl=60, mark=None):
        """mark: optional callable(stage) adding the time since the previous mark to that stage (latency harness)."""
        mark = mark or (lambda _s: None)
        k = max(1, int(np.ceil((n - pre) / rl))); N = pre + k * rl
        a_def, sr_def = librosa.load(str(wav))                    # librosa default 22.05 kHz (official)
        a_raw, sr_raw = librosa.load(str(wav), sr=None)
        need = N / 30.0 + 0.1                                     # zero-pad so every feature covers N frames
        a_def = np.pad(a_def, (0, max(0, int(np.ceil(need * sr_def)) - len(a_def))))
        a_raw = np.pad(a_raw, (0, max(0, int(np.ceil(need * sr_raw)) - len(a_raw))))
        a16 = librosa.resample(a_def, orig_sr=sr_def, target_sr=16000)
        beat = extract_rhythm_pause_features(a_raw, sr_raw)
        mel = librosa.feature.melspectrogram(y=a16, sr=16000, n_mels=128, hop_length=int(16000 / 30))[..., :-1]
        hub = hubert_long(self.hubert, self.proc, torch.from_numpy(a_def).unsqueeze(0), self.device)
        hub = F.interpolate(hub.swapaxes(-1, -2).unsqueeze(0), size=mel.shape[-1], mode='linear',
                            align_corners=True).swapaxes(-1, -2)[0]
        assert len(beat) >= N and len(hub) >= N, (len(beat), len(hub), N)
        mark('audio_feat')
        wid, wsent = word_tracks(words, N, N / 30.0, self.lang)
        sents = [process_annotated_sentence(wsent[i * rl:(i + 1) * rl + pre]) for i in range(k)]
        mark('text_enc')
        a16_lab, _ = librosa.load(str(wav), sr=16000)             # get_max_score_label reads the file at 16 kHz
        a16_lab = np.pad(a16_lab, (0, max(0, int(np.ceil(need * 16000)) - len(a16_lab))))
        emos = [self.emo_label(a16_lab, i * rl, (i + 1) * rl + pre) for i in range(k)]
        mark('audio_feat')                                        # emotion2vec is an audio encoder
        dv = self.device
        out = dict(k=k, N=N, sents=sents, emos=emos,
                   word=torch.from_numpy(wid[:N])[None].to(dv),
                   beat=torch.from_numpy(beat[:N]).float()[None].to(dv),
                   hubert=hub[:N].float()[None].to(dv))
        mark('audio_feat')
        out.update(feat_clip=torch.stack([self.text([s]) for s in sents], 1),
                   emo_clip=torch.stack([self.text([e]) for e in emos], 1))
        mark('text_enc')                                          # CLIP text encoder (sentence + emotion label)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stems', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    stems = [l.strip() for l in open(a.stems) if l.strip()]
    stems = stems[:a.limit] if a.limit else stems
    si, sn = (int(x) for x in a.shard.split('/')); stems = stems[si::sn]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    torch.manual_seed(0); np.random.seed(0)
    args, base, sparse, vq, ctx = build(dev)
    seed_lat, trans0 = seed_latent(args, ctx, dev)
    fx = Feat(dev)
    print('MODEL_READY', len(stems), flush=True)
    t0 = time.time()
    for i, s in enumerate(stems):
        op = out / f'{s}.npz'
        if op.exists():
            continue
        n = int(torch.load(EG / 'motion_39d_30fps' / f'{s}.pt', map_location='cpu', weights_only=True)['real_num_frames'])
        words = json.load(open(EG / 'annotations/words' / f'{s}.json'))['words']
        f = fx(EG / 'audio' / f'{s}.wav', words, n)
        poses, trans, exps = generate(args, base, sparse, vq, ctx, f, seed_lat, trans0, dev)
        assert len(poses) == f['N'] >= n and np.isfinite(poses).all() and np.isfinite(trans).all()
        np.savez_compressed(op, poses=poses[:n].astype(np.float32), trans=trans[:n].astype(np.float32),
                            expressions=exps[:n].astype(np.float32), betas=np.zeros(300, np.float32),
                            model='smplx2020', gender='neutral', mocap_frame_rate=30, source=s, n=n,
                            semtalk_frames=int(f['N']), windows=int(f['k']), sentences=np.array(f['sents']),
                            emotions=np.array(f['emos']), seed='demo/2_scott_0_1_1.npz[:4]', speaker_id=1)
        if i % 20 == 0:
            print(f'{i + 1}/{len(stems)} {s} n={n} N={f["N"]} emo={f["emos"][:2]} {time.time() - t0:.0f}s', flush=True)
    print('SEMTALK_DONE', len(stems), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
