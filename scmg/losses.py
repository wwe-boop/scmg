"""One training step's losses for every stage.

Total = flow matching (unknown latents)                                       [all stages]
      + w_aux  * planner chunk-mean prediction                                [all stages]
      + w_cont * decoded motion reconstruction + boundary pos/vel/acc continuity (frozen VAE decoder)
      + w_rhy  * audio rhythm alignment (hinge: predicted acceleration peaks on audio onsets/beats >= ground truth's)
      + w_kl   * style KL (free bits)                                          [style]
      + state-conditioned recovery: the flow/continuity targets become the simulated recovery trajectory  [state]
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .common import CHUNK, CHUNK_FRAMES, STRIDE, PROSODY_COLS, U80, J
from .model import INF, arrival_mask, stream_state, x2_arrival
from .sim import simulate, state_vector, ARMS
from .vae import recon_losses
from .data import require_causal_text


@dataclass
class StageCfg:
    p_stream: float = 0.0          # chunk-aware causal training: probability of a streaming (causal) sample
    horizons: tuple = (0, 2, 4, 8)  # audio horizon past the chunk start, in 80 ms units (0 = strictly causal)
    outpaint: bool = False         # random known/unknown masks
    p_noctx: float = 0.1           # instances without any known context
    w_aux: float = 0.1
    w_int: float = 0.0             # 11-03: chunk intent CE (needs cfg.intent_aux + batch intent); text-dropped samples excluded
    sem_w: float = 0.0             # 11-03: flow-loss weight 1 + sem_w on chunks that contain an executed semantic gesture
    int_none_w: float = 0.2        # CE class weight of 'none' (0): ~90 % of labelled chunks
    # 11-03 user: Semantic Gesture Head (cfg.sem_head / sem_cond) and selective audio dropout
    w_gest: float = 0.0            # CE of the coarse g_k head (text-dropped samples excluded)
    gest_none_w: float = 0.3       # CE class weight of g = none
    gest_cw: tuple = ()            # 11-05: optional per-class CE weights (len n_sem; overrides gest_none_w), () = old behaviour
    sem_tf: float = 0.5            # P(Local Flow gets the TRUE g_k one-hot) on labelled chunks; else the head's prediction
    sem_drop: float = 0.1          # P(g_k -> null (zeros)) per sample, so g can be guided like a condition
    gest_grad_scale: float = 1.0   # 11-03 review: scale of the gesture-CE gradient that reaches the plan h (head itself trains fully)
    w_word: float = 0.0            # 11-03 word-level intent CE on the LM text-token states (cfg.vox_word_aux, batch tilab)
    word_none_w: float = 0.2       # CE class weight of 'none' (0)
    w_slot: float = 0.0            # 11-03 soft-intent-slot presence BCE (cfg.vox_slots, needs the gest frame labels)
    fsq_sat_w: float = 0.0         # 11-03 penalty on FSQ inputs beyond |z| > fsq_sat_bound (saturated FSQ cuts the text off)
    fsq_sat_bound: float = 1.2           # tanh(x)*4 reaches the outermost FSQ level from |x| >= 1.35
    text_focus_p: float = 0.0      # 11-03 P(batch is TEXT-FOCUSED): mel (and TTS rows) hidden, prosody kept, no text dropout
    sel_audio_premask: bool = False  # remove hidden chunks from encoder input as well as direct audio routes
    p_sel_audio: float = 0.0       # P(per text-bearing sample) that the audio of every gesture chunk is hidden (planner + flow)
    w_tail: float = 0.0            # 11-06 post-speech tails: flow-loss weight 1 + w_tail on target latents after the audio end (aend)
    sel_audio_max_class: int = 0   # 11-05 G1: >0 = only chunks with 1 <= g_k <= this class are hidden (G1 gest track: 1..3 deictic /
                                   # iconic / metaphoric, 4 beat on ~88 % of chunks); 0 = every gesture chunk (old behaviour)
    w_foot: float = 0.0            # 11-05 G1 only: foot-contact consistency on the decoded [ctx | target] frames (scmg/g1_contact.py:
    w_safe: float = 0.0            # 11-05 G1 only: executable-contract loss on the decoded [ctx | target] frames (scmg/g1_safety.py:
                                   # URDF joint limits + calibrated left-right / arm-torso self-collision proxies); 0 = off
    safe_torso_w: float = 0.5      # weight of the weaker arm-torso proxy inside w_safe
                                   # sole xy velocity + lowest-sole height vs ground truth on GT-contact frames, official geometry)
    w_tdelta: float = 0.0          # 11-04 ECHO-G temporal loss (temporal_loss_weight 0.5 there): squared error of the change of
                                   # the velocity-field error between adjacent steps; with the identity codec (z = STRIDE*J)
                                   # also between adjacent FRAMES inside / across latents (raw-space temporal coherence)
    w_cont: float = 0.0
    w_rhy: float = 0.0
    w_kl: float = 0.05
    free_bits: float = 0.05
    p_state: float = 0.0           # fraction of instances with a simulated execution deviation
    state_hist_measured: bool = False   # ablation: feed the measured history as context latents instead of a state token
    sigma_tf: float = 0.5          # planner teacher-forcing noise (exposure bias; r5: 0.2 -> 0.5)
    ctx_noise_p: float = 0.9       # r5: probability that the KNOWN context latents are noised (exposure bias: with clean
    ctx_t: tuple = (0.75, 1.0)     # GT context in training, free-running speed fell to 0.28 of GT vs 0.99 with GT history)
    ctx_infer_t: float = 0.9       # context noise level used at inference by models trained with ctx noise
    w_phys: float = 0.0            # stage-1 physical regularisation on the decoded chunk (11-02 user proposal): hardware joint
                                   # limits, velocity/acceleration/jerk beyond the robot's limits, self-collision (FK + calibrated
                                   # sphere proxy, scmg/kinematics.py); boundary continuity is the existing w_cont term
    phys_frames: int = 256         # frames per step that go through FK / collision (random subset of the decoded targets)
    phys_pairs: int = 40000        # sphere pairs per step (random subset; unbiased up to a constant)
    phys_dyn_w: float = 1.0        # 11-04 landing L1: weight of the velocity/acc/jerk term inside w_phys (0 = joint limits +
                                   # self-collision only; the full term made the upper arm ~20% smaller than the human)
    phys_coll_w: float = 1.0       # weight of the self-collision term inside w_phys
    phys_coll_huber: float = 0.0   # 11-08 (cm, > 0): Huber penetration penalty inside phys_coll -- pen^2 up to delta, delta (2 pen - delta)
                                   # beyond (per-pair gradient bounded by 2 delta); 0 = the squared hinge (bit-identical). Motivation:
                                   # skipped-step diagnosis (ex002-skipdiag-1108): all skips are gradient spikes driven by deep
                                   # penetrations of the model's own x1-hat (phys_coll 7.4 vs 0.06), GT penetration unchanged
    skip_diag: bool = False        # 11-08 diagnostic (train.py --skip-log): per-term |d loss / d y| of the physics / continuity
                                   # terms, model and GT max penetration on the same frame / pair subset (no extra random draw);
                                   # '_'-prefixed out keys, not in the loss. Default off = bit-identical
    p_async: float = 0.0           # ADSC: among STREAMING samples, probability that the text also streams (else text final);
                                   # streaming samples never see timing of not-yet-voiced tokens (offline samples do)
    p_async_seg: float = 0.3       # text arrival unit: TTS segment (robot TTS / sentence-level engines) vs X2Streaming-TTS
    seg_lead: tuple = (0.0, 1.0)   # token streaming (rate below, commitment lag); segment text arrives 0-1 s before its audio
    x2_rate: tuple = (5.0, 40.0)   # committed text tokens / s (LLM + causal commitment)
    x2_lag: tuple = (0.05, 0.4)    # commitment + first-audio delay of a token (s)
    p_adsc_stream: float = 0.67    # ADSC lanes: share of streaming (online) samples; the rest is the offline mode
    self_force_p: float = 0.5      # r5: probability that a batch uses ONE-STEP SELF-FORCED history (each chunk generated
    self_force_steps: int = 4      # by the current model from GT history, no grad, then used as the next chunk's history)
    self_force_rounds: int = 1     # 11-03: >1 = MULTI-STEP self-forcing: round r regenerates every chunk from round r-1's generated
                                   # history, so after R rounds each chunk's history is an R-step own rollout (hold-attractor fix)
    self_force_cfg: float = 0.0   # >0: joint audio/text CFG; local null-audio visibility matches Streamer(cfg_null_fix=True)
    self_force_mix: float = 0.0   # per round, independently replace each generated chunk with GT with this probability
    self_force_noise: float = 0.0 # Gaussian std in normalised latent space; applied before GT mixing / padding restore
    self_force_scope: str = 'all'  # 11-08 (default 'all' = unchanged): which parameters learn from a self-forced batch.
                                   # 'encoders' = exposure-robust conditioning: every nn.Linear executed by the no-grad generation
                                   # pass (planner, Local Flow and their input / output projections) gets NO gradient on that
                                   # batch (grad None -> AdamW skips it); the gradient still flows through their activations to
                                   # the condition encoders (audio / text / speaker / source ...). This is exactly what the
                                   # pre-dcd50fb code did by accident (autocast weight cache filled under no_grad) -- the
                                   # recipe of the lively EX002 sibag-s0/1/2. Single process, --accum 1 only (train.py checks).
    dim_mask: tuple = ()           # 11-04 ECHO-G G1 'leg masking' report lane (default off): motion channels with ZERO weight in
                                   # the decoded-motion losses (w_cont / rhythm unaffected: arms only); pair with the VAE's
                                   # dim_mask so the latents do not carry them at all (view META.json lists legs / root indices)
    # 11-05 input-modality ablation (default off = bit-identical: the dropout draws are made and consumed exactly as before):
    # every sample drops the text (audio-only model) / the audio (text-only model) as a drop_text / drop_audio sample does;
    # the other modality keeps its independent 10 % dropout, so the joint null (CFG unconditional branch) is still learned
    force_drop_text: bool = False
    force_drop_audio: bool = False
    # 11-05 训练匹配 (training-matched causal streaming inputs; default off = bit-identical, nothing extra is drawn or
    # computed). Need the cz_* batch keys (scmg.train --causal-dir <precompute_causal.py output>); batches without them
    # (the dev loss) and samples of clips outside the precompute (cz_ok False) keep the original inputs. Only STREAMING
    # samples change; t_k = window start + (k * CHUNK + H) * STRIDE frames (chunk_audio_mask) = scmg.infer.Streamer's Tk,
    # and h_k / the Local Flow of chunk k see the conditions AS OF t_k, as the Streamer at tick k (SCMG.plan per target).
    causal_front: bool = False     # encoder input of chunk k = the causal prefix features at t_k: mel rows of the clip (equal
                                   # row for row to the prefix mel) + prosody block 80-88 rebuilt from the words ended by t_k
                                   # (causal_feat.front_prosody, bitwise = prefix_feat89), rows >= t_k zero
    causal_spk: bool = False       # global condition of chunk k = the longest precomputed prefix CAM++ grade (1/2/4/8 s,
                                   # whole clip) available at t_k (< 1 s: the empty speaker -> spk_null)
    spk_empty_mean: bool = False   # (11-06, B2) with causal_spk: an EMPTY prefix speaker (no grade available by t_k, or an
                                   # invalid grade) -> the mean training CAM++ embedding (cz_spkmean, SPK_MEAN.npy of the
                                   # precompute) with spk_valid 1, as CausalInputs(spk_empty=...) at inference; drop_spk keeps spk_null
    causal_text: bool = False      # streaming async text: a token is visible once its last word has ENDED by now + H (psync
                                   # 'voiced'), on top of the x2 arrival; ADSC text state 2 only once the audio is final
    # 11-07 §7.55 item 2: the audio carries only word-content-free signals -- the 80 mel columns of feat89 are zeroed (= the
    # normalised mean) and TTS hidden rows hidden in EVERY training batch; inference must use infer_test.py --prosody-only. Default off.
    audio_prosody_only: bool = False
    # 11-07 item 5b mask-conditioned Local Flow (cfg.mask_cond; default off): P(sample) of a random clean latent span of
    # mask_span seconds anywhere in the window; its latents are known (clean, t = 1) in every chunk instance that sees them, flagged
    # by model.f_mask, and excluded from the loss -> library insertion at inference becomes in-distribution inpainting
    p_mask: float = 0.0
    # 11-07 control-token look-ahead (default 0): annotation (group-0) control tokens -- commands from the dialogue system, not
    # from speech -- arrive ctl_lookahead s before their time at the latest and their time is known ctl_lookahead s before it is
    # spoken (the voiced rule shifted by ctl_lookahead; streaming samples)
    ctl_lookahead: float = 0.0
    mask_span: tuple = (0.5, 2.0)
    # 11-06 modality-balanced gradient modulation (OGM-GE, Peng et al. CVPR 2022; default off): every step the flow loss is also
    # evaluated (no grad, same noise / times / history) with the audio dropped and with the text dropped; their increase over the
    # full loss = each modality's contribution. scmg.train scales the GRADIENT of the dominant modality's encoder by
    # 1 - tanh(alpha (ratio - 1)) (+ Gaussian noise of the gradient's std, 'GE'); see train.ogm_modulate.
    ogm_alpha: float = 0.0
    ogm_ge: bool = True
    ogm_ema: float = 0.9           # EMA of the contribution ratio over steps (single batches are noisy)
    # 11-05 explicit control tokens (cfg.ctl / ctl_flow + batch ctl_* from View(ctl_tokens=...); without them nothing below runs)
    ctl_drop: float = 0.5          # P(sample sees no token of a group), annotation / lexical groups independently -> the no-token
                                   # behaviour stays trained (and CFG-able)
    ctl_field_drop: float = 0.0    # P(each value field of a kept token is masked): partial tokens (target only / amplitude only)
    ctl_w: float = 0.0             # flow-loss weight 1 + ctl_w on positions inside [a - 0.3, b + 0.3] of a kept token
    ret_drop: float = 0.5          # 11-06 retrieval route (cfg.ret + batch ret_*): P(sample sees no retrieved exemplar)
    ctl_t_jitter: float = 0.0      # 11-06 interval tokens: annotation-token time += U(-j, +j) s per token (the token says "a gesture of
                                   # this class within this ~2j s span"; the Local Flow places the stroke itself); 0 = exact times
    ctl_jitter_types: tuple = ()   # 11-06 round 5: only these token types are jittered (scmg/ctl.py TYPE_ID, e.g. (1, 2) deictic / iconic);
                                   # () = every annotation token (round-3 behaviour)
    ctl_lr_scale: float = 1.0      # 11-06: learning-rate multiplier of the control-token modules (scmg.train param group)
    ctl_cf_p: float = 0.0          # 11-06 counterfactual control: P(sample's kept POINT tokens get ANOTHER token's target, rolled over
                                   # the batch); their segment [a - 0.3, b + 0.3] is removed from the flow loss and only the ray loss
                                   # (w_ctl_aux, toward the swapped target) supervises it -> following the token is the only way down
    # 11-07 §7.55 item 3 count control (default off): soft-count loss on the assembled decoded target frames of every kept count
    # token (scmg/ctl.py peak_score soft = the evaluation's hard count with sigmoids), and counterfactual counts (P per sample: every
    # kept count token gets a different random count 0..4, the loss pulls toward it). ctl_count_thr = APEX.json action speed (m/s).
    w_ctl_count: float = 0.0
    ctl_cf_count_p: float = 0.0
    # 11-07 duplex mode: COUNTERFACTUAL YIELD (default off; needs cfg ctl + batch ctl_* arrays + causal inputs). P(streaming covered
    # sample) of an interrupt event at a stream-grid time t_y = k_y CHUNK + H inside the speech: a YIELD control token arrives at t_y
    # (never dropped), from t_y on the audio is digital silence (causal_front rows [t_y, t_k) = the feat89 of digital silence, the
    # prosody of the words ended by t_y, speaker grade capped at t_y), the words not ended by t_y are withdrawn, and the motion target
    # after t_y is replaced by a critically damped retraction to the rest pose (waist + arms; legs / base held; 95 % at yield_rest_s);
    # the bridge [t_y, t_y + yield_rest_s) is excluded from the flow / GT-reconstruction / aux losses (free, as ctlcf segments), the
    # held rest pose after it is supervised. yield_human: human_ref json with rest.R_sil.joints (29).
    yield_cf_p: float = 0.0
    yield_rest_s: float = 1.4
    yield_mask_rest_p: float = 0.0  # 11-08 duplex ladder (b): P(a counterfactual-yield sample has its post-bridge latents [t_y + rest, end)
                                    # flagged as known inserted latents, cfg.mask_cond) -> rest insertion at inference is in-distribution;
                                    # 0 = off (no extra random draw, bit-identical)
    yield_human: str = ''
    ctl_count_thr: float = 0.5918
    # 11-07 EX002 finger commands (default off): soft extended-finger count loss (w_ctl_finger) and counterfactual counts
    w_ctl_finger: float = 0.0
    w_ctl_ray_el: float = 0.0      # 11-08 module 7 (default off): extra elevation term 1 - cos(el(ray) - el(wrist -> target)) on the POINT apex frame
    w_ctl_ray_az: float = 0.0      # 11-08 module 7b (default off): azimuth term 1 - cos(az(ray) - az(wrist -> target)) on the POINT apex frame
    w_ctl_raise: float = 0.0       # 11-08 module 7 (default off): hinge relu(shoulder z - 0.05 - wrist z) (m) when the target is above the shoulder
    ctl_cf_finger_p: float = 0.0
    w_ctl_aux: float = 0.0         # pointing-ray loss on the decoded apex frame of kept POINT tokens: 1 - cos(elbow->wrist ray,
                                   # wrist->target), official FK (needs the decode path; weighted like w_cont by the noise level)


def logit_normal(n, device, generator=None):
    return torch.sigmoid(torch.randn(n, device=device, generator=generator))


def segments(x, K, before, after):
    """x [B, T, ...] -> per chunk [B, K, before+after, ...] frames [16k - before, 16k + after) (edge-replicated)."""
    T = x.shape[1]
    idx = (torch.arange(K, device=x.device)[:, None] * CHUNK_FRAMES - before + torch.arange(before + after, device=x.device)[None])
    valid = (idx >= 0) & (idx < T)
    return x[:, idx.clamp(0, T - 1)], valid


def chunk_intent_labels(frames, n_classes):
    """Dominant GESTURE intent (id >= 2) with >= 4 frames (80 ms), ties -> lowest id; else none (0) if every frame is labelled,
    -1 (ignored) otherwise.  A chunk with 4 gesture frames and 12 none frames is that gesture (11-03 Claude)."""
    counts = F.one_hot(frames.clamp(min=0), n_classes) * (frames >= 2)[..., None]   # gesture intents only (0 none, 1 rest)
    counts = counts.sum(-2)
    count, label = counts.max(-1)
    none = torch.where((frames >= 0).all(-1), torch.zeros_like(label), torch.full_like(label, -1))
    return torch.where(count >= 4, label, none)


def _stream_lut(K, horizons, device):
    """latents -> row of the cz_pro table (causal_feat.stream_lat_grid, the order WindowDataset wrote it in)."""
    from .causal_feat import stream_lat_grid
    grid = stream_lat_grid(K, horizons, CHUNK, U80)
    lut = torch.full((grid[-1] + 1,), -1, dtype=torch.long, device=device)
    lut[torch.tensor(grid, device=device)] = torch.arange(len(grid), device=device)
    return lut


def causal_front_feat(batch, bi, Hb, K, horizons):
    """[len(bi), K, T, 89] encoder input of chunk k of samples bi (horizon Hb latents): cz_feat with the FRONT_COLS of
    the cz_pro row of t_k = (k * CHUNK + Hb) latents, rows >= t_k zero (scripts/tests/test_causal_train.py compares it
    with causal_feat.CausalInputs.feat(t_k))."""
    from .causal_feat import FRONT_COLS
    dev = batch['cz_feat'].device
    lat = torch.arange(K, device=dev)[None] * CHUNK + Hb[:, None]                     # [Bs, K] stream time (latents)
    j = _stream_lut(K, horizons, dev)[lat]
    if bool((j < 0).any()): raise ValueError('stream time outside the cz_pro grid (StageCfg.horizons != the data loader\'s)')
    fk = batch['cz_feat'][bi][:, None].repeat(1, K, 1, 1)                              # [Bs, K, T, 89]
    fk[..., list(FRONT_COLS)] = batch['cz_pro'][bi[:, None], j]
    T = fk.shape[2]
    keep = (torch.arange(T, device=dev)[None, None] < lat[..., None] * STRIDE)[..., None]
    return torch.where(keep, fk, torch.zeros_like(fk))


def causal_front_tokens(model, batch, a, stream, H, drop_audio, K, horizons, tfocus, ycf=None):
    """Training-matched audio tokens (StageCfg.causal_front): [B, K, N, d], row k of a streaming sample with causal inputs
    = the causal encoder on chunk k's own prefix input (cz_feat with the FRONT_COLS of cz_pro at t_k, rows >= t_k zero --
    what Streamer feeds the encoder at tick k); other samples keep their shared tokens. Same audio dropout / text-focus."""
    B, N = a.shape[0], a.shape[1]
    sel = stream & batch['cz_ok'].bool()
    a4 = a[:, None].expand(-1, K, -1, -1)
    if not bool(sel.any()): return a4
    bi = sel.nonzero()[:, 0]; Bs = bi.numel(); dev = a.device
    fk = causal_front_feat(batch, bi, H[bi], K, horizons)
    if ycf is not None: fk = yield_silence(fk, batch, bi, H[bi], K, horizons, ycf)
    if tfocus: fk[..., :80] = 0.
    ak = model.audio_tokens(fk.flatten(0, 1), torch.ones(Bs * K, dtype=torch.bool, device=dev)).view(Bs, K, N, -1)
    ak = torch.where(drop_audio[bi][:, None, None, None], model.audio_null.to(ak.dtype).expand_as(ak), ak)
    a4 = a4.clone(); a4[bi] = ak.to(a4.dtype)
    return a4


def causal_spk_select(batch, H, K, t_cap=None):
    """Prefix speaker grade of chunk k: (spk [B, K, 192], spk_valid [B, K], grade [B, K] (-1 = none available)) --
    the grade with the largest availability frame cz_spkt <= t_k (clip frames), ties -> the prefix grade."""
    B = H.shape[0]; dev = H.device; ar = torch.arange(B, device=dev)[:, None]
    tk = batch['start'][:, None] + (torch.arange(K, device=dev)[None] * CHUNK + torch.clamp(H, max=10 ** 6)[:, None]) * STRIDE
    if t_cap is not None: tk = torch.minimum(tk, batch['start'][:, None] + t_cap[:, None])     # 11-07 yield: no speech after t_y
    av = batch['cz_spkt']                                                             # [B, 5]
    can = av[:, None, :] <= tk[..., None]                                             # [B, K, 5]
    score = torch.where(can, av[:, None, :] * 2 + (torch.arange(5, device=dev) < 4).long(), torch.full_like(can, -1, dtype=torch.long))
    jsel = score.argmax(-1); has = can.any(-1)
    spk = torch.where(has[..., None], batch['cz_spk'][ar, jsel], torch.zeros_like(batch['cz_spk'][:, 0])[:, None])   # none: the empty
    return spk, batch['cz_spkv'][ar, jsel] * has.float(), torch.where(has, jsel, torch.full_like(jsel, -1))   # speaker (zeros, 0)


def causal_spk_cond(model, batch, stream, H, drop_spk, src, K, sc=None, t_cap=None):
    """Per-chunk global condition [B, K, d] (StageCfg.causal_spk): streaming chunk k of a covered sample uses the prefix
    CAM++ grade with the largest availability frame <= t_k (clip frames; ties -> the prefix grade, whose value is the
    inference one; none -> empty speaker, spk_valid 0); everything else the original speaker. drop_spk applies on top."""
    B = stream.shape[0]
    use = (stream & batch['cz_ok'].bool())[:, None].expand(B, K)
    spk_c, val_c, _ = causal_spk_select(batch, H, K, t_cap)
    if sc is not None and sc.spk_empty_mean:
        if 'cz_spkmean' not in batch: raise ValueError('spk_empty_mean needs SPK_MEAN.npy in the causal precompute')
        empty = val_c < .5
        spk_c = torch.where(empty[..., None], batch['cz_spkmean'][:, None].to(spk_c.dtype), spk_c); val_c = torch.where(empty, torch.ones_like(val_c), val_c)
    spk = torch.where(use[..., None], spk_c, batch['spk'][:, None].expand(B, K, -1))
    val = torch.where(use, val_c, batch['spk_valid'][:, None].expand(B, K)) * (~drop_spk)[:, None].float()
    g = model.global_cond(spk.reshape(B * K, -1), val.reshape(B * K), src.repeat_interleave(K))
    return g.view(B, K, -1)


_YREST = {}


def yield_silence(fk, batch, bi, Hb, K, horizons, ycf):
    """11-07 counterfactual yield: chunk k of a cf sample with t_k > t_y sees rows < t_y with the prosody of the prefix at t_y (cz_pro
    row of t_y), rows [t_y, t_k) = digital silence (normalised SIL feat89), rows >= t_k zero (not arrived, as before)."""
    from .causal_feat import FRONT_COLS
    on = ycf['on'][bi]
    if not bool(on.any()): return fk
    dev = fk.device; T = fk.shape[2]; ly = ycf['ly'][bi]                                          # [Bs] latents
    lat = torch.arange(K, device=dev)[None] * CHUNK + Hb[:, None]                                 # t_k (latents)
    after = on[:, None] & (lat > ly[:, None])                                                     # [Bs, K]
    j = _stream_lut(K, horizons, dev)[ly.clamp(min=0)]
    if bool((j[on] < 0).any()): raise ValueError('yield time outside the cz_pro grid')
    fk = fk.clone(); rows = torch.arange(T, device=dev)
    pre = (rows[None, None] < (ly * STRIDE)[:, None, None]) & after[..., None]                    # [Bs, K, T]
    sil = (rows[None, None] >= (ly * STRIDE)[:, None, None]) & (rows[None, None] < (lat * STRIDE)[..., None]) & after[..., None]
    pro = batch['cz_pro'][bi, j.clamp(min=0)]                                                     # [Bs, T, 9]
    cols = list(FRONT_COLS)
    fp = fk[..., cols]; fk[..., cols] = torch.where(pre[..., None], pro[:, None].to(fk.dtype), fp)
    fk = torch.where(sil[..., None], ycf['sil'].to(fk.dtype), fk)
    return fk


def yield_cf_prepare(model, vae, batch, sc, stats, stream, H, K, x1, q):
    """draw the counterfactual-yield samples and rewrite their batch rows (q, x1, text mask, control tokens); returns (batch, q, x1, ycf)."""
    from .ctl import YIELD, NV, NVM
    from .causal_feat import WORD_EPS
    dev = q.device; B, T, Jn = q.shape; N = T // STRIDE
    if not getattr(model.cfg, 'ctl', False): raise ValueError('yield_cf_p needs cfg ctl')
    if 'ctl_type' not in batch or 'cz_ok' not in batch: return batch, q, x1, None     # e.g. the dev-loss batches: no counterfactual
    if not sc.yield_human: raise ValueError('yield_cf_p needs yield_human (rest pose)')
    if sc.yield_human not in _YREST:
        import json as _j
        from .common import P as _P
        from pathlib import Path as _Pa
        _YREST[sc.yield_human] = torch.tensor(_j.loads(_Pa(_P(sc.yield_human)).read_text())['rest']['R_sil']['joints'], dtype=torch.float32)
    rest = _YREST[sc.yield_human].to(dev)
    nb = int(round(sc.yield_rest_s * 50 / STRIDE))                                                # bridge latents
    nval = (batch['mask'].view(B, N, STRIDE).min(-1).values > .5).sum(1)                          # valid latents
    Hl = torch.clamp(H, max=10 ** 6)
    kmax = torch.div(nval - Hl - nb - CHUNK, CHUNK, rounding_mode='floor')                        # t_y + bridge + a chunk inside the clip
    elig = stream & batch['cz_ok'].bool() & (kmax >= 1)
    on = elig & (torch.rand(B, device=dev) < sc.yield_cf_p)
    ky = 1 + (torch.rand(B, device=dev) * kmax.clamp(min=1).float()).long().clamp(max=10 ** 6)
    ky = torch.minimum(ky, kmax.clamp(min=1))
    ly = torch.where(on, ky * CHUNK + Hl, torch.full_like(ky, -1))                                # yield time (window latents)
    if not bool(on.any()):
        return batch, q, x1, None
    batch = dict(batch)
    fy = ly * STRIDE; ts = fy.float() / 50.                                                       # window frame / s
    # motion: critically damped retraction from the frame before t_y (infer_retract._targets rule)
    qm, qs = stats['qm'].to(dev), stats['qs'].to(dev)
    qp = q.float() * qs + qm; qn = q.clone()
    wn = 4.744 / max(sc.yield_rest_s, 1e-3)
    if Jn == 39:                    # G1 robot_repr: root 0:10 + 29 joints; waist + arms = channels 22..38; root velocities zeroed
        up = list(range(22, 39)); full = lambda q0_: torch.cat([q0_[:10], rest]); zero_root = True
    else:                           # 11-08 EX002 34 joints (rad): waist (0, 1) + arms (4..17) to the 34-joint rest pose; head / fingers held
        if rest.numel() != Jn: raise ValueError(f'yield_human rest has {rest.numel()} joints, motion has {Jn}')
        up = [0, 1] + list(range(4, 18)); full = lambda q0_: rest; zero_root = False
    for b in on.nonzero()[:, 0].tolist():
        f0 = int(fy[b]); q0 = qp[b, f0 - 1]; v0 = (qp[b, f0 - 1] - qp[b, f0 - 2]) * 50.
        tt = torch.arange(1, T - f0 + 1, device=dev).float()[:, None] / 50.
        e = q0 - full(q0)
        tr = q0[None].repeat(T - f0, 1)
        upv = full(q0)[None] + (e + (v0 + wn * e) * tt) * torch.exp(-wn * tt)
        tr[:, up] = upv[:, up]
        if zero_root: tr[:, 6:10] = 0.
        qn[b, f0:] = ((tr - qm) / qs).to(q.dtype)
    with torch.no_grad():
        mu, _ = vae.encode(qn[on])
    x1 = x1.clone(); x1[on] = (mu - model.lat_mean) / model.lat_std
    batch['q'] = qn
    # text: words not ended by t_y are withdrawn (whole window, as Streamer after the interruption)
    if 'cz_tend' in batch:
        keep = torch.nan_to_num(batch['cz_tend'], nan=1e9) <= ts[:, None] + WORD_EPS
        batch['tmask'] = torch.where(on[:, None], batch['tmask'] & keep, batch['tmask'])
    # control token: YIELD at t_y in the first free slot (never dropped)
    ct = batch['ctl_type'].clone(); free = (ct == 0)
    slot = torch.where(free.any(1), free.float().argmax(1), torch.full_like(ct[:, 0], ct.shape[1] - 1))
    ar = torch.arange(B, device=dev)
    put = torch.zeros_like(free); put[ar[on], slot[on]] = True
    for k_, val in (('ctl_t', ts), ('ctl_a', ts), ('ctl_b', ts + sc.yield_rest_s)):
        x = batch[k_].clone(); x[put] = val[:, None].expand_as(x)[put].to(x.dtype); batch[k_] = x
    x = batch['ctl_apex'].clone(); x[put] = float('nan'); batch['ctl_apex'] = x
    ct[put] = YIELD; batch['ctl_type'] = ct
    g_ = batch['ctl_g'].clone(); g_[put] = 0; batch['ctl_g'] = g_
    for k_ in ('ctl_v', 'ctl_vm'):
        x = batch[k_].clone(); x[put] = 0.; batch[k_] = x
    sil = ycf_sil(stats, dev)
    ycf = dict(on=on, ly=ly, nb=nb, cap=torch.where(on, fy, torch.full_like(fy, 10 ** 9)), put=put, sil=sil)
    return batch, qn, x1, ycf


def ycf_sil(stats, dev):
    """normalised feat89 row of digital silence (logmel log(1e-6) in float16 = -13.8125 as causal_feat emits it, prosody 0)."""
    import numpy as _np
    raw = _np.concatenate([_np.full(80, _np.log(1e-6)), _np.zeros(9)]).astype(_np.float16).astype(_np.float32)   # f16 as causal_feat
    fm = _np.asarray(stats['fm_raw'], _np.float32); fsd = _np.asarray(stats['fs_raw'], _np.float32)
    return torch.as_tensor((raw - fm) / fsd, device=dev)


def step_losses(model, vae, batch, sc: StageCfg, stats):
    require_causal_text(batch.get('text_c'), sc.p_async > 0 and sc.p_async_seg < 1, 'ADSC training')
    cfg = model.cfg; dev = batch['q'].device
    q, mask, feat = batch['q'], batch['mask'], batch['feat']
    if sc.audio_prosody_only: feat = feat.clone(); feat[..., :80] = 0.
    if sc.audio_prosody_only and 'tts_hm' in batch: batch = dict(batch, tts_hm=torch.zeros_like(batch['tts_hm']))
    B, T, Jn = q.shape; N = T // STRIDE; K = N // CHUNK
    from .bagger import target_chunks
    supervise, bagger = target_chunks(batch, K)
    if bagger is not None and cfg.state:
        raise ValueError('BAgger corrective targets cannot be combined with state simulation')
    qm, qs = stats['qm'], stats['qs']
    with torch.no_grad():
        mu, _ = vae.encode(q)
        x1 = (mu - model.lat_mean) / model.lat_std                                   # [B, N, z]
    lmask = mask.view(B, N, STRIDE).min(-1).values > .5
    lat_k = x1.view(B, K, CHUNK, -1)

    # ---- condition dropout / modes
    r = lambda p: torch.rand(B, device=dev) < p
    drop_text, drop_audio, drop_spk = r(cfg.drop_text), r(cfg.drop_audio), r(cfg.drop_spk)
    if bagger is not None:
        drop_text = drop_text | bagger
        drop_spk = drop_spk | batch.get('bagger_null', torch.zeros_like(bagger))
        drop_audio = drop_audio | batch.get('bagger_null', torch.zeros_like(bagger))
    tfocus = False
    if sc.text_focus_p > 0 and float(torch.rand(())) < sc.text_focus_p:
        # text-focused batch (alternates with joint batches): the speech CONTENT channels are hidden -- mel (80 cols, 0 = the
        # normalised mean) and TTS hidden rows -- while prosody (onset/energy/f0/beat) keeps the timing; text is never dropped
        feat = feat.clone(); feat[..., :80] = 0.; tfocus = True
        if 'tts_hm' in batch: batch = dict(batch, tts_hm=torch.zeros_like(batch['tts_hm']))
        drop_text = torch.zeros_like(drop_text)
    if sc.force_drop_text: drop_text = torch.ones_like(drop_text)          # audio-only model (after text_focus: it wins)
    if sc.force_drop_audio: drop_audio = torch.ones_like(drop_audio)       # text-only model
    if sc.p_async > 0:                    # ADSC lanes: offline / online text-final / online text-streaming
        stream = r(sc.p_adsc_stream); asy = stream & r(sc.p_async)
    else:
        stream = r(sc.p_stream); asy = torch.zeros_like(stream)
    hz = U80 * torch.tensor(sc.horizons, device=dev)[torch.randint(len(sc.horizons), (B,), device=dev)]
    H = torch.where(stream, hz, torch.full_like(hz, INF))
    ycf = None
    if sc.yield_cf_p > 0:                 # 11-07 counterfactual yield (default off: no draw, nothing changes)
        batch, q, x1, ycf = yield_cf_prepare(model, vae, batch, sc, stats, stream, H, K, x1, q)
        lat_k = x1.view(B, K, CHUNK, -1)
    if not sc.sel_audio_premask:  # keep the default execution / RNG order unchanged
        a = model.audio_tokens(feat, stream, batch.get('tts_h'), batch.get('tts_hm'))
        a = torch.where(drop_audio[:, None, None], model.audio_null.expand_as(a), a)
    amask_k = model.chunk_audio_mask(K, N, H, dev) & ~drop_audio[:, None, None]
    glab = hide = None
    if 'gest' in batch:                   # coarse g_k per chunk: majority class with >= 80 ms, 0 none, -1 unknown
        fg = batch['gest'].long().view(B, K, CHUNK_FRAMES)
        cnt = F.one_hot(fg.clamp(min=0), model.cfg.n_sem)[..., 1:].sum(2)                # [B, K, n_sem - 1]
        glab = torch.where(cnt.max(-1).values >= 4, cnt.argmax(-1) + 1,
                           torch.where((fg >= 0).all(-1), torch.zeros_like(fg[..., 0]), torch.full_like(fg[..., 0], -1)))
        if sc.p_sel_audio > 0:            # selective audio dropout: gesture chunks of text-bearing samples lose their audio
            hide = (r(sc.p_sel_audio) & ~drop_text & ~drop_audio)[:, None] & (glab >= 1)   # [B, K]
            if sc.sel_audio_max_class > 0: hide = hide & (glab <= sc.sel_audio_max_class)
            amask_k = amask_k & ~hide[:, :, None]
    if sc.sel_audio_premask:
        th, hm = batch.get('tts_h'), batch.get('tts_hm')
        if hide is not None:
            hidden_frames = hide.repeat_interleave(CHUNK_FRAMES, 1)
            feat = feat.masked_fill(hidden_frames[..., None], 0.)
            if th is not None:
                hidden_rows = hidden_frames[:, ::4]
                th = th.masked_fill(hidden_rows[..., None], 0.)
                hm = (~hidden_rows).to(th.dtype) if hm is None else hm.masked_fill(hidden_rows, 0.)
        a = model.audio_tokens(feat, stream, th, hm)
        a = torch.where(drop_audio[:, None, None], model.audio_null.expand_as(a), a)
        if hide is not None:
            hidden_tokens = hide.repeat_interleave(CHUNK, 1)
            a = torch.where(hidden_tokens[..., None], model.audio_null.expand_as(a), a)
            amask_k = amask_k & ~hidden_tokens[:, None, :]  # planner AND MTC lose the hidden source tokens
    if (sc.causal_front or sc.causal_spk) and (cfg.mtc or cfg.planner != 'chunk' or cfg.state or sc.p_sel_audio > 0):
        raise ValueError('causal_front / causal_spk: chunk planner without mtc / state / selective audio dropout only')
    if sc.causal_front and 'cz_pro' in batch:
        if sc.sel_audio_premask or getattr(cfg, 'tts_h', False):
            raise ValueError('causal_front: not implemented with sel_audio_premask / tts_h')
        a = causal_front_tokens(model, batch, a, stream, H, drop_audio, K, sc.horizons, tfocus, ycf)
    dtime = r(cfg.drop_time)
    text, tmask = model.text_tokens(batch['text'], batch['ttime'], batch['tmask'] & ~drop_text[:, None], drop=dtime)
    tsec = model.token_seconds(batch['ttime'], dtime)                                # flow_text_xattn: usable token times
    tidm_d = batch['tidm'] & ~drop_text[:, None] if 'tidm' in batch else None   # VoxCPM LM text: same text dropout
    tvis = lvis = sstate = None
    if sc.p_async > 0:                    # ADSC: what has arrived at the start of each chunk (window-relative seconds)
        U = lambda lohi: lohi[0] + (lohi[1] - lohi[0]) * torch.rand(B, device=dev)
        seg = r(sc.p_async_seg); rate, lag, slead = U(sc.x2_rate), U(sc.x2_lag), U(sc.seg_lead)
        now = torch.arange(K, device=dev).float() * CHUNK_FRAMES / 50.
        final = ~asy
        tvalid = batch['tmask'] & ~drop_text[:, None]
        arr = torch.where(seg[:, None], batch['tseg'] - slead[:, None], x2_arrival(batch['ttime'], tvalid, rate, lag))
        tvis = arrival_mask(arr, tvalid, final, now)
        use_ct = None
        if sc.causal_text and 'cz_tend' in batch:                                     # psync 'voiced': the word has ended
            use_ct = asy & ~seg & batch['cz_ok'].bool()
            Hs = torch.clamp(H, max=10 ** 6).float() * STRIDE / 50.
            from .causal_feat import WORD_EPS                                      # the shared word-ended predicate
            ended = torch.nan_to_num(batch['cz_tend'], nan=1e9)[:, None, :] <= (now[None] + Hs[:, None])[..., None] + WORD_EPS
            tvis = torch.where(use_ct[:, None, None], tvis & ended, tvis)
        if tidm_d is not None and 'titime' in batch:
            larr = torch.where(seg[:, None], batch['tiseg'] - slead[:, None], x2_arrival(batch['titime'], tidm_d, rate, lag))
            lvis = arrival_mask(larr, tidm_d, final, now)
        nvalid = mask.sum(1)                                                          # frames before the clip end
        if 'aend' in batch:               # 11-06 post-speech tails: the audio stream ends at the speech offset, motion goes on
            nvalid = torch.minimum(nvalid, batch['aend'].to(nvalid))
        Hc = torch.clamp(H, max=10 ** 6)
        reach = (torch.arange(K, device=dev)[None] * CHUNK + Hc[:, None]) * STRIDE
        afin = (H[:, None] >= INF) | ((nvalid[:, None] < T) & (reach >= nvalid[:, None]))
        if hasattr(model, 'stream_emb'):
            sstate = stream_state(tvis, tvalid, afin)
            if use_ct is not None:                   # Streamer voiced / asr: text 'final' only once the audio is final
                nv = tvis.sum(-1); nt = tvalid.sum(-1, keepdim=True)
                s_ct = torch.where(afin & (nv >= nt), 2, torch.where(nv > 0, 1, 0)) * 2 + afin.long()
                sstate = torch.where(use_ct[:, None], s_ct, sstate)
        if sstate is not None and hasattr(model, 'floor_emb') and 'floor' in batch:   # floor state of every chunk (at its start)
            fk = batch['floor'][:, ::CHUNK_FRAMES][:, :K].clamp(min=0).long()
            sstate = sstate + 6 * fk
        # timing exists only for voiced tokens when streaming (audio, and with it the anchors, arrived up to now + H)
        heard = now[None] + Hc[:, None].float() * STRIDE / 50.                        # [B, K]
        voiced = torch.isfinite(batch['ttime'])[:, None] & (torch.nan_to_num(batch['ttime'], 1e9)[:, None] <= heard[:, :, None])
        voiced = voiced | ~stream[:, None, None]
        raw = batch['text']
        if 'text_c' in batch:             # token streaming: prefix-causal features (a bidirectional encoder sees later words)
            raw = torch.where((asy & ~seg)[:, None, None], batch['text_c'], raw)
            dtime_c = r(cfg.drop_time)
            tt_, _ = model.text_tokens(raw, batch['ttime'], tmask, drop=dtime_c)
            text = torch.where((asy & ~seg)[:, None, None], tt_, text)
            tsec = torch.where((asy & ~seg)[:, None], model.token_seconds(batch['ttime'], dtime_c), tsec)
        tu, _ = model.text_tokens(raw, batch['ttime'], tmask, drop=torch.ones(B, dtype=torch.bool, device=dev))
        text = torch.cat([text, tu], 1); tmask = torch.cat([tmask, tmask], 1)
        tsec = torch.cat([tsec, torch.full_like(tsec, float('nan'))], 1)              # the untimed copy has no time
        tvis = torch.cat([tvis & voiced, tvis & ~voiced], -1)
    ctl_c = ctl = None
    if getattr(cfg, 'ctl', False) and 'ctl_type' in batch:
        from .ctl import anchor_arrival, visibility
        ct = batch['ctl_t'].float()
        if sc.ctl_t_jitter > 0:
            jm = (batch['ctl_g'] == 0)
            if sc.ctl_jitter_types: jm = jm & torch.isin(batch['ctl_type'], torch.tensor(list(sc.ctl_jitter_types), device=dev))
            ct = ct + jm.float() * (torch.rand(ct.shape, device=dev) * 2 - 1) * sc.ctl_t_jitter
        gdrop = torch.stack([r(sc.ctl_drop), r(sc.ctl_drop)], 1)                    # annotation / lexical groups dropped independently
        cm = (batch['ctl_type'] > 0) & torch.isfinite(ct) & ~torch.gather(gdrop, 1, batch['ctl_g'].long().clamp(0, 1))
        if ycf is not None: cm = cm | ycf['put']                                    # 11-07: a counterfactual YIELD is never dropped
        cvm = batch['ctl_vm'].float()
        if sc.ctl_field_drop > 0: cvm = cvm * (torch.rand(cvm.shape, device=dev) >= sc.ctl_field_drop).float()
        cval = batch['ctl_v'].float(); cf = None
        if sc.ctl_cf_p > 0:
            from .ctl import POINT
            pt = cm & (batch['ctl_type'] == POINT) & (cvm[..., 0] > 0)
            cf = pt & r(sc.ctl_cf_p)[:, None]
            bi, ji = pt.nonzero(as_tuple=True)
            if len(bi) > 1:
                src = torch.roll(torch.arange(len(bi), device=dev), 1)                     # another token's target
                newv = cval.clone(); newv[bi, ji, :3] = cval[bi[src], ji[src], :3]
                cval = torch.where(cf[..., None], newv, cval)
            else: cf = torch.zeros_like(pt)
        if sc.ctl_cf_finger_p > 0:        # 11-07 EX002: every kept finger command gets random counts 0..3 (P per sample)
            from .ctl import POINT
            isf = cm & (batch['ctl_type'] != POINT) & (batch['ctl_g'] == 0) & (cvm[..., 0] > 0)
            fcf = isf & r(sc.ctl_cf_finger_p)[:, None]
            rnd = (torch.randint(0, 4, cval[..., :2].shape, device=dev).float() - 1.5) / 1.5
            cval = cval.clone(); cval[..., :2] = torch.where(fcf[..., None], rnd, cval[..., :2])
        if sc.ctl_cf_count_p > 0:
            from .ctl import POINT, COUNT_MAX, count_value
            isc = cm & (batch['ctl_type'] != POINT) & (batch['ctl_g'] == 0) & (cvm[..., 0] > 0)
            ccf = isc & r(sc.ctl_cf_count_p)[:, None]
            n0 = torch.round(cval[..., 0] * (COUNT_MAX / 2) + COUNT_MAX / 2)
            n1 = torch.remainder(n0 + torch.randint(1, COUNT_MAX + 1, n0.shape, device=dev), COUNT_MAX + 1)
            cval = cval.clone(); cval[..., 0] = torch.where(ccf, (n1 - COUNT_MAX / 2) / (COUNT_MAX / 2), cval[..., 0])
        u = model.ctl_embed(batch['ctl_type'].long() * cm.long(), cval, cvm)
        nowk = torch.arange(K, device=dev).float() * CHUNK_FRAMES / 50.
        if sc.p_async > 0:                # arrives with the word it is attached to; time known once voiced (streaming samples)
            arr_c = anchor_arrival(arr, batch['ttime'], tvalid, ct)
            if ycf is not None or sc.yield_cf_p > 0:                              # 11-07: YIELD = an event, it arrives at its own time
                from .ctl import YIELD
                arr_c = torch.where(batch['ctl_type'] == YIELD, torch.nan_to_num(ct, 0.).float(), arr_c)
            if sc.ctl_lookahead > 0:
                g0 = batch['ctl_g'] == 0
                arr_c = torch.where(g0, torch.minimum(arr_c, torch.nan_to_num(ct, 0.).float() - sc.ctl_lookahead), arr_c)
            keep, timed = visibility(cm, ct, arr_c, final, nowk, heard, stream)
            if sc.ctl_lookahead > 0:          # its time is known ctl_lookahead s before it is spoken (voiced-rule shifted)
                early = (torch.nan_to_num(ct, 1e9)[:, None, :] - sc.ctl_lookahead <= heard[:, :, None])
                timed = torch.where(g0[:, None, :], timed | (keep & early), timed)
        else:
            keep = cm[:, None, :].expand(B, K, -1); timed = keep
        tt_c, tu_c = model.ctl_text_tokens(u, ct)
        ctn = torch.where(cm, ct, torch.full_like(ct, float('nan')))
        if tvis is not None:
            text = torch.cat([text, tt_c.to(text.dtype), tu_c.to(text.dtype)], 1); tmask = torch.cat([tmask, cm, cm], 1)
            tvis = torch.cat([tvis, timed, keep & ~timed], -1); tsec = torch.cat([tsec, ctn, torch.full_like(ctn, float('nan'))], 1)
        else:
            text = torch.cat([text, tt_c.to(text.dtype)], 1); tmask = torch.cat([tmask, cm], 1); tsec = torch.cat([tsec, ctn], 1)
        qsec = model.flow_seconds(torch.arange(K, device=dev), cfg.ctx + CHUNK, cfg.ctx)
        cc = model.ctl_flow_cond(u, ct, keep, timed, qsec)
        if cc is not None: ctl_c = cc.flatten(0, 1)
        ctl = dict(cm=cm, keep=keep, qsec=qsec, cval=cval, cf=cf, cvm=cvm)
    if getattr(cfg, 'ret', False) and 'ret_t' in batch:      # retrieved exemplars: arrive with their word, used once their time is known
        from .ctl import anchor_arrival, visibility
        rt = batch['ret_t'].float(); rm = batch['ret_m'].bool() & torch.isfinite(rt) & ~r(sc.ret_drop)[:, None]
        nowr = torch.arange(K, device=dev).float() * CHUNK_FRAMES / 50.
        if sc.p_async > 0:
            _, rtimed = visibility(rm, rt, anchor_arrival(arr, batch['ttime'], tvalid, rt), final, nowr, heard, stream)
        else: rtimed = rm[:, None, :].expand(B, K, -1)
        rc = model.ret_flow_cond(batch['ret_lat'].float(), rt, rtimed, model.flow_seconds(torch.arange(K, device=dev), cfg.ctx + CHUNK, cfg.ctx))
        if rc is not None: ctl_c = rc.flatten(0, 1) if ctl_c is None else ctl_c + rc.flatten(0, 1).to(ctl_c.dtype)
    st_plan = None
    if getattr(cfg, 'sem_time', False):                    # 11-07 semantic-first temporal stream (text as the planner reads it)
        sf_, st_plan = model.sem_time_cond(text, tmask, tvis, tsec, K)
        if sf_ is not None: ctl_c = sf_.flatten(0, 1) if ctl_c is None else ctl_c + sf_.flatten(0, 1).to(ctl_c.dtype)
    src = torch.where(r(0.1), torch.zeros_like(batch['source']), batch['source'])
    g = model.global_cond(batch['spk'], batch['spk_valid'] * (~drop_spk), src)
    if sc.causal_spk and 'cz_spk' in batch:
        g = causal_spk_cond(model, batch, stream, H, drop_spk, src, K, sc, t_cap=None if ycf is None else ycf['cap'])

    # ---- state simulation (stage 4): replaces targets of the selected chunks by recovery trajectories
    seg_q, seg_valid = segments(q, K, CHUNK_FRAMES * 3, CHUNK_FRAMES)                # [B, K, 64, J] normalised
    seg_q = seg_q.flatten(0, 1); seg_valid = seg_valid[None].expand(B, -1, -1).flatten(0, 1)
    M = B * K
    tgt_lat = lat_k.flatten(0, 1)                                                     # [M, CHUNK, z]
    ctx_lat = torch.cat([torch.zeros_like(lat_k[:, :1]), lat_k[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
    gt_frames = seg_q[:, -2 * CHUNK_FRAMES:]                                          # [M, 32, J] ctx chunk + target
    state_tok = state_ada = None
    if cfg.state:
        plan_rad = seg_q * qs + qm
        boundary = 3 * CHUNK_FRAMES
        sel = torch.rand(M, device=dev) < sc.p_state
        probs = (0., .3, .25, .25, .2)
        # inference timing: the streamer emits a frame one latent late, so when chunk k is generated the robot has executed
        # up to frame 16k - STRIDE; the last STRIDE frames before the boundary are committed-but-not-yet-sent PLAN frames.
        # 11-02 review fix: the deviation is simulated ONLY up to the last observed frame (e_ - 1) and the recovery target
        # starts from that observed state; frames e_..boundary-1 are the pending plan (no unobserved disturbance there).
        e_ = boundary - STRIDE                                                        # first not-yet-executed frame
        act, _, _, target, kind = simulate(plan_rad, e_, probs=probs)
        target = torch.where(sel[:, None, None], target, plan_rad)
        act = torch.cat([act, plan_rad[:, e_:boundary]], 1)                           # executed + pending plan frames
        hist_rad = torch.where(sel[:, None, None], act, plan_rad[:, :boundary])
        sq = hist_rad[:, e_ - 1]; sdq = (hist_rad[:, e_ - 1] - hist_rad[:, e_ - 2]) * 50.
        q_plan_s = plan_rad[:, e_ - 1]
        tn = (target - qm) / qs
        with torch.no_grad():
            mu_t, _ = vae.encode(tn)
            lat_t = (mu_t - model.lat_mean) / model.lat_std                         # [M, 16, z]
        tgt_lat = torch.where(sel[:, None, None], lat_t[:, -CHUNK:], tgt_lat)
        if sc.state_hist_measured:                    # measured = executed frames + the pending plan frames, as at inference
            meas = torch.cat([hist_rad[:, :e_], plan_rad[:, e_:boundary]], 1)
            with torch.no_grad():
                mu_m, _ = vae.encode((meas - qm) / qs)
            lat_m = (mu_m - model.lat_mean) / model.lat_std
            ctx_lat = torch.where(sel[:, None, None], lat_m[:, -cfg.ctx:], ctx_lat)
        gt_frames = tn[:, -2 * CHUNK_FRAMES:]
        sv = state_vector(sq, sdq, q_plan_s, qm, qs)
        st_tok, st_ada = model.state_ad(sv)
        state_tok = st_tok.view(B, K, -1); state_ada = st_ada
        if sc.state_hist_measured:
            state_tok = None; state_ada = None

    # ---- planner (teacher forcing with noise)
    sref = style_tok = None; kl = None
    if cfg.style:
        ok = (batch['ref_valid'] > .5) & ~r(cfg.drop_ref)
        sref = torch.where(ok[:, None], model.style_enc(batch['ref']), model.style_null)
    mt = mt_mask = plan_input = None

    def run_planner(hist):
        nonlocal mt, mt_mask, plan_input
        lat_in = hist + sc.sigma_tf * torch.rand(B, 1, 1, 1, device=dev) * torch.randn_like(hist)
        if bagger is not None: lat_in = torch.where(bagger[:, None, None, None], lat_k, lat_in)
        plan_input = lat_in             # CFG branches share the SAME teacher-noised planner history
        if cfg.mtc:
            mt, mt_mask = model.mt_memory(text, tmask, a, amask_k, g, sref, hist, state_tok, K, tvis=tvis)
            mflat, pm = model.planner_memory(mt, mt_mask)
            return model.plan(lat_in, a, amask_k, text, tmask, g, mflat, pm, batch.get('tids'), tidm_d, sref,
                              tvis=tvis, lvis=lvis, sstate=sstate, **({} if st_plan is None else dict(padd=st_plan)))
        return model.plan(lat_in, a, amask_k, text, tmask, g, tids=batch.get('tids'), tidm=tidm_d, sref=sref,
                          tvis=tvis, lvis=lvis, sstate=sstate, **({} if st_plan is None else dict(padd=st_plan)))

    ftext = model.flow_text(text, tmask, tvis, tsec, K)                             # None without flow_text_xattn
    aw, awm = model.local_audio(a, H, K)
    if hide is not None:                  # the hidden chunks' Local Flow audio window = the null audio (as for dropped audio)
        aw = torch.where(hide[:, :, None, None], model.audio_null.to(aw.dtype).expand_as(aw), aw)
    aw, awm = aw.flatten(0, 1), awm.flatten(0, 1)
    gi = g.flatten(0, 1) if g.dim() == 3 else g[:, None].expand(-1, K, -1).flatten(0, 1)
    Lf = cfg.ctx + CHUNK
    hist = lat_k
    if sc.self_force_p > 0 and not cfg.state and float(torch.rand(())) < sc.self_force_p:
        # exposure-bias fix: generate every chunk from GT history (parallel, no grad), then train chunk k on the
        # GENERATED chunk k-1 as its history (planner input, multi-token history token and local-flow context)
        # 11-04 fix: the no-grad generation pass must NOT fill autocast's weight cache -- with cache_enabled the bf16
        # weight copies made here (no grad_fn) were reused by the gradient pass below, so on self-forced batches the
        # planner and the Local Flow got NO gradient (324 of 406 tensors None).  Same autocast dtype, cache off.
        from .model import autocast_dtype
        _ac = torch.autocast('cuda', dtype=autocast_dtype(), enabled=torch.is_autocast_enabled(), cache_enabled=False)
        if sc.self_force_scope not in ('all', 'encoders'): raise ValueError(f'self_force_scope {sc.self_force_scope!r}')
        _sf_mods, _sf_hooks = [], []
        if sc.self_force_scope == 'encoders':     # record the Linear modules the generation pass executes (= legacy cache set)
            for _m in model.modules():
                if isinstance(_m, torch.nn.Linear):
                    _sf_hooks.append(_m.register_forward_hook(lambda mod, i, o: _sf_mods.append(mod)))
        with torch.no_grad(), _ac:
            src = lat_k
            for _round in range(max(1, sc.self_force_rounds)):
                h0 = run_planner(src)
                st0 = model.style_head(sref[:, None].expand(-1, K, -1), h0)[0].flatten(0, 1) if cfg.style else None
                ctx0 = torch.cat([torch.zeros_like(src[:, :1]), src[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
                kn0 = torch.zeros(M, Lf, dtype=torch.bool, device=dev); kn0[:, :cfg.ctx] = (torch.arange(K, device=dev).repeat(B) > 0)[:, None]
                x = torch.cat([ctx0, torch.randn(M, CHUNK, cfg.z, device=dev)], 1)
                mt0 = (mt.flatten(0, 1), mt_mask.flatten(0, 1)[:, None].expand(-1, Lf, -1)) if cfg.mtc else (None, None)
                if sc.self_force_cfg > 0:
                    # Mirror Streamer: planner hides audio/text (including VoxCPM ids), but retains speaker/style,
                    # history and arrival state. Local Flow keeps the null-audio visibility mask (cfg_null_fix=True).
                    a_u = model.audio_null.expand_as(a); am_u = torch.zeros_like(amask_k)
                    tm_u = torch.zeros_like(tmask)
                    tidm_u = torch.zeros_like(tidm_d) if tidm_d is not None else None
                    if cfg.mtc:
                        mt_u, mtm_u = model.mt_memory(text, tm_u, a_u, am_u, g, sref, src, state_tok, K, tvis=tvis)
                        mf_u, pm_u = model.planner_memory(mt_u, mtm_u)
                        h_u = model.plan(plan_input, a_u, am_u, text, tm_u, g, mf_u, pm_u, batch.get('tids'), tidm_u,
                                         sref, tvis=tvis, lvis=lvis, sstate=sstate)
                        mt0_u = (mt_u.flatten(0, 1), mtm_u.flatten(0, 1)[:, None].expand(-1, Lf, -1))
                    else:
                        h_u = model.plan(plan_input, a_u, am_u, text, tm_u, g, tids=batch.get('tids'), tidm=tidm_u,
                                         sref=sref, tvis=tvis, lvis=lvis, sstate=sstate)
                        mt0_u = (None, None)
                    h0_u = model.sem_h(h_u).flatten(0, 1)
                    ftext_u = model.flow_text(text, tm_u, tvis, tsec, K)              # text hidden like the planner's
                    aw_u = model.audio_null.expand_as(aw); awm_u = awm.clone()
                for i in range(sc.self_force_steps):
                    tt = torch.full((M, Lf), i / sc.self_force_steps, device=dev); tt[kn0] = 1.
                    v0 = model.flow_velocity(x, tt, kn0, model.sem_h(h0).flatten(0, 1), gi, aw, awm, st0, None, *mt0,
                                             ftext=ftext, **({} if ctl_c is None else dict(ctl_c=ctl_c)))
                    if sc.self_force_cfg > 0:
                        v_u = model.flow_velocity(x, tt, kn0, h0_u, gi, aw_u, awm_u, st0, None, *mt0_u, ftext=ftext_u)
                        v0 = v_u + sc.self_force_cfg * (v0 - v_u)
                    x = torch.where(kn0[..., None], x, x + v0 / sc.self_force_steps)
                gen = x[:, cfg.ctx:].view(B, K, CHUNK, -1)
                if sc.self_force_noise > 0:
                    gen = gen + sc.self_force_noise * torch.randn_like(gen)
                if sc.self_force_mix > 0:
                    use_gt = torch.rand(B, K, 1, 1, device=dev) < sc.self_force_mix
                    gen = torch.where(use_gt, lat_k, gen)
                gen = torch.where(lmask.view(B, K, CHUNK)[..., None], gen, lat_k)
                src = gen
        for _h in _sf_hooks: _h.remove()
        if sc.self_force_scope == 'encoders':     # train.py sets grad None for these after backward (before clipping / step)
            _seen = {}
            for _m in _sf_mods:
                for _p in _m.parameters(recurse=False):
                    if _p.requires_grad: _seen[id(_p)] = _p
            model._sf_frozen = list(_seen.values())
        hist = gen
        if bagger is not None:
            gen = torch.where(bagger[:, None, None, None], lat_k, gen)
            hist = gen
        ctx_lat = torch.cat([torch.zeros_like(gen[:, :1]), gen[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
        out_sf = 1.
    else:
        out_sf = 0.
    h = run_planner(hist)
    if cfg.mtc: mtp, mtpm = mt.flatten(0, 1), mt_mask.flatten(0, 1)
    if cfg.style:
        style_tok, kl, _ = model.style_head(sref[:, None].expand(-1, K, -1), h, x_target=lat_k)
        style_tok = style_tok.flatten(0, 1)

    # ---- local flow instances
    L = cfg.ctx + CHUNK
    x1i = torch.cat([ctx_lat, tgt_lat], 1)                                            # [M, L, z]
    known = torch.zeros(M, L, dtype=torch.bool, device=dev)
    kk = torch.arange(K, device=dev).repeat(B)
    has_ctx = (kk > 0) & ~(torch.rand(M, device=dev) < sc.p_noctx)
    if sc.outpaint:
        c = torch.randint(0, cfg.ctx + 1, (M,), device=dev)                           # keep the last c context latents
        known[:, :cfg.ctx] = torch.arange(cfg.ctx, device=dev)[None] >= (cfg.ctx - c)[:, None]
        extra = torch.randint(1, CHUNK, (M,), device=dev) * (torch.rand(M, device=dev) < .3)
        known[:, cfg.ctx:] = torch.arange(CHUNK, device=dev)[None] < extra[:, None]
    else:
        known[:, :cfg.ctx] = True
    known[:, :cfg.ctx] &= has_ctx[:, None]
    if bagger is not None:
        bm = bagger[:, None].expand(B, K).reshape(M)
        has_ctx = torch.where(bm, kk > 0, has_ctx)
        known[bm, :cfg.ctx] = has_ctx[bm, None]
        known[bm, cfg.ctx:] = (~supervise).reshape(M)[bm, None]
    ins = None
    if sc.p_mask > 0 and getattr(cfg, 'mask_cond', False):
        Nw = K * CHUNK; lo_, hi_ = (int(round(s_ * 50 / STRIDE)) for s_ in sc.mask_span)
        ln = torch.randint(lo_, hi_ + 1, (B,), device=dev); st_ = (torch.rand(B, device=dev) * (Nw - ln).clamp(min=1)).long()
        on = torch.rand(B, device=dev) < sc.p_mask
        gpos = (kk * CHUNK - cfg.ctx)[:, None] + torch.arange(L, device=dev)[None]                     # global latent of each position
        bb = torch.arange(B, device=dev).repeat_interleave(K)
        ins = (gpos >= st_[bb][:, None]) & (gpos < (st_ + ln)[bb][:, None]) & on[bb][:, None] & (gpos >= 0)
        ins[:, :cfg.ctx] &= known[:, :cfg.ctx]                    # a context position is inserted only where it exists
        known = known | ins
    if sc.yield_mask_rest_p > 0 and ycf is not None and getattr(cfg, 'mask_cond', False):   # 11-08 ladder (b): rest spans as insertions
        gpr = (kk * CHUNK - cfg.ctx)[:, None] + torch.arange(L, device=dev)[None]
        bbr = torch.arange(B, device=dev).repeat_interleave(K)
        selr = ycf['on'] & (torch.rand(B, device=dev) < sc.yield_mask_rest_p)
        insr = (gpr >= (ycf['ly'] + ycf['nb'])[bbr][:, None]) & selr[bbr][:, None] & (gpr >= 0)
        insr[:, :cfg.ctx] &= known[:, :cfg.ctx]
        ins = insr if ins is None else (ins | insr); known = known | insr
    t = logit_normal(M, dev)[:, None].expand(M, L).clone(); t[known] = 1.
    noise = torch.randn_like(x1i)
    xt = torch.where(known[..., None], x1i, t[..., None] * x1i + (1 - t[..., None]) * noise)
    if sc.ctx_noise_p > 0:                       # noisy-but-known context: its noise level is given to the model via t
        noisy = known.clone(); noisy[:, cfg.ctx:] = False
        noisy &= (torch.rand(M, device=dev) < sc.ctx_noise_p)[:, None]
        if bagger is not None: noisy &= ~bm[:, None]
        tc = (sc.ctx_t[0] + (sc.ctx_t[1] - sc.ctx_t[0]) * torch.rand(M, device=dev))[:, None].expand(M, L)
        xt = torch.where(noisy[..., None], tc[..., None] * x1i + (1 - tc[..., None]) * noise, xt)
        t = torch.where(noisy, tc, t)
    h_flow = h; g_logits = None
    if cfg.sem_head:
        hg = h if sc.gest_grad_scale >= 1 else h * sc.gest_grad_scale + h.detach() * (1 - sc.gest_grad_scale)
        g_logits = model.sem_head(hg.float())                                        # [B, K, n_sem]
        if cfg.sem_cond:
            gp = torch.softmax(g_logits, -1).detach()                                # flow loss does not train the head
            if glab is not None:
                tf = r(sc.sem_tf)[:, None] & (glab >= 0) & ~drop_text[:, None]
                gp = torch.where(tf[..., None], F.one_hot(glab.clamp(min=0), cfg.n_sem).float(), gp)
            gp = gp * (~r(sc.sem_drop)).float()[:, None, None]                       # null g_k
            h_flow = model.sem_h(h, gp)
    v = model.flow_velocity(xt, t, known, h_flow.flatten(0, 1), gi, aw, awm, style_tok, state_ada,
                            mtp if cfg.mtc else None, mtpm[:, None].expand(-1, L, -1) if cfg.mtc else None, ftext=ftext,
                            **({} if ctl_c is None else dict(ctl_c=ctl_c)), **({} if ins is None else dict(ins=ins)))
    tgt_valid = lmask.view(B, K, CHUNK).flatten(0, 1)
    if supervise is not None: tgt_valid = tgt_valid & supervise.reshape(M, 1)
    sup = ~known
    sup[:, cfg.ctx:] &= tgt_valid
    sup[:, :cfg.ctx] = False                                                          # past context never supervised
    if ctl is not None and ctl['cf'] is not None and ctl['cf'].any():   # counterfactual segments: no GT flow target
        a_, b_ = batch['ctl_a'].float(), batch['ctl_b'].float(); q_ = ctl['qsec'][None, :, :, None]
        ins_cf = ((q_ >= (a_ - 0.3)[:, None, None]) & (q_ <= (b_ + 0.3)[:, None, None]) & ctl['cf'][:, None, None]).any(-1)
        sup = sup & ~ins_cf.flatten(0, 1)
    ybr = ybr_f = None
    if ycf is not None:                   # 11-07 counterfactual yield: the bridge [t_y, t_y + rest) has no target
        bb_ = torch.arange(B, device=dev).repeat_interleave(K)
        gl_ = (kk * CHUNK - cfg.ctx)[:, None] + torch.arange(L, device=dev)[None]                     # window latent of each position
        ly_ = ycf['ly'][bb_][:, None]
        ybr = ycf['on'][bb_][:, None] & (gl_ >= ly_) & (gl_ < ly_ + ycf['nb'])                       # [M, L]
        sup = sup & ~ybr
        fr_ = (kk - 1)[:, None] * CHUNK_FRAMES + torch.arange(2 * CHUNK_FRAMES, device=dev)[None]     # window frame of [ctx | target]
        ybr_f = ycf['on'][bb_][:, None] & (fr_ >= ly_ * STRIDE) & (fr_ < (ly_ + ycf['nb']) * STRIDE)  # [M, 2 CF]
        out_y = dict(yield_cf_n=ycf['on'].float().sum())
    err = (v - (x1i - noise)).square().mean(-1)
    ilab = None
    if 'intent' in batch and (sc.w_int > 0 or sc.sem_w > 0):
        fi = batch['intent'].long().view(B, K, CHUNK_FRAMES)
        has = (fi >= 2).sum(-1) >= 4  # semantic weighting remains an existence test
        ilab = chunk_intent_labels(fi, cfg.n_intent)
    if ilab is not None and sc.sem_w > 0:
        wi = (1 + sc.sem_w * has.float()).flatten()[:, None]                   # [M, 1]
        flow = (err * sup * wi).sum() / (sup * wi).sum().clamp(min=1)
    elif sc.w_tail > 0 and 'aend' in batch:
        fr = (kk[:, None] * CHUNK + torch.arange(CHUNK, device=dev)[None]) * STRIDE                    # [M, CHUNK] window frame
        ae = batch['aend'].to(dev).repeat_interleave(K)[:, None]
        wt = torch.ones_like(err); wt[:, cfg.ctx:] = 1 + sc.w_tail * (fr >= ae).float()
        flow = (err * sup * wt).sum() / (sup * wt).sum().clamp(min=1)
    elif ctl is not None and sc.ctl_w > 0:  # positions inside a kept token's segment (+-0.3 s)
        a_, b_ = batch['ctl_a'].float(), batch['ctl_b'].float()
        q_ = ctl['qsec'][None, :, :, None]                                          # [1, K, L, 1]
        ins = (q_ >= (a_ - 0.3)[:, None, None]) & (q_ <= (b_ + 0.3)[:, None, None]) & ctl['cm'][:, None, None]
        wi = (1 + sc.ctl_w * ins.any(-1).float()).flatten(0, 1)                    # [M, L]
        flow = (err * sup * wi).sum() / (sup * wi).sum().clamp(min=1)
        ctl['frac'] = ins.any(-1).float().mean()
    else:
        flow = (err * sup).sum() / sup.sum().clamp(min=1)
    out = dict(flow=flow, self_forced=torch.tensor(out_sf, device=dev))
    if ycf is not None: out.update(out_y)
    if sc.ogm_alpha > 0:
        if cfg.mtc or cfg.planner != 'chunk': raise ValueError('ogm_alpha: chunk planner without mtc only')
        with torch.no_grad():
            base_l = (err * sup).sum() / sup.sum().clamp(min=1)
            def drop_loss(a_, am_, aw_, tm_, ctl_on):
                h_ = model.plan(plan_input, a_, am_, text, tm_, g, tids=batch.get('tids'), tidm=tidm_d, sref=sref, tvis=tvis,
                                lvis=lvis, sstate=sstate)
                ft_ = model.flow_text(text, tm_, tvis, tsec, K)
                v_ = model.flow_velocity(xt, t, known, model.sem_h(h_).flatten(0, 1), gi, aw_, awm, style_tok, state_ada, ftext=ft_,
                                         **({} if (ctl_c is None or not ctl_on) else dict(ctl_c=ctl_c)))
                return ((v_ - (x1i - noise)).square().mean(-1) * sup).sum() / sup.sum().clamp(min=1)
            l_na = drop_loss(model.audio_null.expand_as(a), torch.zeros_like(amask_k), model.audio_null.to(aw.dtype).expand_as(aw), tmask, True)
            l_nt = drop_loss(a, amask_k, aw, torch.zeros_like(tmask), False)
        out['ogm_da'] = (l_na - base_l).float(); out['ogm_dt'] = (l_nt - base_l).float()
    if ctl is not None and 'frac' in ctl: out['ctl_frac'] = ctl['frac']
    if sc.w_tdelta > 0:
        d = v - (x1i - noise)                                                        # [M, L, z] velocity error
        pair = sup[:, 1:] & sup[:, :-1]
        td = ((d[:, 1:] - d[:, :-1]).square().mean(-1) * pair).sum() / pair.sum().clamp(min=1)
        if d.shape[-1] == STRIDE * J:                                                # identity codec: frame-level deltas
            df = d.reshape(d.shape[0], -1, J); sf = sup.repeat_interleave(STRIDE, 1)
            pf = sf[:, 1:] & sf[:, :-1]
            td = ((df[:, 1:] - df[:, :-1]).square().mean(-1) * pf).sum() / pf.sum().clamp(min=1)
        out['tdelta'] = td
    aux = (model.p_aux(h) - lat_k.flatten(-2)).square().mean(-1)
    cm = lmask.view(B, K, CHUNK).all(-1).float()
    if supervise is not None: cm = cm * supervise
    if ycf is not None: cm = cm * (~ybr.view(B, K, L)[:, :, cfg.ctx:].any(-1)).float()                # no aux target in the bridge
    out['aux'] = (aux * cm).sum() / cm.sum().clamp(min=1)
    loss = out['flow'] + sc.w_aux * out['aux'] + (sc.w_tdelta * out['tdelta'] if sc.w_tdelta > 0 else 0.)
    vaux = getattr(getattr(model, 'vox', None), '_aux', None) or {}
    if sc.w_word > 0 and 'word_logits' in vaux and 'tilab' in batch:
        wl = vaux['word_logits']; tl = batch['tilab'].long()[:, :wl.shape[1]]
        tl = torch.where(batch['tidm'][:, :wl.shape[1]] & ~drop_text[:, None], tl, torch.full_like(tl, -1))
        if supervise is not None:
            times = batch.get('titime')
            if times is None: raise ValueError('BAgger word auxiliary requires token times')
            after = times[:, :wl.shape[1]] * 50 >= batch['loss_from_chunk'][:, None] * CHUNK_FRAMES
            tl = torch.where(~bagger[:, None] | after, tl, torch.full_like(tl, -1))
        cw = torch.ones(wl.shape[-1], device=dev); cw[0] = sc.word_none_w
        out['word'] = F.cross_entropy(wl.flatten(0, 1), tl.flatten(), weight=cw, ignore_index=-1) if (tl >= 0).any() else wl.sum() * 0.
        pos = tl >= 2
        out['word_acc'] = ((wl.argmax(-1) == tl) & pos).float().sum() / pos.float().sum().clamp(min=1)
        loss = loss + sc.w_word * out['word']
    if sc.w_slot > 0 and 'slot_logits' in vaux and glab is not None:
        slot_known = (glab >= 0).all(1) & ~drop_text                                # whole window labelled, text visible
        if supervise is not None: slot_known &= supervise.all(1)
        if slot_known.any():
            pres = torch.stack([(glab == c).any(1) for c in range(1, 7)], 1).float()      # gesture classes present in the window
            sl = vaux['slot_logits'][..., 1:].max(1).values                               # [B, 6] best slot per class
            out['slot'] = F.binary_cross_entropy_with_logits(sl[slot_known], pres[slot_known])
            loss = loss + sc.w_slot * out['slot']
    if sc.fsq_sat_w > 0 and 'fsq_pre' in vaux:
        sat = F.relu(vaux['fsq_pre'].float().abs() - sc.fsq_sat_bound).pow(2)
        if supervise is None:
            out['fsq_sat'] = sat.mean()
        else:
            sm = supervise.reshape(*supervise.shape, *([1] * (sat.ndim - 2))).expand_as(sat)
            out['fsq_sat'] = (sat * sm).sum() / sm.sum().clamp(min=1)
        loss = loss + sc.fsq_sat_w * out['fsq_sat']
    if g_logits is not None and glab is not None and sc.w_gest > 0:
        gl = torch.where(drop_text[:, None] | ~cm.bool(), torch.full_like(glab, -1), glab)
        cw = torch.ones(cfg.n_sem, device=dev); cw[0] = sc.gest_none_w
        if sc.gest_cw: cw = torch.tensor([float(x) for x in sc.gest_cw], device=dev)
        out['gest'] = F.cross_entropy(g_logits.flatten(0, 1), gl.flatten(), weight=cw, ignore_index=-1) if (gl >= 0).any() \
            else g_logits.sum() * 0.
        pos = gl >= 1
        out['gest_acc'] = ((g_logits.argmax(-1) == gl) & pos).float().sum() / pos.float().sum().clamp(min=1)
        for k in range(cfg.n_sem): out[f'glab_{k}'] = (gl == k).float().mean()                   # chunk label distribution (logged)
        loss = loss + sc.w_gest * out['gest']
    if ilab is not None and sc.w_int > 0 and hasattr(model, 'int_head'):
        il = torch.where(drop_text[:, None] | ~cm.bool(), torch.full_like(ilab, -1), ilab)   # no text -> intent unknowable
        cw = torch.ones(model.cfg.n_intent, device=dev); cw[0] = sc.int_none_w
        logits = model.int_head(h.float())
        out['intent'] = F.cross_entropy(logits.flatten(0, 1), il.flatten(), weight=cw, ignore_index=-1) if (il >= 0).any() \
            else logits.sum() * 0.
        pos = il >= 2
        out['intent_acc'] = ((logits.argmax(-1) == il) & pos).float().sum() / pos.float().sum().clamp(min=1)
        loss = loss + sc.w_int * out['intent']

    # ---- decoded continuity / rhythm
    if sc.w_cont > 0 or sc.w_rhy > 0 or sc.w_phys > 0 or sc.w_foot > 0 or sc.w_safe > 0 or \
            ((sc.w_ctl_aux > 0 or sc.w_ctl_count > 0 or sc.w_ctl_finger > 0) and ctl is not None):
        x1hat = torch.where(known[..., None], x1i, xt + (1 - t[..., None]) * v)
        # decode with the context the streamer has: 3 more history chunks before the known context (the decoder looks
        # ~7 latents back) and the next chunk's first latent as look-ahead (inference emits a frame only once the latent
        # after it is committed). The bare [ctx | target] window gave 9.9x boundary jerk / 0.77 deg error even for
        # ground-truth latents (scripts/vae_consistency.py) -> the loss was fighting decoder artefacts.
        Hx = 3 * CHUNK; Nl = lat_k.shape[1] * CHUNK
        src = hist.reshape(B, Nl, -1)                                                 # GT, or self-forced history
        padded = torch.cat([torch.zeros_like(src[:, :Hx + CHUNK]), src, x1.new_zeros(B, 1, src.shape[-1])], 1)
        kk_ = torch.arange(K, device=dev)
        ex_idx = kk_[:, None] * CHUNK + torch.arange(Hx, device=dev)[None]            # padded idx of latents [4k-16, 4k-4)
        extra = padded[:, ex_idx].flatten(0, 1)                                       # [M, Hx, z]
        nx = torch.clamp(kk_ + 1, max=K - 1) * CHUNK
        x1_full = torch.cat([x1, x1[:, -1:]], 1)
        nxt = torch.where((kk_ + 1 < K)[None, :, None], x1_full[:, nx], x1hat.view(B, K, L, -1)[:, :, -1]).flatten(0, 1)[:, None]
        zz = torch.cat([extra, x1hat, nxt], 1)
        yfull = vae.decode(zz * model.lat_std + model.lat_mean)
        y = yfull[:, Hx * STRIDE:(Hx + L) * STRIDE]                                   # frames of [ctx | target]
        fv = seg_valid[:, -2 * CHUNK_FRAMES:].float() * torch.cat(
            [has_ctx[:, None].float().expand(M, CHUNK_FRAMES), tgt_valid.float().repeat_interleave(STRIDE, 1)], 1)
        fv = fv * (kk.view(M) >= 4).float()[:, None]                                  # needs 4 chunks of history
        if supervise is not None: fv = fv * supervise.reshape(M, 1)
        fv_all = fv
        if ybr_f is not None: fv = fv * (~ybr_f).float()                              # 11-07 yield bridge: no GT reconstruction there
        w_inst = t[:, -1]                                                             # trust low-noise estimates more
        dw = None
        if sc.dim_mask:
            dw = torch.ones(Jn, device=dev); dw[list(sc.dim_mask)] = 0.
        rl = recon_losses(y[:, CHUNK_FRAMES - 4:], gt_frames[:, CHUNK_FRAMES - 4:], fv[:, CHUNK_FRAMES - 4:] * w_inst[:, None], dw)
        out.update({f'cont_{k}': v_ for k, v_ in rl.items()})
        loss = loss + sc.w_cont * (rl['rec'] + 0.3 * rl['vel'] + 0.1 * rl['acc'])
        if sc.w_foot > 0:
            from .common import EMBODIMENT
            if EMBODIMENT != 'g1': raise ValueError('StageCfg.w_foot is the G1 foot-contact term (SCMG_J=39)')
            from .g1_contact import foot_contact_loss
            fc = foot_contact_loss(y.float() * qs + qm, gt_frames.float() * qs + qm, fv * w_inst[:, None], 50.)
            out.update(fc); loss = loss + sc.w_foot * (fc['foot_slide'] + 10. * fc['foot_height'])   # m/s and m -> comparable scale
        if sc.w_safe > 0:
            from .common import EMBODIMENT
            if EMBODIMENT != 'g1': raise ValueError('StageCfg.w_safe is the G1 executable-contract term (SCMG_J=39)')
            from .g1_safety import contract_loss
            sf = contract_loss(y.float() * qs + qm, fv_all * w_inst[:, None])   # the executable contract holds in the bridge too
            out.update(sf); loss = loss + sc.w_safe * (sf['safe_lim'] + sf['safe_lr'] + sc.safe_torso_w * sf['safe_torso'])
        if sc.w_ctl_aux > 0 and ctl is not None:
            out.update(ctl_ray_loss(y.float() * qs + qm, batch, ctl, fv * w_inst[:, None], B, K, extra=sc.w_ctl_ray_el > 0 or sc.w_ctl_raise > 0 or sc.w_ctl_ray_az > 0))
            loss = loss + sc.w_ctl_aux * out['ctl_ray']
            if sc.w_ctl_ray_el > 0: loss = loss + sc.w_ctl_ray_el * out['ctl_ray_el']
            if sc.w_ctl_raise > 0: loss = loss + sc.w_ctl_raise * out['ctl_raise']
            if sc.w_ctl_ray_az > 0: loss = loss + sc.w_ctl_ray_az * out['ctl_ray_az']
        if sc.w_ctl_finger > 0 and ctl is not None:
            if '_upper' not in _EX:
                import json as _json
                from .data import VIEW as _V
                _EX['_upper'] = torch.tensor(_json.load(open(f'{_V}/META.json'))['ex002']['upper'], dtype=torch.float32)
            out.update(ctl_finger_loss(y.float() * qs + qm, batch, ctl, fv * w_inst[:, None], B, K, _EX['_upper'].to(dev)))
            loss = loss + sc.w_ctl_finger * out['ctl_finger']
        if sc.w_ctl_count > 0 and ctl is not None:
            out.update(ctl_count_loss(y.float() * qs + qm, batch, ctl, fv * w_inst[:, None], B, K, sc.ctl_count_thr))
            loss = loss + sc.w_ctl_count * out['ctl_count']
        if sc.w_phys > 0 and not sc.skip_diag:
            out.update(phys_reg(y, fv * w_inst[:, None], qm, qs, sc))
            loss = loss + sc.w_phys * (out['phys_joint'] + sc.phys_dyn_w * out['phys_dyn'] + sc.phys_coll_w * out['phys_coll'])
        elif sc.w_phys > 0:                                    # 11-08 skip diagnostic: the same terms + gradient / penetration probes
            pr = phys_reg(y, fv * w_inst[:, None], qm, qs, sc, gt=gt_frames)
            out.update({k_: v_ for k_, v_ in pr.items() if not k_.startswith('_')})
            terms = dict(phys_joint=sc.w_phys * pr['phys_joint'], phys_dyn=sc.w_phys * sc.phys_dyn_w * pr['phys_dyn'],
                         phys_coll=sc.w_phys * sc.phys_coll_w * pr['phys_coll'], cont=sc.w_cont * (rl['rec'] + 0.3 * rl['vel'] + 0.1 * rl['acc']))
            loss = loss + terms['phys_joint'] + terms['phys_dyn'] + terms['phys_coll']
            for k_, v_ in terms.items():
                if not (y.requires_grad and v_.requires_grad): continue           # dev evaluation (no grad)
                gy = torch.autograd.grad(v_, y, retain_graph=True, allow_unused=True)[0]
                out[f'_gy_{k_}'] = float(gy.float().norm()) if gy is not None else 0.
            out.update({k_: v_ for k_, v_ in pr.items() if k_.startswith('_')})
            out['_K'] = float(K)
        if sc.w_rhy > 0:
            yt = y[:, CHUNK_FRAMES:].reshape(B, K * CHUNK_FRAMES, Jn)
            gt = gt_frames[:, CHUNK_FRAMES:].reshape(B, K * CHUNK_FRAMES, Jn)
            beat = (feat[:, :K * CHUNK_FRAMES, PROSODY_COLS['onset']] + feat[:, :K * CHUNK_FRAMES, PROSODY_COLS['beat']]).clamp(min=0)
            fm = mask[:, :K * CHUNK_FRAMES]
            if supervise is not None: fm = fm * supervise.repeat_interleave(CHUNK_FRAMES, 1)
            def ba(m):
                acc = (m[:, 2:, ARMS] - 2 * m[:, 1:-1, ARMS] + m[:, :-2, ARMS]).norm(dim=-1)
                acc = acc / (acc * fm[:, 1:-1]).mean(1, keepdim=True).clamp(min=1e-4)
                return (acc * beat[:, 1:-1] * fm[:, 1:-1]).sum(1) / (beat[:, 1:-1] * fm[:, 1:-1]).sum(1).clamp(min=1e-3)
            out['rhy'] = F.relu(ba(gt).detach() - ba(yt)).mean()
            loss = loss + sc.w_rhy * out['rhy']
    if kl is not None:
        klm = kl.clamp(min=sc.free_bits).sum(-1)
        out['kl'] = (klm * cm).sum() / cm.sum().clamp(min=1)
        out['kl_raw'] = (kl.sum(-1) * cm).sum() / cm.sum().clamp(min=1)
        loss = loss + sc.w_kl * out['kl']
    out['loss'] = loss
    return out


def ctl_ray_loss(yp, batch, ctl, fw, B, K, extra=False):
    """yp [M, 2 * CHUNK_FRAMES, J] decoded [ctx | target] frames (physical), fw [M, F] frame weights. For every kept POINT
    token visible to the chunk that contains its GT apex frame: 1 - cos(elbow -> wrist ray, wrist -> target) at that frame."""
    from .common import EMBODIMENT
    if EMBODIMENT not in ('g1', 'ex002'): raise ValueError('StageCfg.w_ctl_aux: G1 (SCMG_J=39) or EX002 (SCMG_J=34) pointing term')
    from .ctl import POINT, POS_SCALE, arm_points
    dev = yp.device
    fap = torch.round(torch.nan_to_num(batch['ctl_apex'].float(), -1e3) * 50.).long()           # [B, Kc] window frame
    kap = torch.div(fap, CHUNK_FRAMES, rounding_mode='floor')
    ok = ctl['cm'] & (batch['ctl_type'] == POINT) & (batch['ctl_vm'][..., 0] > 0) & (fap >= 0) & (kap < K)
    bi, ji = ok.nonzero(as_tuple=True)
    if len(bi) == 0: return dict(ctl_ray=yp.sum() * 0., ctl_ray_n=torch.zeros((), device=dev))
    kk = kap[bi, ji]; vis = ctl['keep'][bi, kk, ji]
    bi, ji, kk = bi[vis], ji[vis], kk[vis]
    if len(bi) == 0: return dict(ctl_ray=yp.sum() * 0., ctl_ray_n=torch.zeros((), device=dev))
    m = bi * K + kk; lf = fap[bi, ji] - (kk - 1) * CHUNK_FRAMES                               # local frame in [ctx | target]
    hand = (batch['ctl_v'][bi, ji, 3] > 0).long(); ar = torch.arange(len(bi), device=dev)
    if EMBODIMENT == 'ex002':                                                                 # 11-07: EX002 exact FK, base frame
        from .ex002_ctl import points_t
        g = points_t(yp[m, lf]); w_ = g['wrist'][ar, hand]; e_ = g['elbow'][ar, hand]
    else:
        q = yp[m, lf][:, None, 10:39]                                                          # [n, 1, 29]
        g = arm_points(q)
        w_ = g['wrist'][ar, 0, hand]; e_ = g['elbow'][ar, 0, hand]
    tgt = ctl.get('cval', batch['ctl_v'])[bi, ji, :3].float() * POS_SCALE
    cos = F.cosine_similarity(w_ - e_, tgt - w_, dim=-1)
    wt = fw[m, lf]
    out = dict(ctl_ray=((1 - cos) * wt).sum() / wt.sum().clamp(min=1e-3), ctl_ray_n=torch.tensor(float(len(bi)), device=dev))
    if extra and EMBODIMENT == 'ex002':                                                       # 11-08 module 7 (default off)
        el = lambda v: torch.atan2(v[..., 2], v[..., :2].norm(dim=-1).clamp(min=1e-6))
        d = tgt - w_; out['ctl_ray_el'] = ((1 - torch.cos(el(w_ - e_) - el(d))) * wt).sum() / wt.sum().clamp(min=1e-3)
        s_ = g['shoulder'][ar, hand]; up = (tgt[:, 2] > s_[:, 2]).float()
        out['ctl_raise'] = (F.relu(s_[:, 2] - 0.05 - w_[:, 2]) * up * wt).sum() / wt.sum().clamp(min=1e-3)
        az = lambda v: torch.atan2(v[..., 1], v[..., 0])                                      # 11-08 module 7b
        out['ctl_ray_az'] = ((1 - torch.cos(az(w_ - e_) - az(d))) * wt).sum() / wt.sum().clamp(min=1e-3)
    return out


def ctl_finger_loss(yp, batch, ctl, fw, B, K, upper):
    """EX002 finger commands (11-07): for every kept non-POINT annotation token with a finger value (slots 0 / 1 = extended
    fingers left / right, (n - 1.5) / 1.5), visible to the chunk of its time: smooth-L1 between the soft extended-finger counts
    (scmg/ex002_ctl.py soft_counts) of the decoded frame at the token time and the commanded counts, both hands."""
    from .common import EMBODIMENT
    if EMBODIMENT != 'ex002': raise ValueError('StageCfg.w_ctl_finger is the EX002 finger term (SCMG_J=34)')
    from .ctl import POINT
    from .ex002_ctl import soft_counts
    dev = yp.device; cvm = ctl.get('cvm', batch['ctl_vm'])
    ft = torch.round(torch.nan_to_num(batch['ctl_t'].float(), -1e3) * 50.).long(); kt = torch.div(ft, CHUNK_FRAMES, rounding_mode='floor')
    ok = ctl['cm'] & (batch['ctl_type'] != POINT) & (batch['ctl_g'] == 0) & (cvm[..., 0] > 0) & (ft >= 0) & (kt < K)
    bi, ji = ok.nonzero(as_tuple=True)
    if len(bi) == 0: return dict(ctl_finger=yp.sum() * 0., ctl_finger_n=torch.zeros((), device=dev))
    kk = kt[bi, ji]; vis = ctl['keep'][bi, kk, ji]; bi, ji, kk = bi[vis], ji[vis], kk[vis]
    if len(bi) == 0: return dict(ctl_finger=yp.sum() * 0., ctl_finger_n=torch.zeros((), device=dev))
    m = bi * K + kk; lf = ft[bi, ji] - (kk - 1) * CHUNK_FRAMES
    sc_ = soft_counts(yp[m, lf], upper)                                                        # [n, 2]
    tg = ctl['cval'][bi, ji, :2].float() * 1.5 + 1.5
    wt = fw[m, lf]
    l = F.smooth_l1_loss(sc_, tg, reduction='none').mean(-1)
    return dict(ctl_finger=(l * wt).sum() / wt.sum().clamp(min=1e-3), ctl_finger_n=torch.tensor(float(len(bi)), device=dev),
                ctl_finger_abs=(sc_.detach() - tg).abs().mean())


def ctl_count_loss(yp, batch, ctl, fw, B, K, thr):
    """yp [M = B K, 2 CF, J] decoded [ctx | target] frames (physical), fw [M, 2 CF]. The target halves of the K chunks of a sample
    form its window estimate; for every kept count token (non-POINT annotation token with count slot, visible to the chunk of its
    span start, whole span supervised): smooth-L1(soft count of the span, given count) ('4 or more': hinge below 4)."""
    from .common import EMBODIMENT
    if EMBODIMENT != 'g1': raise ValueError('StageCfg.w_ctl_count is the G1 count term (SCMG_J=39)')
    from .ctl import POINT, COUNT_MAX, arm_points, count_speed, peak_score
    dev = yp.device; CF = CHUNK_FRAMES; T = K * CF
    cvm = ctl.get('cvm', batch['ctl_vm']); cval = ctl['cval']
    fa = torch.round(torch.nan_to_num(batch['ctl_a'].float(), -1e3) * 50.).long()
    fb = torch.round(torch.nan_to_num(batch['ctl_b'].float(), -1e3) * 50.).long()
    ka = torch.div(fa.clamp(min=0), CF, rounding_mode='floor').clamp(max=K - 1)
    ok = ctl['cm'] & (batch['ctl_type'] != POINT) & (batch['ctl_g'] == 0) & (cvm[..., 0] > 0) & (fa >= 0) & (fb <= T) & (fb - fa >= 2)
    ok = ok & torch.gather(ctl['keep'], 1, ka[:, None, :]).squeeze(1)
    yw = yp.view(B, K, 2 * CF, -1)[:, :, CF:].reshape(B, T, -1)
    fw_ = fw.view(B, K, 2 * CF)[:, :, CF:].reshape(B, T)
    fr = torch.arange(T, device=dev)
    insp = (fr[None, None] >= fa[..., None]) & (fr[None, None] < fb[..., None])                  # [B, Kc, T]
    ok = ok & ((fw_[:, None] > 0) | ~insp).all(-1)
    bi, ji = ok.nonzero(as_tuple=True)
    if len(bi) == 0: return dict(ctl_count=yp.sum() * 0., ctl_count_n=torch.zeros((), device=dev))
    ub, inv = torch.unique(bi, return_inverse=True)
    w = arm_points(yw[ub][..., 10:39])['wrist']                                                  # [nb, T, 2, 3]
    ps = peak_score(count_speed(w), thr, soft=True)                                               # [nb, T, 2]
    m = insp[bi, ji].float()                                                                      # [n, T]
    wd = w.detach()[inv]
    big = torch.where(m[..., None, None] > 0, wd, torch.full_like(wd, -1e3)).amax(1)
    sml = torch.where(m[..., None, None] > 0, wd, torch.full_like(wd, 1e3)).amin(1)
    hand = torch.linalg.norm(big - sml, dim=-1).argmax(-1)                                        # [n]
    soft = (ps[inv, :, :].gather(2, hand[:, None, None].expand(-1, T, 1)).squeeze(-1) * m).sum(-1)
    n = torch.round(cval[bi, ji, 0] * (COUNT_MAX / 2) + COUNT_MAX / 2)
    l = torch.where(n >= COUNT_MAX, F.relu(COUNT_MAX - soft), F.smooth_l1_loss(soft, n, reduction='none'))
    wt = (fw_[bi] * m).sum(-1) / m.sum(-1).clamp(min=1)
    return dict(ctl_count=(l * wt).sum() / wt.sum().clamp(min=1e-3), ctl_count_n=torch.tensor(float(len(bi)), device=dev),
                ctl_count_abs=(soft.detach() - n).abs().mean())


_PHYS = {}
_EX = {}


def phys_reg(y, fw, qm, qs, sc, gt=None):
    """y [M, F, J] decoded [ctx | target] frames (normalised), fw [M, F] frame weights (validity x low-noise trust).
    Returns phys_joint (hardware-limit hinge^2), phys_dyn (|v|,|a|,|j| beyond the robot's limits, relative, hinge^2),
    phys_coll (self-penetration^2 in cm^2 over a random frame / pair subset)."""
    import json, os
    from .common import CPFS, require_ex002
    require_ex002('StageCfg.w_phys (EX002 limits + FK collision proxy)')
    from .data import VIEW
    from .kinematics import Kinematics
    from .safety import limits34
    dev = y.device
    if dev not in _PHYS:
        meta = json.load(open(f'{VIEW}/META.json'))
        L = limits34(meta['lower'], meta['upper'], margin=0.0)
        npz = os.environ.get('SCMG_PROXY', CPFS + '/CKPT/zehan/gesture-scmg/models/sphere_proxy_vox4e.npz')
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
        _PHYS[dev] = dict(lo=t(L['lo']), hi=t(L['hi']), v=t(L['v']), a=t(L['a']), j=t(L['j']), kin=Kinematics(npz, dev))
    P = _PHYS[dev]
    q = y.float() * qs + qm                                                          # rad
    w = fw.float()
    wn = w.sum().clamp(min=1)
    jl = (torch.relu(q - P['hi']) ** 2 + torch.relu(P['lo'] - q) ** 2).sum(-1)
    v = torch.diff(q, dim=1) * 50.; a = torch.diff(v, dim=1) * 50.; jk = torch.diff(a, dim=1) * 50.
    hv = lambda x, lim: (torch.relu(x.abs() / lim - 1.) ** 2).sum(-1)
    dyn = (hv(v, P['v']) * w[:, 1:]).sum() / wn + (hv(a, P['a']) * w[:, 2:]).sum() / wn + 0.1 * (hv(jk, P['j']) * w[:, 3:]).sum() / wn
    out = dict(phys_joint=(jl * w).sum() / wn, phys_dyn=dyn)
    tg = q[:, q.shape[1] // 2:].reshape(-1, q.shape[-1]); tw = w[:, w.shape[1] // 2:].reshape(-1)
    idx = torch.nonzero(tw > 0.5).flatten()
    if len(idx) == 0:
        out['phys_coll'] = y.sum() * 0.; return out
    idx = idx[torch.randperm(len(idx), device=dev)[:sc.phys_frames]]
    kin = P['kin']; npair = kin.pairs.shape[0]
    pi = torch.randint(0, npair, (min(sc.phys_pairs, npair),), device=dev)
    pen = kin.penetration(tg[idx], pi, d_safe=0.005) * 100.                          # cm
    if sc.phys_coll_huber > 0:
        hd = sc.phys_coll_huber
        out['phys_coll'] = torch.where(pen <= hd, pen ** 2, hd * (2 * pen - hd)).sum(-1).mean() * (npair / len(pi))
    else:
        out['phys_coll'] = (pen ** 2).sum(-1).mean() * (npair / len(pi))
    if gt is not None:              # skip diagnostic: same frames / pairs on the GT (no extra random draw)
        with torch.no_grad():
            gq = (gt.float() * qs + qm)[:, q.shape[1] // 2:].reshape(-1, q.shape[-1])
            gp = kin.penetration(gq[idx], pi, d_safe=0.005) * 100.
            pf = pen.detach().max(-1).values; gf = gp.max(-1).values; fpw = q.shape[1] - q.shape[1] // 2
            out.update(_pen_max_cm=float(pf.max()), _gt_pen_max_cm=float(gf.max()), _pen_mean_cm2=float((pen.detach() ** 2).sum(-1).mean()),
                       _gt_pen_mean_cm2=float((gp ** 2).sum(-1).mean()), _pen_argmax_window=float(idx[int(pf.argmax())] // fpw),
                       _gt_pen_frames=float((gf > 0).float().mean()), _pen_frames=float((pf > 0).float().mean()))
    return out
