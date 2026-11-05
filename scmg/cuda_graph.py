"""11-06 opt-in CUDA-graph replay of the Streamer's Local Flow sampling loop (scmg.infer.Streamer(cuda_graph=True)).

Per tick the eager loop launches steps x (1 + CFG) flow_velocity calls of many small kernels; on a batch-1 stream the
GPU waits on the host (A800 DLC: ~16.8 k kernels per tick, ~52 ms of kernels in ~137 ms of wall time). The loop is
captured once (shapes are fixed: L = ctx + CHUNK latents, the local audio window has a fixed width) and replayed with
the tick's tensors copied into static buffers. The captured body is the eager loop line for line (same ops, same order;
the unconditional audio is built inside the graph exactly as eagerly: audio_null.expand_as(aw), mask = awm.clone() with
cfg_null_fix); the per-step time schedule is precomputed on the host side exactly as the eager loop builds it.
Supported: no style token, no state adapter, no MTC memory, no flow text route, joint CFG (cfg_t None)."""
import torch


class FlowGraph:
    def __init__(self, m, steps, cfg_w, cfg_null_fix=True, warmup=3):
        self.m, self.steps, self.cfg_w, self.null_fix, self.warmup = m, steps, cfg_w, cfg_null_fix, warmup
        self.graphs = {}; self.pool = torch.cuda.graph_pool_handle()

    def _body(self, S):
        m, steps, cfg_w = self.m, self.steps, self.cfg_w
        x = S['x']; known = S['known']
        if cfg_w != 1:
            aw_u = m.audio_null.expand_as(S['aw']); awm_u = S['awm'].clone() if self.null_fix else torch.zeros_like(S['awm'])
        for s in range(steps):
            t = S['T'][s]
            v = m.flow_velocity(x, t, known, S['hk'], S['g'], S['aw'], S['awm'], None, None, None, None, ftext=None)
            if cfg_w != 1:
                v_u = m.flow_velocity(x, t, known, S['hk_u'], S['g'], aw_u, awm_u, None, None, None, None, ftext=None)
                v = v_u + cfg_w * (v - v_u)
            x = torch.where(known[..., None], x, x + v / steps)
        S['out'].copy_(x)

    def _capture(self, S):
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup): self._body(S)
        torch.cuda.current_stream().wait_stream(side)
        G = torch.cuda.CUDAGraph()
        with torch.cuda.graph(G, pool=self.pool): self._body(S)
        return G

    @torch.no_grad()
    def __call__(self, x, known, ct, ctx, hk, hk_u, g, aw, awm):
        L = x.shape[1]; dev = x.device
        ts = []
        for s in range(self.steps):                       # = the eager loop's schedule, built the same way
            t = torch.full((1, L), s / self.steps, device=dev); t[known] = 1.
            if ct < 1: t[:, :ctx] = torch.where(known[:, :ctx], torch.full_like(t[:, :ctx], ct), t[:, :ctx])
            ts.append(t)
        T = torch.stack(ts)
        inp = dict(x=x, known=known, T=T, hk=hk, g=g, aw=aw, awm=awm, **({} if hk_u is None else dict(hk_u=hk_u)))
        key = tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(inp.items()))
        if key not in self.graphs:
            S = {k: v.clone() for k, v in inp.items()}; S['out'] = torch.empty_like(x)
            self.graphs[key] = (S, self._capture(S))
        S, G = self.graphs[key]
        for k, v in inp.items(): S[k].copy_(v)
        G.replay()
        return S['out'].clone()
