"""
HRSD — Hierarchical Repetition Suppression Decoding (core algorithm)
================================================================

HRSD is a 4-stage, *layered reshaping* decoding method for autoregressive
language models. Given the next-token logits ``z`` (a 1-D float tensor over the
vocabulary), it returns a single sampled token index.

The four stages are applied in sequence:

  Stage 1 — Global power-subset sampling.
      p   = softmax(z);  w = p ** LAM
      g   = log(w + 1e-12) + Gumbel(size=|V|)          # top-k keeps half the vocab
      S_half = topk(g, k=|V| // 2)

  Stage 2 — Arctan unimodal rescoring.
      p_tilde = softmax(z[S_half])
      phi = arctan(K1 * p_tilde) * arctan(K2 * (1 - p_tilde))
      s   = phi                                      # or, for THETA != 1.0,
      s   = p_tilde**(1 - THETA) * phi**THETA        # (Lemma 12.3 reshaping)

  Stage 3 — Dynamic candidate set.
      valid  = {i : p_tilde_i >= TAU}
      K      = clip(alpha * |valid|, KMIN, KMAX)
      S_topK = the K indices of ``valid`` with largest ``s``

  Stage 4 — Temporal rank soft-penalty.
      q_i     = s_i * exp(-ETA * counter_i)           # counter_i = times rank i was picked
      y       ~ Categorical(q / sum(q))
      counter[y] += 1;  counter[others] *= GAMMA

Two ablations are provided by the experiment pipeline (see ``hrds_experiment_qwen.py``):

  * ``hrds_no_stage2`` — Stage 2 is skipped; candidates are scored by the raw
    subset probability ``p_tilde``.
  * ``hrds_no_stage4`` — Stage 4 is skipped; the candidate distribution is the
    plain softmax ``s`` (no temporal rank penalty).

This module is **model-agnostic** and depends only on PyTorch. The math is
byte-for-byte identical to the implementations embedded in the full experiment
pipeline under ``experiments/``; this file is the citable, importable core.

Example
-------
    >>> import torch
    >>> from hrsd import HRSDConfig, HRSDSampler
    >>> cfg = HRSDConfig.primary()           # the reported "HRSD-primary" config
    >>> sampler = HRSDSampler(cfg, device="cpu")
    >>> logits = torch.randn(10)             # next-token logits over a 10-token vocab
    >>> token_id = sampler.step(logits, generator=torch.Generator().manual_seed(0))
"""

from dataclasses import dataclass

import torch


@dataclass
class HRSDConfig:
    """Hyper-parameters for the HRSD sampler.

    Defaults follow the spec in ``hrds_experiment.py``. The configuration used
    for all reported results (the "HRSD-primary" config) is available via the
    :meth:`primary` classmethod.
    """

    lam: float = 0.7     # Stage-1 power exponent
    k1: float = 5.0      # Stage-2 arctan scale for p_tilde
    k2: float = 1.0      # Stage-2 arctan scale for (1 - p_tilde)
    theta: float = 1.0   # Stage-2 reshaping exponent (Lemma 12.3)
    tau: float = 0.001   # Stage-3 valid-threshold on p_tilde
    alpha: float = 0.3   # Stage-3 candidate fraction of the valid set
    kmin: int = 3        # Stage-3 minimum number of candidates
    eta: float = 0.5      # Stage-4 rank-penalty strength
    gamma: float = 0.9    # Stage-4 decay applied to unchosen ranks
    kmax: int = 20        # Stage-4 counter size AND hard cap on K

    @classmethod
    def primary(cls) -> "HRSDConfig":
        """The configuration used for all reported HRSD-primary results."""
        return cls(lam=1.0, k1=2.0, k2=1.0, theta=1.0, eta=0.2, gamma=0.9, kmax=2,
                   tau=0.001, alpha=0.3)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _sample_gumbel(size: int, generator, device):
    u = torch.rand(size, generator=generator, device=device).clamp(min=1e-10, max=1.0 - 1e-10)
    return -torch.log(-torch.log(u))


def stage1_subset(z: torch.Tensor, cfg: HRSDConfig, generator) -> torch.Tensor:
    """Stage 1: keep the half of the vocabulary with the largest
    ``log(p**LAM) + Gumbel`` score."""
    p = torch.softmax(z, dim=-1)
    w = p ** cfg.lam
    g = torch.log(w + 1e-12) + _sample_gumbel(z.shape[-1], generator, z.device)
    k = z.shape[-1] // 2
    _, idx = torch.topk(g, k)
    return idx  # S_half, shape [k]


def stage2_rescore(z: torch.Tensor, s_half: torch.Tensor, cfg: HRSDConfig):
    """Stage 2: arctan unimodal rescoring of the subset distribution."""
    z_sub = z[s_half]
    p_tilde = torch.softmax(z_sub, dim=-1)
    # float32 keeps the arctan product well-defined; a 1e-9 floor would
    # underflow to 0.0 in fp16 and break the Stage-4 multinomial.
    phi = (torch.arctan(cfg.k1 * p_tilde.float())
           * torch.arctan(cfg.k2 * (1.0 - p_tilde.float())))
    if cfg.theta == 1.0:
        s = phi
    else:
        # phi_theta(p) ~ p**(1-theta) * phi(p)**theta
        # (Lemma 12.3: Delta is non-decreasing in theta)
        s = (p_tilde.float() ** (1.0 - cfg.theta)) * (phi ** cfg.theta)
    s = s.clamp_min(1e-9)
    return s, p_tilde


def stage2_rescore_none(z: torch.Tensor, s_half: torch.Tensor, cfg: HRSDConfig):
    """Ablation (drop Stage 2): score directly by subset probability."""
    z_sub = z[s_half]
    p_tilde = torch.softmax(z_sub, dim=-1)
    return p_tilde.float(), p_tilde


def stage3_candidates(p_tilde: torch.Tensor, s: torch.Tensor, cfg: HRSDConfig) -> torch.Tensor:
    """Stage 3: dynamic candidate set — the ``K`` valid tokens with largest ``s``."""
    valid_mask = p_tilde >= cfg.tau
    valid_idx = torch.nonzero(valid_mask).flatten()
    n = valid_idx.numel()
    if n == 0:  # numerical safety
        valid_idx = torch.topk(p_tilde, 1).indices
        n = 1
    k = min(n, max(int(cfg.alpha * n), cfg.kmin), cfg.kmax)
    k = max(k, 1)
    s_valid = s[valid_idx]
    topk = torch.topk(s_valid, k)
    return valid_idx[topk.indices]  # S_topK


def stage4_select(s_topk: torch.Tensor, counter: torch.Tensor, cfg: HRSDConfig,
                  generator, degen_counter=None) -> int:
    """Stage 4: temporal rank soft-penalty over the candidate scores."""
    k = s_topk.numel()
    q = s_topk * torch.exp(-cfg.eta * counter[:k])
    q_sum = q.sum()
    if not torch.isfinite(q_sum) or q_sum <= 0:
        q = torch.ones_like(s_topk)  # degenerate rescue: uniform over candidates
        if degen_counter is not None:
            degen_counter[0] += 1
    q = q / (q.sum() + 1e-12)
    y = torch.multinomial(q, 1, generator=generator).item()
    counter[y] += 1
    for i in range(cfg.kmax):
        if i != y:
            counter[i] *= cfg.gamma
    return y


def stage4_select_no_penalty(s_topk: torch.Tensor, generator, degen_counter=None) -> int:
    """Ablation (drop Stage 4): plain softmax over candidates."""
    q = s_topk / (s_topk.sum() + 1e-12)
    if not torch.isfinite(q).all() or (q < 0).any():
        q = torch.ones_like(s_topk)  # degenerate rescue
        if degen_counter is not None:
            degen_counter[0] += 1
    q = q / (q.sum() + 1e-12)
    return torch.multinomial(q, 1, generator=generator).item()


# --------------------------------------------------------------------------- #
# baselines (kept for completeness / fair comparison)
# --------------------------------------------------------------------------- #
def top_p_step(z: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0,
              generator=None) -> int:
    """Nucleus (top-p) sampling."""
    if temperature != 1.0:
        z = z / temperature
    p = torch.softmax(z, dim=-1)
    sorted_p, sorted_idx = torch.sort(p, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    keep = cumsum <= top_p
    if not keep.any():
        keep[0] = True
    probs = sorted_p[keep]
    probs = probs / probs.sum()
    idx = sorted_idx[keep]
    return idx[torch.multinomial(probs, 1, generator=generator).item()].item()


def typical_step(z: torch.Tensor, typical_p: float = 0.9, temperature: float = 1.0,
                 generator=None) -> int:
    """Typical sampling (Megatel et al., 2021) — matches HF TypicalLogitsWarper."""
    if temperature != 1.0:
        z = z / temperature
    x = torch.log_softmax(z, dim=-1)
    x_exp = torch.exp(x)
    entropy = torch.sum(-x_exp * x)
    surprise = torch.abs(x + entropy)  # | -log p - H |
    sorted_s, sorted_idx = torch.sort(surprise, descending=False)
    x_sorted = x[sorted_idx]
    x_exp_sorted = x_exp[sorted_idx]
    cumsum = torch.cumsum(x_exp_sorted, dim=-1)
    keep = cumsum < typical_p
    if not keep.any():
        keep[0] = True
    probs = x_exp_sorted[keep]
    probs = probs / probs.sum()
    chosen = torch.multinomial(probs, 1, generator=generator).item()
    return sorted_idx[keep][chosen].item()


# --------------------------------------------------------------------------- #
# full HRSD steps
# --------------------------------------------------------------------------- #
def hrds_step(z: torch.Tensor, counter: torch.Tensor, cfg: HRSDConfig,
              generator, degen_counter=None) -> int:
    s_half = stage1_subset(z, cfg, generator)
    s, p_tilde = stage2_rescore(z, s_half, cfg)
    s_topk_idx = stage3_candidates(p_tilde, s, cfg)
    s_topk = s[s_topk_idx]
    y = stage4_select(s_topk, counter, cfg, generator, degen_counter)
    return int(s_half[s_topk_idx[y]].item())


def hrds_no_stage2_step(z: torch.Tensor, counter: torch.Tensor, cfg: HRSDConfig,
                         generator, degen_counter=None) -> int:
    s_half = stage1_subset(z, cfg, generator)
    s, p_tilde = stage2_rescore_none(z, s_half, cfg)
    s_topk_idx = stage3_candidates(p_tilde, s, cfg)
    s_topk = s[s_topk_idx]
    y = stage4_select(s_topk, counter, cfg, generator, degen_counter)
    return int(s_half[s_topk_idx[y]].item())


def hrds_no_stage4_step(z: torch.Tensor, cfg: HRSDConfig,
                         generator, degen_counter=None) -> int:
    s_half = stage1_subset(z, cfg, generator)
    s, p_tilde = stage2_rescore(z, s_half, cfg)
    s_topk_idx = stage3_candidates(p_tilde, s, cfg)
    s_topk = s[s_topk_idx]
    y = stage4_select_no_penalty(s_topk, generator, degen_counter)
    return int(s_half[s_topk_idx[y]].item())


class HRSDSampler:
    """Stateful wrapper around the HRSD steps.

    Holds the temporal rank ``counter`` (reset per prompt) so callers can drive
    a full generation loop the same way the experiment pipeline does::

        sampler = HRSDSampler(HRSDConfig.primary(), device="cuda")
        counter = sampler.new_counter()
        for t in range(max_new_tokens):
            tok = sampler.step(logits_t, counter=counter, generator=gen)
            ...
    """

    def __init__(self, cfg: HRSDConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self._degen = [0]

    def new_counter(self) -> torch.Tensor:
        """Fresh rank counter (call once per prompt)."""
        return torch.zeros(self.cfg.kmax, device=self.device)

    def step(self, z: torch.Tensor, counter: torch.Tensor, generator,
             ablation: str = "full") -> int:
        if ablation == "full":
            return hrds_step(z, counter, self.cfg, generator, self._degen)
        if ablation == "no_stage2":
            return hrds_no_stage2_step(z, counter, self.cfg, generator, self._degen)
        if ablation == "no_stage4":
            return hrds_no_stage4_step(z, self.cfg, generator, self._degen)
        raise ValueError(f"unknown ablation: {ablation!r}")

    @property
    def degenerate_rescues(self) -> int:
        return self._degen[0]
