"""
HRSD — Hierarchical Reshaping Sampling Decoding.

Clean, model-agnostic core of the HRSD decoding algorithm. The runnable
experiment pipeline (model loading, metrics, paired-bootstrap significance, all
baselines and sweeps) lives under ``experiments/``.

Quick start
-----------
    from hrsd import HRSDConfig, HRSDSampler
    cfg = HRSDConfig.primary()
    sampler = HRSDSampler(cfg, device="cuda")
    counter = sampler.new_counter()
    tok = sampler.step(logits, counter=counter, generator=gen)
"""

from .sampler import (
    HRSDConfig,
    HRSDSampler,
    hrds_step,
    hrds_no_stage2_step,
    hrds_no_stage4_step,
    top_p_step,
    typical_step,
    stage1_subset,
    stage2_rescore,
    stage3_candidates,
    stage4_select,
)

__all__ = [
    "HRSDConfig",
    "HRSDSampler",
    "hrds_step",
    "hrds_no_stage2_step",
    "hrds_no_stage4_step",
    "top_p_step",
    "typical_step",
    "stage1_subset",
    "stage2_rescore",
    "stage3_candidates",
    "stage4_select",
]

__version__ = "0.1.0"
