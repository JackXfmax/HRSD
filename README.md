# HRSD — Hierarchical Reshaping Sampling Decoding

**HRSD** is a 4-stage, *layered reshaping* decoding method for autoregressive
language models that suppresses degeneration (repetition) while paying a small,
bounded, and *budgetable* self-perplexity (PPL) cost — unlike unstructured
knobs (top-p, temperature, typical sampling) which must let PPL explode to push
repetition down, or repetition penalties (RP) which sit on the same
self-PPL frontier and are therefore Pareto-dominated.

This repository contains:

- `hrsd/` — the **clean, model-agnostic algorithm core** (pure PyTorch, no
  model loading, no framework hacks). This is the citable implementation.
- `experiments/` — the **full reproduction pipeline**: model loading, the
  metrics (distinct-n, rep-n, self-PPL), paired-bootstrap significance tests,
  all baselines (top-p, typical, RP, greedy, contrastive search), and the
  scale / cross-model / Pareto sweeps that back every figure and claim.

---

## The algorithm

Given next-token logits `z` (1-D tensor over the vocabulary), HRSD samples one
token through four stages:

| Stage | Name | What it does |
|-------|------|--------------|
| 1 | Global power-subset | `p = softmax(z); w = p**LAM`; keep the top-`|V|/2` entries by `log(w) + Gumbel`. |
| 2 | Arctan unimodal rescoring | `phi = arctan(K1·p̃)·arctan(K2·(1−p̃))`; optionally reshape via `phi_theta = p̃^(1−θ)·phi^θ` (Lemma 12.3). |
| 3 | Dynamic candidate set | keep the `K = clip(α·|valid|, KMIN, KMAX)` valid (`p̃ ≥ TAU`) tokens with largest score `s`. |
| 4 | Temporal rank soft-penalty | `q_i = s_i·exp(−ETA·counter_i)`; draw; increment `counter[y]`, decay others by `GAMMA`. |

Two ablations are included: **drop-Stage-2** (score by raw subset probability)
and **drop-Stage-4** (plain softmax over candidates).

The reported **HRSD-primary** configuration is
`HRSDConfig.primary()` → `LAM=1.0, K1=2.0, K2=1.0, THETA=1.0, ETA=0.2, KMAX=2, ALPHA=0.3, TAU=0.001, GAMMA=0.9`.

---

## Install

```bash
pip install -r requirements.txt
```

`hrsd/` has **no** dependency beyond PyTorch. The `experiments/` scripts need
`transformers`, `numpy`, and (for the dataset fallback path) `datasets`.

---

## Quick start (core only)

```python
import torch
from hrsd import HRSDConfig, HRSDSampler

cfg = HRSDConfig.primary()
sampler = HRSDSampler(cfg, device="cuda")
gen = torch.Generator(device="cuda").manual_seed(42)

counter = sampler.new_counter()          # reset once per prompt
for t in range(max_new_tokens):
    logits_t = model(...)                # next-token logits, shape [|V|]
    tok = sampler.step(logits_t, counter=counter, generator=gen)
    ...
```

---

## Reproduction (experiments/)

> All experiment scripts were run on a single local instruction model
> (Qwen2.5-7B-Instruct) because the lab host cannot download gpt2/gpt2-large
> (firewall MITM). PPL is therefore **self-PPL** — the generator scores its own
> continuation only, excluding the prompt. `hrds_experiment.py` is the gpt2
> reference version of the same math.

1. Point `MODEL_PATH` (in `experiments/hrds_experiment_qwen.py`) at a local
   causal-LM checkpoint, then:
   ```bash
   cd experiments
   python hrds_experiment_qwen.py --smoke      # 3-prompt sanity run
   python hrds_experiment_qwen.py              # full 100-prompt run -> results.json
   ```
2. The remaining scripts each target one claim / figure:

| Script | Produces | Claim / figure |
|--------|----------|----------------|
| `hrds_experiment_qwen.py` | `results.json` | main table: HRSD vs top-p vs typical + 2 ablations |
| `robustness_significance.py` | `robustness_significance.json` | paired bootstrap vs top-p; fairness baselines |
| `paired_hrsd_rp.py` | `paired_hrsd_rp.json` | **HRSD vs RP** (paired, same seed) + semantic topic-alignment |
| `rp_sweep.py` | `rp_sweep.json` | RP penalties 1.1–1.4 on the same PPL band as HRSD |
| `pareto_frontier.py` | `pareto_frontier.json` | (PPL, rep-4) Pareto frontier, HRSD-family vs top-p/temp family |
| `scale_sweep.py` | `scale_sweep_{model}.json` | PPL/rep4 across 0.5B–7B (scale robustness) |
| `cross_model_generalization.py` | `cross_model_generalization.json` | HRSD vs top-p/RP on a 2nd model (Llama) |
| `contrastive_search_baseline.py` | `*.json` | contrastive search as a 2nd structured baseline |
| `greedy_selfppl.py` | `*.json` | greedy self-PPL (denominator for Δ-PPL) |
| `topp_texts.py` | `topp_T10_texts.json` | top-p generated texts (for side-by-side samples) |
| `extrinsic_gpt2_ppl.py` | `*.json` | **external** PPL via a separate LM (extrinsic quality signal) |

All scripts accept `--smoke` for a 3-prompt quick check and `--n-prompts` /
`--max-new-tokens` for scale.

### Headline numbers (Qwen2.5-7B-Instruct, 100 prompts × 200 tokens, seed 42)

| Method | self-PPL | rep-4 | distinct-3 |
|--------|---------:|------:|-----------:|
| HRSD (ours) | **3.458** | **0.0012** | **0.9959** |
| Top-p (T=1.0, p=0.9) | 1.772 | 0.0127 | 0.9791 |
| Typical (m=0.95) | 1.951 | 0.0139 | 0.9761 |
| RP=1.2 + top-p 0.9 | 3.384 | 0.0003 | 0.9986 |
| greedy | 1.414 | 0.0160 | — |

HRSD Pareto-dominates every baseline that has not reached the self-PPL boundary
(top-p, typical, greedy); only RP is complementary (mutually non-dominated).

---

## Repository layout

```
HRSD/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── hrsd/                     # clean algorithm core (pure PyTorch)
│   ├── __init__.py
│   └── sampler.py            # 4 stages + baselines + HRSDSampler
└── experiments/              # full reproduction pipeline (see table above)
    ├── hrds_experiment.py           # gpt2 reference version
    ├── hrds_experiment_qwen.py      # runnable Qwen version (entry point)
    ├── robustness_significance.py   # shared eval lib: metrics + paired bootstrap + RP
    ├── paired_hrsd_rp.py
    ├── scale_sweep.py
    ├── rp_sweep.py
    ├── pareto_frontier.py
    ├── cross_model_generalization.py
    ├── contrastive_search_baseline.py
    ├── greedy_selfppl.py
    ├── topp_texts.py
    └── extrinsic_gpt2_ppl.py
```

---

## Citation

```bibtex
@article{hrsd2026,
  title   = {Hierarchical Reshaping Sampling Decoding},
  author  = {xufei},
  journal = {in submission},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
