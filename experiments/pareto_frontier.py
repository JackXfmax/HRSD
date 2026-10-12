"""
Pareto-frontier experiment: HRSD vs the Top-p / temperature baseline family,
compared on the SAME (PPL, degeneration) plane at full scale (100 prompts x 200 tokens).

Why this design (and not "HRSD must hit PPL <= 3.0 in isolation"):
  self-PPL is minimised by greedy decoding *by construction*, so ANY stochastic
  decoder pays PPL for diversity. The scientifically meaningful question is not
  "does HRSD beat greedy on PPL" (it cannot) but

      on the (PPL, rep-4) plane, does the HRSD family dominate the
      Top-p / temperature family?  i.e. at matched PPL, less repetition?

  Arm A sweeps the baseline family (Top-p 0.9 with temperature T), which traces a
  curve from (PPL 1.77, rep4>0 = 37) upwards/rightwards.
  Arm B sweeps the HRSD family over its two real levers, found by falsification:
      K    -- candidate width  (Lemma: Delta(K) is s-weighted average of
              log(p_(1)/p_(i)), hence monotone nondecreasing in K, Delta(K) <= log(p1/pK))
      eta  -- Stage-4 rank-penalty strength (controls P(rank 0))
  phi shape (k1,k2,lambda) and Stage-1 subset size were both shown NOT to be levers.

Everything is written incrementally so a long run can be inspected at any time.
"""
import os
import sys
import json
import math
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import hrds_experiment_qwen as H


# ---------------------------------------------------------------- config
ARM_A = [
    {"arm": "A", "kind": "topp_T", "top_p": 0.9, "temp": 1.0},
    {"arm": "A", "kind": "topp_T", "top_p": 0.9, "temp": 1.2},
    {"arm": "A", "kind": "topp_T", "top_p": 0.9, "temp": 1.4},
    {"arm": "A", "kind": "topp_T", "top_p": 0.9, "temp": 1.6},
    {"arm": "A", "kind": "topp_T", "top_p": 0.9, "temp": 1.9},
]

ARM_B = [
    {"arm": "B", "kind": "hrds", "k1": 2.0, "k2": 1.0, "lam": 1.0, "eta": 0.5, "kmax": 2},
    {"arm": "B", "kind": "hrds", "k1": 2.0, "k2": 1.0, "lam": 1.0, "eta": 0.2, "kmax": 2},
    {"arm": "B", "kind": "hrds", "k1": 2.0, "k2": 1.0, "lam": 1.0, "eta": 0.0, "kmax": 2},
    {"arm": "B", "kind": "hrds", "k1": 2.0, "k2": 1.0, "lam": 1.0, "eta": 0.5, "kmax": 3},
    {"arm": "B", "kind": "hrds", "k1": 2.0, "k2": 1.0, "lam": 1.0, "eta": 0.2, "kmax": 3},
    {"arm": "B", "kind": "hrds", "k1": 2.0, "k2": 1.0, "lam": 1.0, "eta": 0.0, "kmax": 3},
    {"arm": "B", "kind": "hrds", "k1": 2.0, "k2": 1.0, "lam": 1.0, "eta": 0.2, "kmax": 5},
    {"arm": "B", "kind": "hrds", "k1": 5.0, "k2": 1.0, "lam": 0.7, "eta": 0.2, "kmax": 2},
]


def build_prompt_data(tok, prompts, device):
    data = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True,
                                      return_tensors="pt").to(device)
        if ids.shape[1] > H.PROMPT_MAX_TOKENS:
            ids = ids[:, :H.PROMPT_MAX_TOKENS]
        data.append((ids, ids.shape[1]))
    return data


def eval_config(model, tok, prompt_data, cfg, max_new, device):
    kind = cfg["kind"]
    if kind == "topp_T":
        H.TOP_P = float(cfg["top_p"])
        H.TEMPERATURE = float(cfg["temp"])
        method = "topp"
    else:
        H.TOP_P = 0.9
        H.TEMPERATURE = 1.0
        H.K1 = float(cfg["k1"])
        H.K2 = float(cfg["k2"])
        H.LAM = float(cfg["lam"])
        H.ETA = float(cfg["eta"])
        H.KMAX = int(cfg["kmax"])
        method = "hrds"

    H.RUNTIME["max_new_tokens"] = max_new
    gen = torch.Generator(device=device).manual_seed(H.SEED)
    texts, ppls = [], []
    t0 = time.time()

    for ids, p_len in prompt_data:
        g = None
        for attempt in range(3):
            try:
                g, _lat = H.generate_sequence(model, ids, method, gen)
                break
            except Exception as e:
                print(f"    retry {attempt+1}: {type(e).__name__}: {e}", flush=True)
                time.sleep(5)
        if g is None:
            continue
        full = ids[0].tolist() + g
        p = H.compute_ppl(model, full, p_len)
        if not math.isnan(p):
            ppls.append(p)
        texts.append(tok.decode(g, skip_special_tokens=True))

    per_r4 = []
    for t in texts:
        gs = H.ngrams(t, 4)
        per_r4.append(1.0 - len(set(gs)) / len(gs) if gs else 0.0)
    return {
        "ppl": sum(ppls) / len(ppls) if ppls else float("nan"),
        "rep4": H.rep_n(texts, 4),
        "distinct3": H.distinct_n(texts, 3),
        "n_rep4_pos": sum(1 for x in per_r4 if x > 0),
        "n_evaluated": len(texts),
        "secs": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--arm", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="pareto_frontier.json")
    args = ap.parse_args()

    n_prompts = 3 if args.smoke else args.n_prompts
    max_new = 30 if args.smoke else args.max_new_tokens
    device = H.DEVICE
    print(f"Device: {device}; prompts={n_prompts}; max_new={max_new}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(H.MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        H.MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    print("Model loaded.", flush=True)

    prompts = H.fallback_prompts(n_prompts)
    prompt_data = build_prompt_data(tok, prompts, device)

    cfgs = []
    if "A" in args.arm:
        cfgs += ARM_A
    if "B" in args.arm:
        cfgs += ARM_B

    done = []
    for cfg in cfgs:
        r = eval_config(model, tok, prompt_data, cfg, max_new, device)
        rec = {**cfg, **r}
        done.append(rec)
        tag = (f"topp p={cfg['top_p']} T={cfg['temp']}" if cfg["kind"] == "topp_T"
               else f"hrds k1={cfg['k1']} eta={cfg['eta']} K={cfg['kmax']}")
        print(f"[{cfg['arm']}] {tag:<26} | PPL {r['ppl']:7.3f} | rep4 {r['rep4']:.4f} "
              f"| d3 {r['distinct3']:.4f} | rep4>0 {r['n_rep4_pos']:3d}/{r['n_evaluated']} "
              f"| {r['secs']:.0f}s", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"config": {"n_prompts": n_prompts, "max_new": max_new},
                       "results": done}, f, ensure_ascii=False, indent=2)

    print("\n=== PARETO TABLE (sorted by PPL) ===")
    print(f"{'arm':>4}{'config':>28}{'PPL':>9}{'rep4>0':>9}{'d3':>9}")
    for d in sorted(done, key=lambda x: x["ppl"]):
        tag = (f"topp p={d['top_p']} T={d['temp']}" if d["kind"] == "topp_T"
               else f"hrds k1={d['k1']} eta={d['eta']} K={d['kmax']}")
        print(f"{d['arm']:>4}{tag:>28}{d['ppl']:>9.3f}"
              f"{d['n_rep4_pos']:>7d}/{d['n_evaluated']:<3d}{d['distinct3']:>9.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
