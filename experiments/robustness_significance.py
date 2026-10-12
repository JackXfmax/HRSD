#!/usr/bin/env python
# robustness_significance.py
# Part 1: paired bootstrap significance (HRSD primary config vs Top-p T=1.0) on
#         rep-4 / distinct-3 / PPL  -- per user's hard rule: no "significant"
#         claim without bootstrap.
# Part 2: fairness baselines at the SAME (PPL, rep-4) plane so we don't only beat
#         temperature: top-p variant p, typical sampling, repetition penalty.
#
# Reuses hrds_experiment_qwen.py primitives (model load, compute_ppl, distinct_n,
# rep_n, topp/typical/hrds steps). Repetition penalty is implemented inline with
# its own KV-cache loop (the shared module has no RP variant).

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- import the shared experiment module (constants + helpers) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hrds_experiment_qwen as H

RNG_SEED = 42
SEED = 42


def build_prompts(n, max_new):
    H.RUNTIME["max_new_tokens"] = max_new
    from transformers import AutoTokenizer
    tok = H.tok if hasattr(H, "tok") else AutoTokenizer.from_pretrained(
        H.MODEL_PATH, local_files_only=True)
    H.tok = tok
    prompts = H.load_prompts(n)
    data = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        if ids.shape[1] > H.PROMPT_MAX_TOKENS:
            ids = ids[:, :H.PROMPT_MAX_TOKENS]
        data.append((ids, int(ids.shape[1])))
    return data


def per_prompt_metrics(model, tok, input_ids, prompt_len, gen_ids):
    full = input_ids[0].tolist() + gen_ids
    t = tok.decode(gen_ids, skip_special_tokens=True)
    d3 = H.distinct_n([t], 3)
    r4 = H.rep_n([t], 4)
    p = H.compute_ppl(model, full, prompt_len)
    ppl = None if (p is None or math.isnan(p)) else float(p)
    return {"distinct3": float(d3), "rep4": float(r4), "ppl": ppl,
            "gen_len": len(gen_ids), "text": t}


def gen_builtin(model, data, method, overrides, gen_rng, max_new):
    # apply global overrides (H.K1, H.TOP_P, H.TYPICAL_P, H.TEMPERATURE, ...)
    saved = {}
    for k, v in overrides.items():
        saved[k] = getattr(H, k)
        setattr(H, k, v)
    gens = []
    for (ids, plen) in data:
        g, _ = H.generate_sequence(model, ids, method, gen_rng)
        gens.append(g)
    for k, v in saved.items():
        setattr(H, k, v)
    return gens


def gen_rp(model, data, penalty, top_p, gen_rng, max_new):
    """Repetition-penalty + nucleus sampling, own KV-cache loop (HF-style RP)."""
    gens = []
    for (input_ids, prompt_len) in data:
        out = model(input_ids=input_ids, use_cache=True)
        logits = out.logits[0, -1]
        past = out.past_key_values
        gen_ids = []
        hist = []
        for step in range(max_new):
            lp = logits.float().clone()
            if penalty != 1.0 and hist:
                seen = set(hist)
                for t in seen:
                    if lp[t] > 0:
                        lp[t] = lp[t] / penalty
                    else:
                        lp[t] = lp[t] * penalty
            p = torch.softmax(lp, dim=-1)
            sorted_p, sorted_idx = torch.sort(p, descending=True)
            cum = torch.cumsum(sorted_p, dim=0)
            keep = cum <= top_p
            if not keep.any():
                keep[0] = True
            probs = sorted_p[keep]
            probs = probs / probs.sum()
            idx = sorted_idx[keep]
            tok_i = idx[torch.multinomial(probs, 1, generator=gen_rng).item()].item()
            gen_ids.append(tok_i)
            hist.append(tok_i)
            if step == max_new - 1:
                break
            nt = torch.tensor([[tok_i]], device=DEVICE)
            pos = torch.tensor([[prompt_len + step]], device=DEVICE, dtype=torch.long)
            out = model(input_ids=nt, position_ids=pos,
                        past_key_values=past, use_cache=True)
            logits = out.logits[0, -1]
            past = out.past_key_values
        gens.append(gen_ids)
    return gens


def evaluate(model, tok, data, config, gen_rng, max_new):
    method = config["method"]
    if method == "hrds":
        overrides = {"K1": 2.0, "K2": 1.0, "LAM": 1.0, "ETA": 0.2,
                     "THETA": 1.0, "KMAX": 2, "ALPHA": 0.3}
        gens = gen_builtin(model, data, "hrds", overrides, gen_rng, max_new)
    elif method == "topp":
        overrides = {"TOP_P": config["p"], "TEMPERATURE": 1.0}
        gens = gen_builtin(model, data, "topp", overrides, gen_rng, max_new)
    elif method == "typical":
        overrides = {"TYPICAL_P": config["p"], "TEMPERATURE": 1.0}
        gens = gen_builtin(model, data, "typical", overrides, gen_rng, max_new)
    elif method == "rp":
        gens = gen_rp(model, data, config["penalty"], config.get("top_p", 0.9),
                      gen_rng, max_new)
    else:
        raise ValueError(method)
    pp = [per_prompt_metrics(model, tok, data[i][0], data[i][1], gens[i])
          for i in range(len(data))]
    ppls = [x["ppl"] for x in pp if x["ppl"] is not None]
    agg = {
        "distinct3": float(np.mean([x["distinct3"] for x in pp])),
        "rep4": float(np.mean([x["rep4"] for x in pp])),
        "ppl": float(np.mean(ppls)) if ppls else float("nan"),
        "n_rep4_pos": int(sum(1 for x in pp if x["rep4"] > 0)),
        "n_evaluated": len(pp),
        "mean_gen_len": float(np.mean([x["gen_len"] for x in pp])),
    }
    return agg, pp


def paired_bootstrap(hrds_pp, base_pp, n_boot=2000, seed=12345):
    hr = np.array([x["rep4"] for x in hrds_pp], dtype=float)
    br = np.array([x["rep4"] for x in base_pp], dtype=float)
    hd = np.array([x["distinct3"] for x in hrds_pp], dtype=float)
    bd = np.array([x["distinct3"] for x in base_pp], dtype=float)
    hp = np.array([x["ppl"] for x in hrds_pp if x["ppl"] is not None], dtype=float)
    bp = np.array([x["ppl"] for x in base_pp if x["ppl"] is not None], dtype=float)
    rng = np.random.default_rng(seed)
    N = len(hr)
    idx = rng.integers(0, N, size=(n_boot, N))
    out = {}
    for name, a, b, direction in [
        ("rep4", hr, br, "lower"),       # HRSD suppresses repetition
        ("distinct3", hd, bd, "higher"),  # HRSD more diverse
        ("ppl", hp, bp, "higher"),       # HRSD pays PPL (reported honestly)
    ]:
        if len(a) < 2 or len(b) < 2:
            out[name] = {"obs": None, "ci": None, "p_one": None, "p_two": None}
            continue
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        obs = float(a.mean() - b.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        if direction == "lower":
            p_one = float((diffs >= 0).mean())   # P(HRSD not lower)
        else:
            p_one = float((diffs <= 0).mean())   # P(HRSD not higher)
        p_two = float(2 * min(p_one, 1 - p_one))
        out[name] = {"obs": obs, "ci": [float(lo), float(hi)],
                     "p_one": p_one, "p_two": p_two, "direction": direction}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="robustness_significance.json")
    args = ap.parse_args()
    n = 3 if args.smoke else args.n_prompts
    max_new = 30 if args.smoke else args.max_new_tokens

    print(f"Device: {DEVICE}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(H.MODEL_PATH, local_files_only=True)
    H.tok = tok
    model = AutoModelForCausalLM.from_pretrained(
        H.MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to(DEVICE).eval()
    print("Model loaded.")

    data = build_prompts(n, max_new)

    # ---- Part 1: bootstrap (HRSD primary vs Top-p T=1.0) ----
    print("\n=== Part 1: bootstrap (HRSD primary vs Top-p T=1.0) ===")
    hrds_agg, hrds_pp = evaluate(model, tok, data, {"method": "hrds"},
                                 torch.Generator(device=DEVICE).manual_seed(SEED), max_new)
    base_agg, base_pp = evaluate(model, tok, data, {"method": "topp", "p": 0.9},
                                 torch.Generator(device=DEVICE).manual_seed(SEED), max_new)
    boot = paired_bootstrap(hrds_pp, base_pp, n_boot=2000)
    print(f"  HRSD primary : PPL={hrds_agg['ppl']:.3f} rep4={hrds_agg['rep4']:.4f} "
          f"rep4>0={hrds_agg['n_rep4_pos']}/{n} d3={hrds_agg['distinct3']:.4f}")
    print(f"  Top-p T=1.0  : PPL={base_agg['ppl']:.3f} rep4={base_agg['rep4']:.4f} "
          f"rep4>0={base_agg['n_rep4_pos']}/{n} d3={base_agg['distinct3']:.4f}")
    for k, v in boot.items():
        print(f"    {k:10s} obs={v['obs']:.4f} 95%CI={v['ci']} p_one={v['p_one']:.4f} p_two={v['p_two']:.4f}")

    # ---- Part 2: fairness baselines (equal-PPL plane) ----
    print("\n=== Part 2: fairness baselines ===")
    fairness = []
    configs = [
        {"method": "topp", "p": 0.85, "tag": "top-p p=0.85"},
        {"method": "topp", "p": 0.95, "tag": "top-p p=0.95"},
        {"method": "typical", "p": 0.90, "tag": "typical m=0.90"},
        {"method": "typical", "p": 0.95, "tag": "typical m=0.95"},
        {"method": "rp", "penalty": 1.5, "top_p": 0.9, "tag": "RP=1.5 + top-p0.9"},
        {"method": "rp", "penalty": 2.0, "top_p": 0.9, "tag": "RP=2.0 + top-p0.9"},
    ]
    for c in configs:
        agg, _ = evaluate(model, tok, data, c,
                          torch.Generator(device=DEVICE).manual_seed(SEED), max_new)
        rec = {"tag": c["tag"], "method": c["method"], **agg}
        fairness.append(rec)
        print(f"  {c['tag']:22s} PPL={agg['ppl']:.3f} rep4={agg['rep4']:.4f} "
              f"rep4>0={agg['n_rep4_pos']}/{n} d3={agg['distinct3']:.4f}")

    result = {
        "n_prompts": n, "max_new": max_new,
        "hrds_primary": hrds_agg, "topp_baseline": base_agg,
        "bootstrap": boot, "fairness": fairness,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
