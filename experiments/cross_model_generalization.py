#!/usr/bin/env python
# cross_model_generalization.py
#
# PURPOSE: fill the single biggest gap in the HRSD paper -- everything so far is
# measured on ONE model (Qwen2.5-7B-Instruct). An ACL reviewer will ask whether
# the Pareto dominance over the sampling-knob family survives a second model.
# This script reproduces the core comparison on a DIFFERENT model (default:
# Llama-3.1-8B-Instruct, pre-installed on the server, no download needed).
#
# Runs, on the SAME prompt set with the SAME seed:
#   - HRSD primary   (K1=2, K2=1, LAM=1, ETA=0.2, THETA=1.0, KMAX=2, ALPHA=0.3)
#   - Top-p T=1.0 p=0.9
#   - Repetition penalty 1.2 + top-p 0.9
# and does PAIRED bootstrap (per-prompt) for HRSD-vs-Top-p and HRSD-vs-RP.
#
# Reuses hrds_experiment_qwen.py primitives exactly as robustness_significance.py
# does, overriding only H.MODEL_PATH so the comparison stays protocol-identical.

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hrds_experiment_qwen as H

SEED = 42


def discover_model():
    """Pick a NON-Qwen instruction model so we actually generalize.
    Prefers Llama; falls back to any other local model dir."""
    bases = ["/home/xufei/models", "/home/xufei", "/data/models", "/home/xufei/.cache/huggingface/hub"]
    for base in bases:
        if not os.path.isdir(base):
            continue
        try:
            cands = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        except OSError:
            continue
        cands = [d for d in cands if not d.startswith('.')]
        llama = [d for d in cands if 'llama' in d.lower()]
        if llama:
            return os.path.join(base, sorted(llama)[0])
        non_qwen = [d for d in cands if 'qwen' not in d.lower()]
        # heuristic: a real model dir contains a config.json
        real = [d for d in non_qwen
                if os.path.exists(os.path.join(base, d, 'config.json'))]
        if real:
            return os.path.join(base, sorted(real)[0])
    return None


def build_prompts(n, max_new, tok):
    H.RUNTIME["max_new_tokens"] = max_new
    data = []
    for p in H.load_prompts(n):
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
    """HF-style repetition penalty + nucleus, own KV-cache loop."""
    gens = []
    for (input_ids, prompt_len) in data:
        out = model(input_ids=input_ids, use_cache=True)
        logits = out.logits[0, -1]
        past = out.past_key_values
        gen_ids, hist = [], []
        for step in range(max_new):
            lp = logits.float().clone()
            if penalty != 1.0 and hist:
                for t in set(hist):
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


def paired_bootstrap(a_pp, b_pp, n_boot=2000, seed=12345):
    """a - b, per-prompt paired. Returns obs/CI/p for rep4, distinct3, ppl."""
    keys = [("rep4", "lower"), ("distinct3", "higher"), ("ppl", "higher")]
    rng = np.random.default_rng(seed)
    out = {}
    for name, direction in keys:
        a = np.array([x[name] for x in a_pp if x[name] is not None], dtype=float)
        b = np.array([x[name] for x in b_pp if x[name] is not None], dtype=float)
        m = min(len(a), len(b))
        if m < 2:
            out[name] = {"obs": None, "ci": None, "p_one": None, "p_two": None}
            continue
        a, b = a[:m], b[:m]
        idx = rng.integers(0, m, size=(n_boot, m))
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        obs = float(a.mean() - b.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_one = float((diffs >= 0).mean()) if direction == "lower" else float((diffs <= 0).mean())
        out[name] = {"obs": obs, "ci": [float(lo), float(hi)],
                     "p_one": p_one, "p_two": float(2 * min(p_one, 1 - p_one)),
                     "direction": direction}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None,
                    help="explicit model dir; if omitted, auto-discover a non-Qwen model")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--rp-penalty", type=float, default=1.2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="cross_model_generalization.json")
    args = ap.parse_args()

    n = 3 if args.smoke else args.n_prompts
    max_new = 30 if args.smoke else args.max_new_tokens

    model_path = args.model_path or discover_model()
    if not model_path:
        print("FATAL: no non-Qwen model found on this host.", file=sys.stderr)
        sys.exit(2)
    print(f"Device: {DEVICE}")
    print(f"Model : {model_path}")

    # override so every helper (tokenizer/ppl/prompts) uses the new model
    H.MODEL_PATH = model_path

    from transformers import AutoTokenizer, AutoModelForCausalLM
    low = model_path.lower()
    dtype = torch.bfloat16 if ("llama" in low or "mistral" in low or "qwen2" in low) else torch.float16
    print(f"dtype : {dtype}")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    H.tok = tok
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True).to(DEVICE).eval()
    print("Model loaded.")

    data = build_prompts(n, max_new, tok)
    print(f"prompts: {len(data)} (max_new={max_new})")

    t0 = time.time()
    cfgs = [
        ("hrsd", {"method": "hrds"}, "HRSD primary"),
        ("topp", {"method": "topp", "p": 0.9}, "Top-p T=1.0 p=0.9"),
        ("rp", {"method": "rp", "penalty": args.rp_penalty, "top_p": 0.9},
         f"RP={args.rp_penalty} + top-p0.9"),
    ]
    aggs, pps = {}, {}
    for key, cfg, tag in cfgs:
        s = time.time()
        agg, pp = evaluate(model, tok, data, cfg,
                           torch.Generator(device=DEVICE).manual_seed(SEED), max_new)
        aggs[key] = agg
        pps[key] = pp
        print(f"  {tag:22s} PPL={agg['ppl']:.3f} rep4={agg['rep4']:.4f} "
              f"rep4>0={agg['n_rep4_pos']}/{n} d3={agg['distinct3']:.4f} "
              f"({time.time()-s:.0f}s)")

    print("\n=== PAIRED BOOTSTRAP: HRSD vs Top-p ===")
    boot_topp = paired_bootstrap(pps["hrsd"], pps["topp"])
    for k, v in boot_topp.items():
        print(f"  {k:10s} obs={v['obs']:.4f} 95%CI={v['ci']} p_two={v['p_two']:.4f}")

    print("\n=== PAIRED BOOTSTRAP: HRSD vs RP ===")
    boot_rp = paired_bootstrap(pps["hrsd"], pps["rp"])
    for k, v in boot_rp.items():
        print(f"  {k:10s} obs={v['obs']:.4f} 95%CI={v['ci']} p_two={v['p_two']:.4f}")

    result = {
        "model_path": model_path,
        "n_prompts": n, "max_new": max_new, "seed": SEED,
        "hrsd_primary": aggs["hrsd"], "topp_baseline": aggs["topp"],
        f"rp_{args.rp_penalty}": aggs["rp"],
        "bootstrap_vs_topp": boot_topp,
        "bootstrap_vs_rp": boot_rp,
        "secs": time.time() - t0,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
