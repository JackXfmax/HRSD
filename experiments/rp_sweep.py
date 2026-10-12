#!/usr/bin/env python
# rp_sweep.py
# Fill the decisive fairness gap: repetition penalty was only tested at 1.5 / 2.0,
# which sit at PPL 14.1 / 19.5 -- far to the right of HRSD's 3.458. To claim
# "at EQUAL PPL, HRSD suppresses degeneration more", we need RP operating points
# in the same PPL band. Sweep mild penalties 1.1 - 1.4 and report (PPL, rep-4>0, d3).

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_significance as R
import hrds_experiment_qwen as H

DEVICE = R.DEVICE
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--penalties", type=float, nargs="+",
                    default=[1.1, 1.2, 1.3, 1.4])
    ap.add_argument("--out", default="rp_sweep.json")
    args = ap.parse_args()
    n = 3 if args.smoke else args.n_prompts
    max_new = 30 if args.smoke else args.max_new_tokens

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(H.MODEL_PATH, local_files_only=True)
    H.tok = tok
    model = AutoModelForCausalLM.from_pretrained(
        H.MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to(DEVICE).eval()
    print("Model loaded.")

    data = R.build_prompts(n, max_new)
    results = []
    for pen in args.penalties:
        cfg = {"method": "rp", "penalty": pen, "top_p": 0.9,
               "tag": f"RP={pen} + top-p0.9"}
        agg, _ = R.evaluate(model, tok, data, cfg,
                            torch.Generator(device=DEVICE).manual_seed(SEED), max_new)
        rec = {"tag": cfg["tag"], "penalty": pen, **agg}
        results.append(rec)
        print(f"  RP={pen:<4} PPL={agg['ppl']:.3f} rep4={agg['rep4']:.5f} "
              f"rep4>0={agg['n_rep4_pos']}/{n} d3={agg['distinct3']:.4f}")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n_prompts": n, "max_new": max_new,
                       "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
