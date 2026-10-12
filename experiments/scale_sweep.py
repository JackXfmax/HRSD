#!/usr/bin/env python
# scale_sweep.py
# 尺度阶梯扫描：在 Qwen2.5-{0.5B,1.5B,3B} 上以【与主实验完全相同的协议】跑
#   HRSD-primary / Top-p(T=1.0,p=0.9) / RP(1.2 + top-p 0.9)
# 目的：
#   (a) 检验 HRSD 对 Top-p 的 Pareto 支配是否跨【模型规模】稳健；
#   (b) 检验 Corollary 12.2 的"代价可预算/可预测"是否跨规模成立（HRSD vs RP 的
#       跨模型 PPL 漂移是否仍远小于 RP）。
# 协议与 paired_hrsd_rp.py / robustness_significance.py 完全一致：
#   同 100 条 fallback prompts / seed=42 / 200 max_new_tokens / self-PPL / 同超参。
# 复用 robustness_significance.R 的 build_prompts / evaluate / paired_bootstrap。

import os
import sys
import json
import argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hrds_experiment_qwen as H
import robustness_significance as R

SEED = 42
HRDS_PRIMARY = {"K1": 2.0, "K2": 1.0, "LAM": 1.0, "ETA": 0.2,
                "THETA": 1.0, "KMAX": 2, "ALPHA": 0.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n = 3 if args.smoke else args.n_prompts
    max_new = 30 if args.smoke else args.max_new_tokens
    short = os.path.basename(args.model_path.rstrip("/"))

    # 覆盖模型路径（必须在 build_prompts 之前，tokenizer 从同一路径加载）
    H.MODEL_PATH = args.model_path

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    H.tok = tok
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to("cuda").eval()
    print(f"[{short}] model loaded", flush=True)

    data = R.build_prompts(n, max_new)

    arms = {}
    pps = {}
    for tag, cfg in [
        ("hrds", {"method": "hrds"}),
        ("topp", {"method": "topp", "p": 0.9}),
        ("rp", {"method": "rp", "penalty": 1.2, "top_p": 0.9}),
    ]:
        agg, pp = R.evaluate(model, tok, data, cfg,
                             torch.Generator(device="cuda").manual_seed(SEED), max_new)
        arms[tag] = agg
        pps[tag] = pp
        print(f"[{short}] {tag:5s} PPL={agg['ppl']:.3f} rep4={agg['rep4']:.5f} "
              f"rep4>0={agg['n_rep4_pos']}/{n} d3={agg['distinct3']:.4f}", flush=True)

    boot_topp = R.paired_bootstrap(pps["hrds"], pps["topp"], n_boot=2000)
    boot_rp = R.paired_bootstrap(pps["hrds"], pps["rp"], n_boot=2000)
    print(f"[{short}] BOOT vs topp: "
          + " ".join(f"{k}={v['obs']:+.4f}(p_two={v['p_two']:.4f})" for k, v in boot_topp.items()),
          flush=True)
    print(f"[{short}] BOOT vs rp  : "
          + " ".join(f"{k}={v['obs']:+.4f}(p_two={v['p_two']:.4f})" for k, v in boot_rp.items()),
          flush=True)

    out = {
        "model_path": args.model_path, "model": short,
        "config": {"seed": SEED, "n_prompts": n, "max_new_tokens": max_new,
                   "ppl_model": "SELF",
                   "hrds_primary": HRDS_PRIMARY,
                   "topp": {"p": 0.9, "temp": 1.0},
                   "rp": {"penalty": 1.2, "top_p": 0.9}},
        "aggs": arms,
        "bootstrap": {"hrds_vs_topp": boot_topp, "hrds_vs_rp": boot_rp},
        "per_prompt": pps,
    }
    outname = args.out or f"scale_sweep_{short}.json"
    with open(outname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[{short}] Wrote {outname}", flush=True)


if __name__ == "__main__":
    main()
