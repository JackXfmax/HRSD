#!/usr/bin/env python
# topp_texts.py
# 目的：为论文"同一 prompt 三段对照"补采 Top-p (T=1.0, p=0.9) 的逐 prompt 生成文本。
# 现有 JSON 中只有 paired_hrsd_rp.json 存了 HRSD / RP 的 text，pareto 只存聚合 -> 无 Top-p 文本。
#
# 协议与历史实验完全一致：同 Qwen2.5-7B-Instruct / 同 H.load_prompts(100) 的 100 条
# fallback prompts / seed=42 / 200 max_new_tokens / self-PPL。
# 默认 TOP_P=0.9, TEMPERATURE=1.0 即 pareto arm A (kind=topp_T, temp=1.0, top_p=0.9)。
# 输出按 index 0..99 与 paired_hrsd_rp.json 的 per_prompt.hrds / per_prompt.rp 对齐。

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys, json, argparse, time

import torch

H = __import__("hrds_experiment_qwen")
DEVICE = H.DEVICE
SEED = H.SEED
MODEL_PATH = H.MODEL_PATH
PROMPT_MAX_TOKENS = H.PROMPT_MAX_TOKENS
RUNTIME = H.RUNTIME

# 显式声明 Top-p arm A 参数（与 pareto_frontier.json arm A 一致）
TOP_P = 0.9
TEMPERATURE = 1.0


def load_tok_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_prompts = 3 if args.smoke else 100
    max_new = 30 if args.smoke else 200
    RUNTIME["max_new_tokens"] = max_new

    # Top-p arm A 参数生效（覆盖模块级常量，确保与 pareto 一致）
    H.TOP_P = TOP_P
    H.TEMPERATURE = TEMPERATURE

    tok, model = load_tok_model()

    prompts = H.load_prompts(n_prompts)
    prompt_data = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        if ids.shape[1] > PROMPT_MAX_TOKENS:
            ids = ids[:, :PROMPT_MAX_TOKENS]
        prompt_data.append((ids, ids.shape[1]))

    gen_t = torch.Generator(device=DEVICE).manual_seed(SEED)

    rows = []
    t0 = time.time()
    for i, (ids, plen) in enumerate(prompt_data):
        res = H.generate_sequence(model, ids, "topp", gen_t)
        gids = res[0] if isinstance(res, tuple) else res
        text = tok.decode(gids, skip_special_tokens=True)
        full = ids[0].tolist() + gids
        ppl = H.compute_ppl(model, full, plen)
        r4 = H.rep_n([text], 4)
        d3 = H.distinct_n([text], 3)
        rows.append({
            "idx": i,
            "prompt": prompts[i],
            "text": text,
            "rep4": float(r4),
            "distinct3": float(d3),
            "ppl": float(ppl),
            "gen_len": len(gids),
        })
        if (i + 1) % 20 == 0 or args.smoke:
            print(f"  topp done {i+1}/{n_prompts}  ({time.time()-t0:.0f}s)", flush=True)

    import numpy as np
    agg = {
        "rep4": float(np.mean([r["rep4"] for r in rows])),
        "distinct3": float(np.mean([r["distinct3"] for r in rows])),
        "ppl": float(np.mean([r["ppl"] for r in rows])),
        "n_rep4_pos": int(sum(1 for r in rows if r["rep4"] > 0)),
        "n_evaluated": len(rows),
    }
    out = {
        "config": {
            "method": "topp", "temp": TEMPERATURE, "top_p": TOP_P,
            "seed": SEED, "n_prompts": n_prompts, "max_new_tokens": max_new,
            "gen_model": MODEL_PATH, "ppl_model": "SELF",
            "note": "index-aligned with paired_hrsd_rp.json per_prompt.hrds / per_prompt.rp",
        },
        "agg": agg,
        "per_prompt": rows,
    }
    print("TOP-P AGG:", json.dumps(agg, ensure_ascii=False))
    if not args.smoke:
        with open("topp_T10_texts.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("Wrote topp_T10_texts.json")


if __name__ == "__main__":
    main()
