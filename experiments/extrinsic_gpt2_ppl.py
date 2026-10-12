#!/usr/bin/env python
# extrinsic_gpt2_ppl.py
# Extrinsic 信号：用**外部 LM（gpt2）**给已生成的文本打分，回答 Limitations 里
# "全是 intrinsic（self-PPL/rep4/distinct-3）" 这一最大短板。
#
# 直接消费已有产物，无需重新生成：
#   paired_hrsd_rp.json  -> per_prompt.hrds[].text / per_prompt.rp[].text  (Qwen-7B 生成)
#   topp_T10_texts.json  -> per_prompt[].text                            (Qwen-7B 生成, Top-p T=1.0 p=0.9)
# 三者按 index 对齐（同 100 prompts / seed=42 / 200 tokens）。
#
# 指标：external-PPL = exp(-mean log p_gpt2(token))，对生成续写逐 token 计算。
# 与 self-PPL 的区别：self-PPL 用生成模型自己打分（会被 HRSD 的分布塑造"高估"），
# external-PPL 用一个**独立的、未参与解码**的 LM 打分，是真正的 extrinsic 质量信号。
# 附带 paired bootstrap（2000 resamples, n=100），遵守"无 bootstrap 不称显著"铁律。

import os
import sys
import json
import argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch
import torch.nn.functional as F

SEED = 42
GPT2 = "/home/xufei/models/gpt2"


def ext_ppl(model, tok, text):
    """gpt2 对该文本的平均 token logprob -> external perplexity。"""
    if not text or not text.strip():
        return None
    ids = tok(text, return_tensors="pt", add_special_tokens=True)
    input_ids = ids["input_ids"].to("cuda")
    if input_ids.shape[1] < 2:
        return None
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, :-1].float()          # [T-1, V]
        tgt = input_ids[0, 1:]                        # [T-1]
        logp = F.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        mean_logp = tok_logp.mean().item()
    return float(torch.exp(torch.tensor(-mean_logp)).item())


def paired_boot(a, b, n_boot=2000, seed=12345):
    import numpy as np
    a = np.array([x for x in a if x is not None], dtype=float)
    b = np.array([x for x in b if x is not None], dtype=float)
    n = min(len(a), len(b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[:n][idx].mean(axis=1) - b[:n][idx].mean(axis=1)
    obs = float(a[:n].mean() - b[:n].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"obs": obs, "ci": [float(lo), float(hi)], "p_two": p_two, "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="paired_hrsd_rp.json")
    ap.add_argument("--topp", default="topp_T10_texts.json")
    ap.add_argument("--out", default="extrinsic_gpt2_ppl.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(GPT2, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        GPT2, local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
    print("gpt2 loaded", flush=True)

    paired = json.load(open(args.paired, encoding="utf-8"))
    topp = json.load(open(args.topp, encoding="utf-8"))
    hrds_pp = paired["per_prompt"]["hrds"]
    rp_pp = paired["per_prompt"]["rp"]
    tp_pp = topp["per_prompt"]
    n = min(len(hrds_pp), len(rp_pp), len(tp_pp))
    print(f"aligned prompts: {n}", flush=True)

    rows = {"hrds": [], "topp": [], "rp": []}
    for i in range(n):
        for tag, pp in (("hrds", hrds_pp), ("topp", tp_pp), ("rp", rp_pp)):
            txt = pp[i].get("text", "")
            p = ext_ppl(model, tok, txt)
            rows[tag].append({"idx": i, "ext_ppl": p,
                              "self_ppl": pp[i].get("ppl")})
        if (i + 1) % 20 == 0:
            print(f"  scored {i+1}/{n}", flush=True)

    import numpy as np
    def agg(xs):
        v = [x["ext_ppl"] for x in xs if x["ext_ppl"] is not None]
        return {"ext_ppl_mean": float(np.mean(v)), "n": len(v),
                "ext_ppl_median": float(np.median(v))}
    aggs = {t: agg(rows[t]) for t in rows}
    print("AGG:", json.dumps(aggs, ensure_ascii=False), flush=True)

    a = [x["ext_ppl"] for x in rows["hrds"]]
    b = [x["ext_ppl"] for x in rows["topp"]]
    c = [x["ext_ppl"] for x in rows["rp"]]
    boot = {"hrds_vs_topp": paired_boot(a, b), "hrds_vs_rp": paired_boot(a, c)}
    print("BOOT:", json.dumps(boot, ensure_ascii=False), flush=True)

    out = {
        "config": {"featurizer": GPT2, "metric": "external-LM perplexity (gpt2)",
                   "seed": SEED, "n_prompts": n,
                   "gen_model": "Qwen2.5-7B-Instruct",
                   "note": "texts reused from paired_hrsd_rp.json + topp_T10_texts.json; index-aligned"},
        "aggs": aggs, "bootstrap": boot, "per_prompt": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
