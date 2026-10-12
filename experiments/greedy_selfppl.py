#!/usr/bin/env python
# greedy_selfppl.py  (single-model, robust)
# P0 补实验：greedy 解码的 self-PPL / rep4 / distinct-3（同协议：100 prompts / 200 tokens / self-PPL 只算续写）。
# 用途：作为 Δ = PPL_HRSD / PPL_greedy 的分母（相对 greedy 的超额代价基准）。
# 每次只跑一个模型，结果立即写 JSON（即使后续模型挂了也不丢），加载前 empty_cache 防显存累积。

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys, json, argparse, torch

H = __import__("hrds_experiment_qwen")
DEVICE = H.DEVICE
SEED = H.SEED
RUNTIME = H.RUNTIME
PROMPT_MAX_TOKENS = H.PROMPT_MAX_TOKENS


def load_tok_model(path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dtype = torch.bfloat16 if "Llama" in path else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    return tok, model


@torch.no_grad()
def greedy_generate(model, input_ids, max_new):
    out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[0, -1]
    past = out.past_key_values
    gids = []
    for step in range(max_new):
        tok_id = int(torch.argmax(logits, dim=-1).item())
        gids.append(tok_id)
        if step == max_new - 1:
            break
        nt = torch.tensor([[tok_id]], device=DEVICE)
        pos = torch.tensor([[input_ids.shape[1] + step]], device=DEVICE, dtype=torch.long)
        out = model(input_ids=nt, position_ids=pos, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1]
        past = out.past_key_values
    return gids


def run_one(path, n_prompts, max_new):
    RUNTIME["max_new_tokens"] = max_new
    tok, model = load_tok_model(path)
    prompts = H.load_prompts(n_prompts)
    import numpy as np
    ppls, r4s, d3s, n_pos = [], [], [], 0
    for i, p in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        if ids.shape[1] > PROMPT_MAX_TOKENS:
            ids = ids[:, :PROMPT_MAX_TOKENS]
        plen = ids.shape[1]
        gids = greedy_generate(model, ids, max_new)
        full = ids[0].tolist() + gids
        ppl = H.compute_ppl(model, full, plen)
        text = tok.decode(gids, skip_special_tokens=True)
        r4 = H.rep_n([text], 4)
        d3 = H.distinct_n([text], 3)
        ppls.append(ppl); r4s.append(r4); d3s.append(d3)
        if r4 > 0:
            n_pos += 1
        if (i + 1) % 20 == 0 or n_prompts <= 3:
            print(f"  done {i+1}/{n_prompts}", flush=True)
    return {
        "model": path,
        "ppl": float(np.mean(ppls)),
        "rep4": float(np.mean(r4s)),
        "distinct3": float(np.mean(d3s)),
        "n_rep4_pos": int(n_pos),
        "n_evaluated": len(ppls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_prompts = 3 if args.smoke else 100
    max_new = 30 if args.smoke else 200
    import re
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", args.model_path.split("/")[-1])
    out_path = f"/home/xufei/greedy_{tag}.json"
    print(f"=== greedy on {args.model_path} -> {out_path} ===", flush=True)
    torch.cuda.empty_cache()
    res = run_one(args.model_path, n_prompts, max_new)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("GREEDY_RESULT " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
