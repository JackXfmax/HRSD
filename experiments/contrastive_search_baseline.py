#!/usr/bin/env python
# contrastive_search_baseline.py
# 补充实验：把 contrastive search 作为第二个"结构化重复抑制器" baseline，
# 与 HRSD-primary 在【完全相同协议】(同模型/同100条 fallback prompts/seed=42/200 tokens)
# 下做配对 bootstrap，回答："HRSD 是否也支配 contrastive search？"
# 理论预期：与 RP 一样，contrastive 也落在同一条 self-PPL 紧边界上 -> 互不支配
#           (普适边界定理的进一步实证)。
#
# 复用 hrds_experiment_qwen 的 prompts / 指标 / 常量，确保协议一致。

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys, json, time, math, argparse
import torch
import torch.nn.functional as F

H = __import__("hrds_experiment_qwen")
DEVICE = H.DEVICE
SEED = H.SEED
MODEL_PATH = H.MODEL_PATH
PROMPT_MAX_TOKENS = H.PROMPT_MAX_TOKENS
RUNTIME = H.RUNTIME

ALPHA_CS = 0.6   # contrastive penalty weight (Su et al. 2022 default)
K_CS = 5         # candidate set size
WINDOW_CS = 16   # past hidden states window


def load_tok_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    return tok, model


def expand_cache(past, k):
    """把 KV 缓存扩批到 batch=k（兼容 DynamicCache 与旧 tuple 格式）。"""
    if hasattr(past, "key_cache"):  # transformers >=4.36 DynamicCache
        from transformers.cache_utils import DynamicCache
        pk = DynamicCache()
        for li in range(len(past.key_cache)):
            pk.update(past.key_cache[li].repeat(k, 1, 1, 1),
                      past.value_cache[li].repeat(k, 1, 1, 1), li)
        return pk
    return tuple(tuple(t.repeat(k, 1, 1, 1) for t in layer) for layer in past)


@torch.no_grad()
def contrastive_generate(model, input_ids, gen, alpha=ALPHA_CS, k=K_CS, window=WINDOW_CS):
    prompt_len = input_ids.shape[1]
    out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    logits = out.logits[0, -1].float()
    past = out.past_key_values
    last_h = out.hidden_states[-1][0, -1]
    past_h = [last_h]
    gen_ids = []
    M = RUNTIME["max_new_tokens"]
    for step in range(M):
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k)
        # batched candidate forward: expand past to batch k
        cands = topk_idx.unsqueeze(1)                       # (k,1)
        pos = torch.tensor([[prompt_len + step]] * k, device=DEVICE, dtype=torch.long)
        pk = expand_cache(past, k)
        cout = model(input_ids=cands, position_ids=pos, past_key_values=pk,
                     use_cache=True, output_hidden_states=True)
        cand_h = cout.hidden_states[-1][:, -1, :]            # (k, h)
        buf = torch.stack(past_h[-window:])                  # (<=window, h)
        buf_n = F.normalize(buf, dim=-1)
        cand_n = F.normalize(cand_h, dim=-1)
        sim = torch.matmul(cand_n, buf_n.t())                # (k, buf)
        max_sim = sim.max(dim=-1).values
        score = (1.0 - alpha) * torch.log(topk_vals) + alpha * max_sim
        tok = int(topk_idx[int(torch.argmax(score).item())].item())
        gen_ids.append(tok)
        if step == M - 1:
            break
        nt = torch.tensor([[tok]], device=DEVICE)
        pos1 = torch.tensor([[prompt_len + step]], device=DEVICE, dtype=torch.long)
        out2 = model(input_ids=nt, position_ids=pos1, past_key_values=past,
                     use_cache=True, output_hidden_states=True)
        logits = out2.logits[0, -1].float()
        past = out2.past_key_values
        past_h.append(out2.hidden_states[-1][0, -1])
    return gen_ids


def paired_bootstrap(a, b, n=2000):
    """per-prompt paired bootstrap for two aligned lists of per-prompt values."""
    import numpy as np
    diff = np.array([x - y for x, y in zip(a, b)], dtype=np.float64)
    mean = float(diff.mean())
    rng_np = np.random.default_rng(SEED)
    means = np.array([rng_np.choice(diff, size=diff.size, replace=True).mean()
                      for _ in range(n)])
    ci_lo, ci_hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    p_two = float(min((means <= 0).mean(), (means >= 0).mean()) * 2)
    return {"obs": mean, "ci": [ci_lo, ci_hi], "p_two": p_two}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_prompts = 3 if args.smoke else 100
    max_new = 30 if args.smoke else 200
    RUNTIME["max_new_tokens"] = max_new

    tok, model = load_tok_model()

    # HRSD primary 配置（与 §4.4 / paired 完全一致）
    H.K1, H.K2, H.LAM, H.ETA, H.THETA, H.KMAX, H.ALPHA = 2.0, 1.0, 1.0, 0.2, 1.0, 2, 0.3

    prompts = H.load_prompts(n_prompts)
    prompt_data = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        if ids.shape[1] > PROMPT_MAX_TOKENS:
            ids = ids[:, :PROMPT_MAX_TOKENS]
        prompt_data.append((ids, ids.shape[1]))

    def run_method(fn, gen):
        reps4, ds3, ppls, n_pos, texts, fulls = [], [], [], [], [], []
        for i, (ids, plen) in enumerate(prompt_data):
            res = fn(model, ids, gen)
            gids = res[0] if isinstance(res, tuple) else res
            text = tok.decode(gids, skip_special_tokens=True)
            full = ids[0].tolist() + gids
            ppl = H.compute_ppl(model, full, plen)
            r4 = H.rep_n([text], 4)
            d3 = H.distinct_n([text], 3)
            reps4.append(r4); ds3.append(d3); ppls.append(ppl)
            n_pos.append(1 if r4 > 0 else 0); texts.append(text); fulls.append(full)
            if (i + 1) % 20 == 0 or args.smoke:
                print(f"  done {i+1}/{n_prompts}", flush=True)
        return reps4, ds3, ppls, n_pos, texts, fulls

    gen_h = torch.Generator(device=DEVICE).manual_seed(SEED)
    gen_c = torch.Generator(device=DEVICE).manual_seed(SEED)
    hr, cr = run_method(lambda m, ids, g: H.generate_sequence(m, ids, "hrds", g), gen_h), \
             run_method(lambda m, ids, g: contrastive_generate(m, ids, g), gen_c)

    def agg(x):
        import numpy as np
        return {"rep4": float(np.mean(x[0])), "distinct3": float(np.mean(x[1])),
                "ppl": float(np.mean(x[2])), "n_rep4_pos": int(sum(x[3])),
                "n_evaluated": len(x[0]), "mean_gen_len": float(max_new)}

    h_agg, c_agg = agg(hr), agg(cr)
    boot = {
        "rep4": paired_bootstrap(hr[0], cr[0]),
        "distinct3": paired_bootstrap(hr[1], cr[1]),
        "ppl": paired_bootstrap(hr[2], cr[2]),
    }
    result = {
        "config": {"seed": SEED, "n_prompts": n_prompts, "max_new_tokens": max_new,
                   "gen_model": MODEL_PATH, "ppl_model": "SELF", "device": DEVICE,
                   "contrastive": {"alpha": ALPHA_CS, "k": K_CS, "window": WINDOW_CS},
                   "hrds_primary": {"K1": 2.0, "K2": 1.0, "LAM": 1.0, "ETA": 0.2,
                                     "THETA": 1.0, "KMAX": 2, "ALPHA": 0.3}},
        "hrds_primary_agg": h_agg, "contrastive_agg": c_agg, "paired_bootstrap": boot,
    }
    print("HRSD-primary:", h_agg)
    print("Contrastive :", c_agg)
    print("Bootstrap   :", boot)
    if not args.smoke:
        with open("contrastive_search.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("Wrote contrastive_search.json")


if __name__ == "__main__":
    main()
