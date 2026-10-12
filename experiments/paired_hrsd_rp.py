#!/usr/bin/env python
# paired_hrsd_rp.py
# DECISIVE closure of the open HRSD-vs-RP question.
#
# Prior state (BUG): rp_sweep.py compared HRSD (run A, rep=8) to RP=1.2 (run B,
# rep=2) as CROSS-RUN point estimates. The two runs used independent RNG streams,
# so the "same prompt" was drawn with different randomness -> NOT paired. Any
# apparent RP edge could be sampling noise.
#
# This script runs HRSD-primary and RP=1.2 in the SAME process, SAME model load,
# SAME 100 prompts, SAME seed -> true per-prompt pairing. We then:
#   1) paired bootstrap on n-gram metrics (rep4 lower / distinct3 higher / ppl higher)
#      using the existing R.paired_bootstrap.
#   2) a SEMANTIC axis (Path B): prompt<->generation topic alignment via the local
#      Qwen's own last-hidden mean-pool (no external model, firewall-safe). Tests
#      whether RP's blind token penalty drifts off-topic more than HRSD's principled
#      reshaping. Paired bootstrap on the cosine score (higher = HRSD more on-topic).
#
# Output: paired_hrsd_rp.json  (aggs, bootstrap, topic_bootstrap, per_prompt w/ text)

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hrds_experiment_qwen as H
import robustness_significance as R

SEED = 42


def topic_alignment(model, tok, prompt_ids, gen_ids):
    """Prompt<->generation semantic alignment using the generator's own hidden states.
    Returns cosine(prompt_emb, gen_emb) or None on failure. Proxy only."""
    try:
        if not gen_ids:
            return None
        with torch.no_grad():
            p_out = model(prompt_ids, output_hidden_states=True)
            ph = p_out.hidden_states[-1][0]                 # [Lp, D]
            pm = ph.mean(dim=0, keepdim=True)               # [1, D]
            full = torch.cat([prompt_ids[0],
                              torch.tensor(gen_ids, device=DEVICE)]).unsqueeze(0)
            g_out = model(full, output_hidden_states=True)
            gh = g_out.hidden_states[-1][0]                 # [Lf, D]
            gen_part = gh[prompt_ids.shape[1]:]             # generated portion
            if gen_part.numel() == 0:
                return None
            gm = gen_part.mean(dim=0, keepdim=True)
            cos = F.cosine_similarity(pm, gm).item()
            return float(cos)
    except Exception as e:
        return None


def paired_boot_1d(a, b, higher_is_better, n_boot=2000, seed=777):
    a = np.array([x for x in a if x is not None], dtype=float)
    b = np.array([x for x in b if x is not None], dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[:n][idx].mean(axis=1) - b[:n][idx].mean(axis=1)
    obs = float(a[:n].mean() - b[:n].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one = (float((diffs <= 0).mean()) if higher_is_better
             else float((diffs >= 0).mean()))
    p_two = float(2 * min(p_one, 1 - p_one))
    return {"obs": obs, "ci": [float(lo), float(hi)],
            "p_one": p_one, "p_two": p_two,
            "n": int(n), "higher_is_better": bool(higher_is_better)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rp", type=float, default=1.2,
                    help="repetition penalty for the RP arm")
    ap.add_argument("--out", default="paired_hrsd_rp.json")
    args = ap.parse_args()
    n = 3 if args.smoke else args.n_prompts
    max_new = 30 if args.smoke else args.max_new_tokens

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(H.MODEL_PATH, local_files_only=True)
    H.tok = tok
    model = AutoModelForCausalLM.from_pretrained(
        H.MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to(DEVICE).eval()
    print("Model loaded on", DEVICE)

    data = R.build_prompts(n, max_new)

    # ---- HRSD primary (matches reported PPL 3.458 / rep8 / d3 0.9959) ----
    hrds_cfg = {"method": "hrds"}
    hrds_agg, hrds_pp = R.evaluate(
        model, tok, data, hrds_cfg,
        torch.Generator(device=DEVICE).manual_seed(SEED), max_new)

    # ---- RP = args.rp + top-p 0.9 ----
    rp_cfg = {"method": "rp", "penalty": args.rp, "top_p": 0.9}
    rp_agg, rp_pp = R.evaluate(
        model, tok, data, rp_cfg,
        torch.Generator(device=DEVICE).manual_seed(SEED), max_new)

    print(f"\nHRSD primary : PPL={hrds_agg['ppl']:.3f} rep4={hrds_agg['rep4']:.5f} "
          f"rep4>0={hrds_agg['n_rep4_pos']}/{n} d3={hrds_agg['distinct3']:.4f}")
    print(f"RP={args.rp:<4}     : PPL={rp_agg['ppl']:.3f} rep4={rp_agg['rep4']:.5f} "
          f"rep4>0={rp_agg['n_rep4_pos']}/{n} d3={rp_agg['distinct3']:.4f}")

    # ---- (1) n-gram paired bootstrap ----
    boot = R.paired_bootstrap(hrds_pp, rp_pp, n_boot=2000)
    print("\n=== N-GRAM PAIRED BOOTSTRAP (HRSD vs RP=%.1f) ===" % args.rp)
    for k, v in boot.items():
        print(f"  {k:10s} obs={v['obs']:+.4f} 95%CI={v['ci']} "
              f"p_one={v['p_one']:.4f} p_two={v['p_two']:.4f}")

    # ---- (2) semantic topic-alignment (Path B) ----
    # R.evaluate pp entries store 'text' (not raw gen_ids); re-tokenize the
    # generated text (no special tokens) to approximate the generated id stream
    # for the hidden-state alignment forward.
    print("\n=== SEMANTIC: prompt<->generation topic alignment ===")
    def realign_from_text(model, tok, ids, text):
        if not text:
            return None
        try:
            gids = tok.encode(text, add_special_tokens=False)
            return topic_alignment(model, tok, ids, gids)
        except Exception:
            return None
    hrds_topic = [realign_from_text(model, tok, data[i][0], hrds_pp[i].get("text", ""))
                  for i in range(len(data))]
    rp_topic = [realign_from_text(model, tok, data[i][0], rp_pp[i].get("text", ""))
                for i in range(len(data))]

    topic_boot = paired_boot_1d(hrds_topic, rp_topic, higher_is_better=True,
                                n_boot=2000, seed=777)
    ht = [x for x in hrds_topic if x is not None]
    rt = [x for x in rp_topic if x is not None]
    print(f"  HRSD mean topic-align = {np.mean(ht):.4f}" if ht else "  HRSD topic n/a")
    print(f"  RP   mean topic-align = {np.mean(rt):.4f}" if rt else "  RP   topic n/a")
    if topic_boot:
        print(f"  topic paired boot: obs={topic_boot['obs']:+.4f} "
              f"95%CI={topic_boot['ci']} p_two={topic_boot['p_two']:.4f}")

    # attach topic scores into per_prompt
    for i in range(len(hrds_pp)):
        hrds_pp[i]["topic"] = hrds_topic[i]
        rp_pp[i]["topic"] = rp_topic[i]

    result = {
        "n_prompts": n, "max_new": max_new, "rp_penalty": args.rp,
        "hrds_config": {"K1": 2.0, "K2": 1.0, "LAM": 1.0, "ETA": 0.2,
                        "THETA": 1.0, "KMAX": 2, "ALPHA": 0.3},
        "rp_config": {"penalty": args.rp, "top_p": 0.9},
        "hrds_agg": hrds_agg, "rp_agg": rp_agg,
        "ngram_bootstrap": boot, "topic_bootstrap": topic_boot,
        "per_prompt": {"hrds": hrds_pp, "rp": rp_pp},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
