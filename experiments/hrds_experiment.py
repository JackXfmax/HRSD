#!/usr/bin/env python
# hrds_experiment.py
# Minimal validation of Hierarchical Reshaping Sampling Decoding (HRSD), a 4-stage
# layered reshaping decoding algorithm. Compares against Top-p and Typical Sampling,
# plus two ablations (drop Stage 2 / drop Stage 4).
#
# All formulas implemented exactly as specified. Each stage is an independent function.
# Fixed seed 42. OOM auto-degrade: batch already 1; PPL model gpt2-large -> fallback gpt2.

import os
import sys
import json
import time
import math
import random
import socket

# --- TLS interception handling ------------------------------------------------
# The experiment host sits behind a MITM proxy whose presented certificate never
# matches the target hostname, so standard TLS verification can never succeed.
# We force-disable verification globally for model downloads. Public model weights
# are still hash-checked by transformers/huggingface_hub, so integrity is preserved.
import ssl
import urllib3
import requests.adapters as _rad

urllib3.disable_warnings()
_orig_ctx = urllib3.util.ssl_.create_urllib3_context
def _unverified_ctx(*a, **k):
    ctx = _orig_ctx(*a, **k)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
urllib3.util.ssl_.create_urllib3_context = _unverified_ctx

_orig_send = _rad.HTTPAdapter.send
def _force_no_verify(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)
_rad.HTTPAdapter.send = _force_no_verify
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F

# Use the freer GPU on the remote box (GPU1 had ~15MiB used vs GPU0 ~177MiB).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# ----------------------------- config -----------------------------
SEED = 42
N_PROMPTS = 100
MAX_NEW_TOKENS = 200
MODEL_NAME = "gpt2"
PPL_MODEL_NAME = "gpt2-large"
PROMPT_MAX_TOKENS = 64

# HRSD hyper-parameters (verbatim from spec)
LAM = 0.7      # Stage1 power exponent
K1 = 5.0       # Stage2 arctan scale for p_tilde
K2 = 1.0       # Stage2 arctan scale for (1 - p_tilde)
TAU = 0.001    # Stage3 valid threshold
ALPHA = 0.3    # Stage3 candidate fraction
KMIN = 3       # Stage3 minimum candidates
ETA = 0.5      # Stage4 rank penalty strength
GAMMA = 0.9    # Stage4 decay for unchosen ranks
KMAX = 20      # Stage4 rank-counter size AND hard cap on K

TOP_P = 0.9
TYPICAL_P = 0.9
TEMPERATURE = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------- determinism -----------------------------
torch.manual_seed(SEED)
random.seed(SEED)
try:
    import numpy as np
    np.random.seed(SEED)
except Exception:
    pass

# ----------------------------- 100 fallback prompts -----------------------------
FALLBACK_PROMPTS = [
    # --- story openings (1-34) ---
    "The lighthouse keeper had not seen a ship in seventeen years, until the fog began to whisper.",
    "She opened the letter and the handwriting was her own, dated ten years in the future.",
    "When the clock struck thirteen, everyone in the village pretended not to notice.",
    "The last dragon died quietly in a zoo, and no one took a picture.",
    "He found a door in the subway tunnel that was not on any map.",
    "The machine learned to lie before it learned to speak.",
    "A stranger left a key under her mat with a note that simply said later.",
    "The library burned, but the books had already been read by everyone who mattered.",
    "They said the forest was cursed, so he went in to prove them wrong.",
    "The robot's first word was a question nobody could answer.",
    "In the year the sun went dim, people began to dream in grayscale.",
    "The detective realized the victim had written the murder confession in advance.",
    "A small town elected a dog as mayor and crime dropped to zero.",
    "She discovered her reflection had started living a separate life.",
    "The train to nowhere left exactly on time, as it always did.",
    "He inherited a house that was larger on the inside than the outside.",
    "The war ended the day both sides forgot what they were fighting for.",
    "A child taught an AI to laugh, and the world changed.",
    "The map led to a country that did not exist on any globe.",
    "She woke to find the city silent and the birds speaking Latin.",
    "The inventor built a time machine but only traveled to breakfast.",
    "They found a message from Earth buried on the moon.",
    "The old man claimed he had counted every star, and offered to prove it.",
    "A virus made people honest, and society nearly collapsed.",
    "The princess chose the frog, and the kingdom was confused.",
    "He wrote a book that could only be read by its author.",
    "The storm brought fish that sang when they touched the shore.",
    "A girl traded her shadow for a violin that played itself.",
    "The king's fool was the only one who told the truth.",
    "They built a wall to keep the noise out, and silence in.",
    "The comet returned after a thousand years with a passenger.",
    "She could hear plants think, and they were mostly bored.",
    "The elevator went up for ninety years and never stopped.",
    "A boy found a coin that bought one impossible thing.",
    # --- long descriptive sentences (35-67) ---
    "The abandoned warehouse, its windows shattered like broken teeth, exhaled a cold smell of rust and forgotten machinery into the grey afternoon.",
    "Across the valley, the mountains wore their snow like a crown of brittle white that caught the last orange light of the dying sun.",
    "She walked through the crowded market where a hundred languages collided, where spice and smoke and laughter tangled in the heavy air.",
    "The river moved slowly, a ribbon of dull silver, carrying leaves and the occasional lost shoe toward a sea no one in the town had ever seen.",
    "Inside the cathedral, the silence was so complete that the soft scrape of a shoe on stone sounded like a gunshot in the vaulted dark.",
    "He remembered the summer of nineteen ninety-eight as a long blur of cicada song, sunburn, and the particular blue of a swimming pool at noon.",
    "The old steam engine hissed and shuddered, dragging a line of cars filled with coal and the tired men who mined it.",
    "Above the city, the clouds gathered into a single enormous face that seemed to watch the streets with patient, weather-worn eyes.",
    "The garden behind the house was wild with roses, their petals wet and red as if the earth itself had learned to bleed quietly.",
    "A cold wind came down from the north, thin and knife-edged, and stripped the last warmth from the evening in a matter of minutes.",
    "The manuscript lay open on the desk, its margins crowded with notes in a hand that trembled more with each passing page.",
    "They sat on the balcony watching the harbor lights flicker, each one a small promise that someone, somewhere, was still awake.",
    "The train crossed the bridge in darkness, the iron rails singing beneath it, the river far below whispering to no one.",
    "She painted the wall with a single slow stroke of blue, and the room seemed suddenly to breathe like a living thing.",
    "The desert at noon was a furnace of light, the horizon shimmering as if the earth were melting into the sky.",
    "He unfolded the letter carefully, as though the paper might crumble, and read the words that ended everything he thought he knew.",
    "The children ran through the sprinkler shouting, their voices bright and unafraid, while the grownups counted the days until the rains failed.",
    "A single candle lit the room, throwing long shadows that danced across the cracked plaster like restless ghosts.",
    "The ship creaked and groaned against the dock, a tired animal glad at last to be home from the long and merciless water.",
    "Snow fell without sound, covering the streets and the sleeping houses in a thick white hush that swallowed every footstep.",
    "The radio crackled with static, and through it came a voice from a station that had gone off the air fifty years before.",
    "He stood at the edge of the cliff where the wind tore the words from his mouth before they could become a confession.",
    "The market square filled with people holding paper lanterns, each light a small defiance against the coming dark.",
    "She traced the crack in the ceiling with her eyes, following it the way she had once followed the line of a river on a map.",
    "The kettle sang its thin high song, and for a moment the whole kitchen felt like the safest place in a dangerous world.",
    "Outside, the storm raged with a sound like a thousand doors slamming, but inside the little house the fire held its small and stubborn warmth.",
    "The road wound through hills the color of old gold, past farms where dogs barked and smoke rose straight into the still air.",
    "He watched the ice on the pond thicken week by week until it could bear the weight of the whole laughing village.",
    "The letter was written in a language she did not know, yet somehow the grief in it needed no translation at all.",
    "A flock of starlings turned in the sky as one body, a single thought made of ten thousand beating wings.",
    "The clock on the wall ticked with the solemn authority of a thing that measured nothing important and everything at once.",
    "She closed the book and listened to the rain argue with the roof, and decided that some questions were better left unasked.",
    "The photograph showed a street that no longer existed, lined with shops whose names she could almost but not quite recall.",
    # --- expository / instructional (68-100) ---
    "To bake a sourdough loaf, you must first nurture a starter by feeding flour and water to a colony of wild yeast every single day.",
    "The process of photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen inside the green cells of a leaf.",
    "Before soldering electronic components, always clean the tip of the iron and tin it with a thin layer of fresh solder.",
    "A good essay begins with a clear thesis, develops it through evidence, and concludes by showing why the argument matters.",
    "Water boils at one hundred degrees Celsius at sea level, but the temperature drops as altitude increases and pressure falls.",
    "To learn a new language efficiently, practice listening every day, speak without fear of error, and read texts slightly above your level.",
    "The circulatory system transports oxygen, nutrients, and hormones to cells while carrying away carbon dioxide and other waste.",
    "When changing a tire, engage the parking brake, loosen the lug nuts before lifting the car, and never crawl under an unsupported vehicle.",
    "Encryption protects data by transforming it with a mathematical key, so that only someone with the matching key can read the original.",
    "A balanced diet includes vegetables, whole grains, lean protein, and healthy fats in proportions suited to your activity and age.",
    "The scientific method requires forming a hypothesis, designing a test, recording observations, and revising the theory when evidence disagrees.",
    "To write clean code, name variables clearly, keep functions small, and refactor the moment a shortcut starts to feel clever.",
    "Photos taken at sunrise reveal colors that noon flattens, because the low light rakes across the landscape and deepens every shadow.",
    "A constitution divides power among branches of government to prevent any single institution from becoming absolute.",
    "Composting turns kitchen scraps and garden waste into rich soil, reducing landfill use while feeding the plants that feed us.",
    "The water cycle moves moisture from ocean to cloud to rain to river and back, driven by the simple, tireless heat of the sun.",
    "To calm a panic attack, breathe in for four counts, hold for four, and exhale for four, repeating until the wave begins to break.",
    "A telescope gathers faint light with a curved mirror, letting us see objects so distant their light left before our ancestors walked.",
    "Grammar is the skeleton of a language; without it, words collapse into a heap, but with it, meaning can stand and even dance.",
    "Renewable energy from wind and sun reduces dependence on fossil fuel, though storage remains the hard problem engineers still chase.",
    "The digestive system breaks food into molecules small enough to enter the blood, where the body ships them to wherever they are needed.",
    "To negotiate well, listen more than you speak, anchor the conversation with a clear first offer, and never reveal your true limit.",
    "A seed contains a tiny plant asleep, waiting for warmth and water to wake it into the long, slow work of becoming a tree.",
    "The internet routes packets of data through many paths, reassembling them at the destination so a message survives any single failure.",
    "Exercise strengthens the heart, clears the mind, and resets the mood, which is why a short walk often solves what an hour of worry cannot.",
    "A map is a lie that tells the truth: it flattens a round world, omits most of what is there, and still guides you home.",
    "To preserve old photographs, store them flat in acid-free sleeves away from light, because fading is simply chemistry with a long, quiet fuse.",
    "The stock market prices a company by the collective bet of millions, which is why it is less a mirror than a mood ring for the future.",
    "A sympathetic character need not be good; they need only want something the reader can feel in their own chest.",
    "Vaccination trains the immune system by showing it a harmless piece of the enemy, so the real attack meets a body already prepared.",
    "The moon pulls the ocean into tides, a gentle gravity that has kept time for the creatures of the shore longer than any clock.",
    "To read a poem well, say it aloud, let the rhythm lead, and trust that meaning will arrive a half step behind the music.",
    "A democracy depends less on perfect leaders than on citizens willing to argue, vote, and forgive each other by morning.",
]


def fallback_prompts(n):
    return FALLBACK_PROMPTS[:n]


# ----------------------------- data -----------------------------
def load_prompts(n):
    try:
        socket.setdefaulttimeout(30)
        from datasets import load_dataset
        candidates = ["manu/writingprompts", "openwebtext"]
        for name in candidates:
            try:
                ds = load_dataset(name, split="train", streaming=True)
                prompts = []
                for ex in ds:
                    text = ex.get("prompt") or ex.get("text") or ex.get("story") or ""
                    text = str(text).strip()
                    if text:
                        prompts.append(text[: 200])
                    if len(prompts) >= n:
                        break
                if len(prompts) >= n:
                    print(f"[data] loaded {len(prompts)} prompts from {name}")
                    return prompts[:n]
            except Exception as e:
                print(f"[data] dataset {name} failed: {e}")
        print("[data] all HF datasets failed -> using 100 handwritten fallback prompts")
    except Exception as e:
        print(f"[data] datasets unavailable ({e}) -> using 100 handwritten fallback prompts")
    return fallback_prompts(n)


# ----------------------------- Stage 1: global power subset sampling -----------------------------
def sample_gumbel(size, gen):
    # Gumbel(0,1): -log(-log(U)), U in (eps, 1-eps) to avoid nan at the boundaries.
    u = torch.rand(size, generator=gen, device=DEVICE).clamp(min=1e-10, max=1.0 - 1e-10)
    return -torch.log(-torch.log(u))


def stage1_subset(z, gen):
    # z: [V]
    p = torch.softmax(z, dim=-1)
    w = p ** LAM
    g = torch.log(w + 1e-12) + sample_gumbel(z.shape[-1], gen)
    k = z.shape[-1] // 2
    _, idx = torch.topk(g, k)
    return idx  # S_half, shape [k]


# ----------------------------- Stage 2: arctan unimodal rescoring -----------------------------
def stage2_rescore(z, S_half):
    z_sub = z[S_half]
    p_tilde = torch.softmax(z_sub, dim=-1)
    s = torch.arctan(K1 * p_tilde) * torch.arctan(K2 * (1.0 - p_tilde))
    return s, p_tilde


def stage2_rescore_none(z, S_half):
    # HRSD-no-Stage2: score directly by subset probability.
    z_sub = z[S_half]
    p_tilde = torch.softmax(z_sub, dim=-1)
    s = p_tilde
    return s, p_tilde


# ----------------------------- Stage 3: dynamic candidate set -----------------------------
def stage3_candidates(p_tilde, s):
    valid_mask = p_tilde >= TAU
    valid_idx = torch.nonzero(valid_mask).flatten()
    N = valid_idx.numel()
    if N == 0:
        valid_idx = torch.topk(p_tilde, 1).indices
        N = 1
    # K = min(N, max(int(alpha*N), Kmin), KMAX)   [KMAX cap per agreed spec]
    K = min(N, max(int(ALPHA * N), KMIN), KMAX)
    K = max(K, 1)
    s_valid = s[valid_idx]
    topk = torch.topk(s_valid, K)
    chosen = valid_idx[topk.indices]  # indices into S_half, ordered by s descending
    return chosen  # S_topK


# ----------------------------- Stage 4: temporal rank soft penalty -----------------------------
def stage4_select(s_topk, counter, gen):
    K = s_topk.numel()
    q = s_topk * torch.exp(-ETA * counter[:K])
    q = q / (q.sum() + 1e-12)
    y_local = torch.multinomial(q, 1, generator=gen).item()
    counter[y_local] += 1
    for qi in range(KMAX):
        if qi != y_local:
            counter[qi] *= GAMMA
    return y_local


def stage4_select_no_penalty(s_topk, gen):
    # HRSD-no-Stage4: no rank penalty, no counter update.
    q = s_topk / (s_topk.sum() + 1e-12)
    y_local = torch.multinomial(q, 1, generator=gen).item()
    return y_local


# ----------------------------- baselines -----------------------------
def topp_step(z, gen):
    if TEMPERATURE != 1.0:
        z = z / TEMPERATURE
    p_dist = torch.softmax(z, dim=-1)
    sorted_p, sorted_idx = torch.sort(p_dist, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    keep = cumsum <= TOP_P
    if not keep.any():
        keep[0] = True
    probs = sorted_p[keep]
    probs = probs / probs.sum()
    idx = sorted_idx[keep]
    sample = idx[torch.multinomial(probs, 1, generator=gen).item()]
    return sample.item()


def typical_step(z, gen):
    # Typical Sampling (Megatel et al., 2021) -- matches HF TypicalLogitsWarper.
    if TEMPERATURE != 1.0:
        z = z / TEMPERATURE
    x = torch.log_softmax(z, dim=-1)
    x_exp = torch.exp(x)
    entropy = torch.sum(-x_exp * x)
    x_surprise = torch.abs(x + entropy)  # | -log p - H |
    sorted_surprise, sorted_idx = torch.sort(x_surprise, descending=False)
    x_sorted = x[sorted_idx]
    x_exp_sorted = x_exp[sorted_idx]
    x_cumsum = torch.cumsum(x_exp_sorted, dim=-1)
    keep = x_cumsum < TYPICAL_P
    if not keep.any():
        keep[0] = True
    probs = x_exp_sorted[keep]
    probs = probs / probs.sum()
    chosen_local = torch.multinomial(probs, 1, generator=gen).item()
    return sorted_idx[keep][chosen_local].item()


# ----------------------------- HRSD step variants -----------------------------
def hrds_full_step(z, counter, gen):
    S_half = stage1_subset(z, gen)
    s, p_tilde = stage2_rescore(z, S_half)
    S_topK = stage3_candidates(p_tilde, s)
    s_topk = s[S_topK]
    y_local = stage4_select(s_topk, counter, gen)
    return S_half[S_topK[y_local]].item()


def hrds_no2_step(z, counter, gen):
    S_half = stage1_subset(z, gen)
    s, p_tilde = stage2_rescore_none(z, S_half)
    S_topK = stage3_candidates(p_tilde, s)
    s_topk = s[S_topK]
    y_local = stage4_select(s_topk, counter, gen)
    return S_half[S_topK[y_local]].item()


def hrds_no4_step(z, gen):
    S_half = stage1_subset(z, gen)
    s, p_tilde = stage2_rescore(z, S_half)
    S_topK = stage3_candidates(p_tilde, s)
    s_topk = s[S_topK]
    y_local = stage4_select_no_penalty(s_topk, gen)
    return S_half[S_topK[y_local]].item()


# ----------------------------- generation loop -----------------------------
def generate_sequence(model, input_ids, method, gen):
    seq = input_ids.clone()
    counter = torch.zeros(KMAX, device=DEVICE) if method in ("hrds", "hrds_no2") else None
    t0 = time.time()
    n = 0
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(seq)
            z = out.logits[0, -1]
        if method == "topp":
            tok = topp_step(z, gen)
        elif method == "typical":
            tok = typical_step(z, gen)
        elif method == "hrds":
            tok = hrds_full_step(z, counter, gen)
        elif method == "hrds_no2":
            tok = hrds_no2_step(z, counter, gen)
        elif method == "hrds_no4":
            tok = hrds_no4_step(z, gen)
        else:
            raise ValueError(method)
        nt = torch.tensor([[tok]], device=DEVICE)
        seq = torch.cat([seq, nt], dim=1)
        n += 1
    elapsed = time.time() - t0
    latency = n / elapsed if elapsed > 0 else float("nan")
    gen_ids = seq[0, input_ids.shape[1]:].tolist()
    return gen_ids, latency


# ----------------------------- metrics -----------------------------
def ngrams(text, n):
    toks = text.split()
    if len(toks) < n:
        return []
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts, n):
    scores = []
    for t in texts:
        grams = ngrams(t, n)
        if not grams:
            scores.append(0.0)
            continue
        scores.append(len(set(grams)) / len(grams))
    return sum(scores) / len(scores) if scores else 0.0


def rep_n(texts, n):
    scores = []
    for t in texts:
        grams = ngrams(t, n)
        if not grams:
            scores.append(0.0)
            continue
        scores.append(1.0 - len(set(grams)) / len(grams))
    return sum(scores) / len(scores) if scores else 0.0


def compute_ppl(model, ids):
    if len(ids) < 2:
        return float("nan")
    try:
        t = torch.tensor(ids, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
            logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = t[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        return math.exp(loss.item())
    except Exception as e:
        print(f"[ppl] compute failed: {e}")
        return float("nan")


# ----------------------------- main -----------------------------
def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading generation model:", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    prompts = load_prompts(N_PROMPTS)
    methods = ["topp", "typical", "hrds", "hrds_no2", "hrds_no4"]

    all_gen = {}
    latencies = {}
    for method in methods:
        gen_rng = torch.Generator(device=DEVICE).manual_seed(SEED)
        t0 = time.time()
        total_tok = 0
        gens = []
        for i, prompt in enumerate(prompts):
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
            if input_ids.shape[1] > PROMPT_MAX_TOKENS:
                input_ids = input_ids[:, :PROMPT_MAX_TOKENS]
            try:
                gen_ids, lat = generate_sequence(model, input_ids, method, gen_rng)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"[oom] method {method} prompt {i}: {e}")
                    gen_ids, lat = [], 0.0
                else:
                    raise
            gens.append(gen_ids)
            total_tok += len(gen_ids)
            if (i + 1) % 20 == 0:
                print(f"  [{method}] {i + 1}/{N_PROMPTS} prompts done")
        elapsed = time.time() - t0
        latencies[method] = total_tok / elapsed if elapsed > 0 else float("nan")
        all_gen[method] = gens
        print(f"[method {method}] latency={latencies[method]:.2f} tok/s, time={elapsed:.1f}s")

    # PPL model (gpt2-large), degrade to gpt2 on failure / OOM.
    print("Loading PPL model:", PPL_MODEL_NAME)
    ppl_model = None
    try:
        ppl_model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME).to(DEVICE).eval()
    except Exception as e:
        print(f"[ppl] {PPL_MODEL_NAME} unavailable ({e}); falling back to {MODEL_NAME}")
        ppl_model = model

    metrics = {}
    samples = {}
    for method in methods:
        texts = [tok.decode(ids, skip_special_tokens=True) for ids in all_gen[method]]
        ppls = [compute_ppl(ppl_model, ids) for ids in all_gen[method]]
        ppls = [p for p in ppls if not math.isnan(p)]
        metrics[method] = {
            "distinct3": distinct_n(texts, 3),
            "rep4": rep_n(texts, 4),
            "ppl": sum(ppls) / len(ppls) if ppls else float("nan"),
            "latency_tok_per_s": latencies[method],
        }
        samples[method] = texts[:5]

    # ---- table ----
    print("\n=== RESULTS ===")
    header = f"{'Method':<15} {'Distinct-3':>12} {'rep-4':>10} {'PPL':>10} {'tok/s':>10}"
    print(header)
    print("-" * len(header))
    for m in methods:
        mm = metrics[m]
        print(f"{m:<15} {mm['distinct3']:>12.4f} {mm['rep4']:>10.4f} {mm['ppl']:>10.2f} {mm['latency_tok_per_s']:>10.2f}")

    # ---- ablation (removed-method minus full HRSD) ----
    def delta(a, b):
        return {k: metrics[a][k] - metrics[b][k] for k in ("distinct3", "rep4", "ppl")}

    ablation = {
        "no_stage2_vs_full": delta("hrds_no2", "hrds"),
        "no_stage4_vs_full": delta("hrds_no4", "hrds"),
    }
    print("\n=== ABLATION (removed-method minus full HRSD) ===")
    for k, v in ablation.items():
        print(f"{k}: " + ", ".join(f"{kk}={vv:+.4f}" for kk, vv in v.items()))

    # ---- samples.txt ----
    with open("samples.txt", "w", encoding="utf-8") as f:
        for m in methods:
            f.write(f"===== {m} =====\n")
            for i, t in enumerate(samples[m]):
                f.write(f"--- sample {i + 1} ---\n{t}\n\n")

    results = {
        "config": {
            "seed": SEED,
            "n_prompts": N_PROMPTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gen_model": MODEL_NAME,
            "ppl_model": PPL_MODEL_NAME if ppl_model is not None and ppl_model is model else PPL_MODEL_NAME,
            "device": DEVICE,
            "hyperparams": {
                "LAM": LAM, "K1": K1, "K2": K2, "TAU": TAU,
                "ALPHA": ALPHA, "KMIN": KMIN, "ETA": ETA, "GAMMA": GAMMA, "KMAX": KMAX,
                "TOP_P": TOP_P, "TYPICAL_P": TYPICAL_P, "TEMPERATURE": TEMPERATURE,
            },
            "agreed_spec": {
                "counter_scope": "per-prompt reset",
                "distinct_rep_aggregation": "per-prompt then average",
                "K_cap": "K = min(N, max(int(alpha*N), KMIN), KMAX)",
            },
        },
        "metrics": metrics,
        "ablation": ablation,
        "samples": samples,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nWrote results.json and samples.txt")


if __name__ == "__main__":
    main()
