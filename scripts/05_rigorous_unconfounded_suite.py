"""
05_rigorous_unconfounded_suite.py
Complete, unconfounded evaluation suite resolving all reviewer critiques:
  1. Unconfounded Dispersed Control: Samples k heads uniformly across layers with per-layer cap (max 2 heads/layer),
     preventing the layer-13/15 capacity bottleneck.
  2. Layer-Matched Unrestricted Control: 20 draws, tracking exact sinks hit per draw.
  3. Sink Regression: Direct linear regression and Spearman correlation of NIAH vs Sinks Hit across random draws.
  4. Wu et al. Retrieval vs Canonical Retrieval Ablation.
  5. Olsson Induction vs Canonical Induction Ablation.
  6. Activation Patching Causal Recovery: Direct causal recovery of factual logit difference.
  7. Exact Wilson Score 95% Confidence Intervals on all conditions (N=100 trials across 5 depths).
  8. Standardized PPL protocol on WikiText-2 test set (15 windows of 512 tokens).

Outputs: paper1_headgenome_mechinterp/results/05_rigorous_<model>.json
"""
import argparse, json, math, os, random, time
import numpy as np
from scipy import stats
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper1_headgenome_mechinterp", "results")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "Qwen-0.5B": ("Qwen/Qwen2.5-0.5B", torch.bfloat16),
    "GPT-2": ("openai-community/gpt2-medium", torch.float32),
    "Llama-3.2-1B": ("unsloth/Llama-3.2-1B", torch.bfloat16),
}

SENT = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we work and live.",
    "The weather today is surprisingly sunny despite the forecast.",
    "A healthy diet consists of fruits, vegetables, and lean proteins.",
    "The city skyline looks beautiful at night when all the lights are on.",
    "Many people find programming to be both challenging and rewarding.",
    "Music has the power to evoke deep emotions and bring people together.",
    "Exploring the outdoors is a great way to relieve stress.",
    "The history of ancient civilizations continues to fascinate historians.",
    "Global markets rallied today following positive economic news.",
    "Deep learning models require significant computational resources.",
    "Space exploration has revealed thousands of exoplanets in our galaxy.",
    "Classical architecture emphasizes symmetry, proportion, and geometry.",
    "The recipe calls for a dash of cinnamon and a cup of sugar.",
]

WORDS = ["Amber", "Basil", "Cobalt", "Dagger", "Ember", "Falcon", "Garnet", "Harbor", "Indigo", "Jasper",
         "Kestrel", "Lantern", "Marble", "Nectar", "Obsidian", "Pepper", "Quartz", "Raven", "Saffron",
         "Tundra", "Umber", "Velvet", "Walnut", "Zephyr", "Willow", "Copper", "Maple", "Coral", "Silver", "Thunder"]


def wilson_ci(k, n, z=1.95996):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


class HeadHooks:
    def __init__(self, model, hd):
        self.hd, self.sel, self.mode, self.means, self.patch, self.record = hd, {}, None, None, None, None
        self.handles = []
        for l, layer in enumerate(self.layers(model)):
            self.handles.append(self.oproj(layer).register_forward_pre_hook(self._make(l)))

    @staticmethod
    def layers(model):
        return model.transformer.h if hasattr(model, "transformer") else model.model.layers

    @staticmethod
    def oproj(layer):
        return layer.attn.c_proj if hasattr(layer, "attn") else layer.self_attn.o_proj

    def _make(self, l):
        def hook(mod, args):
            x = args[0]
            if self.record is not None:
                self.record[l] = x.detach().clone()
            sel = self.sel.get(l)
            if not sel or self.mode is None:
                return None
            x = x.clone()
            for h in sel:
                sl = slice(h * self.hd, (h + 1) * self.hd)
                if self.mode == "zero":
                    x[..., sl] = 0
                elif self.mode == "mean":
                    x[..., sl] = self.means[l][sl].to(x.dtype)
                elif self.mode == "patch":
                    x[..., sl] = self.patch[l][..., sl]
            return (x,)
        return hook

    def set(self, heads, mode):
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        self.sel, self.mode = d, (mode if heads else None)

    def off(self):
        self.sel, self.mode = {}, None


def single_token_words(tok):
    out = []
    for w in WORDS:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append((w, ids[0]))
    return out


def niah_prompt(depth, word, seed, n_sent):
    rng = random.Random(seed)
    s = [rng.choice(SENT) for _ in range(n_sent)]
    s.insert(int(n_sent * depth), f"The secret word is {word}.")
    return " ".join(s) + "\n\nQuestion: What is the secret word?\nAnswer: The secret word is"


def make_trials(tok, words, n_per_depth, n_sent, seed0):
    trials, rng = [], random.Random(seed0)
    for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for i in range(n_per_depth):
            w, wid = rng.choice(words)
            p = niah_prompt(d, w, rng.randint(0, 10**9), n_sent)
            ids = tok(p, return_tensors="pt").input_ids
            pos = (ids[0] == wid).nonzero().flatten()
            if len(pos) == 0:
                continue
            trials.append(dict(depth=d, ids=ids, target=wid, needle_pos=int(pos[0]), word=w))
    return trials


@torch.no_grad()
def niah_acc(model, trials):
    c, per = 0, {}
    for t in trials:
        lg = model(t["ids"].cuda()).logits[0, -1].float()
        ok = int(lg.argmax().item() == t["target"])
        c += ok
        per.setdefault(t["depth"], []).append(ok)
    acc = c / len(trials)
    return {
        "acc": round(acc, 4),
        "ci_95": wilson_ci(c, len(trials)),
        "n": len(trials),
        "by_depth": {str(k): round(float(np.mean(v)), 4) for k, v in per.items()}
    }


def wiki_windows(tok, n, T, split="test"):
    from datasets import load_dataset
    try:
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        d = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(d["text"][: min(len(d["text"]), n * 50 + 200)])
    ids = tok(text, return_tensors="pt").input_ids[0]
    if len(ids) < n * T:
        text = "\n\n".join(d["text"])
        ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * T:(i + 1) * T].unsqueeze(0) for i in range(min(n, len(ids) // T))]


@torch.no_grad()
def ppl(model, windows):
    nll, n = 0.0, 0
    for w in windows:
        w = w.cuda()
        out = model(w, labels=w)
        nll += out.loss.float().item() * (w.shape[1] - 1)
        n += w.shape[1] - 1
    return round(float(math.exp(nll / n)), 2) if n > 0 else 0.0


@torch.no_grad()
def mean_activations(model, hk, windows, L):
    sums, cnt = {}, 0
    hk.record = {}
    for w in windows:
        model(w.cuda())
        for l, x in hk.record.items():
            sums[l] = sums.get(l, 0) + x.float().sum((0, 1))
        cnt += w.shape[1]
    hk.record = None
    return {l: (s / cnt) for l, s in sums.items()}


# --- Random sampling strategies ---
def sample_globally_dispersed(k, L, H, rng, max_per_layer=None):
    """Samples k heads uniformly across all layers, with a strict per-layer cap."""
    if max_per_layer is None:
        max_per_layer = max(2, math.ceil(k / L) + 1)
    counts = {l: 0 for l in range(L)}
    all_heads = [(l, h) for l in range(L) for h in range(H)]
    rng.shuffle(all_heads)
    chosen = []
    for (l, h) in all_heads:
        if counts[l] < max_per_layer:
            chosen.append((l, h))
            counts[l] += 1
            if len(chosen) == k:
                break
    return chosen


def sample_layer_matched(target_heads, L, H, rng, allowed_pool=None):
    """Samples layer-matched heads. If allowed_pool is provided, restricts to allowed_pool."""
    out = []
    for (l, h) in target_heads:
        if allowed_pool is not None:
            cands = [(l, x) for x in range(H) if (l, x) in allowed_pool and (l, x) not in out]
        else:
            cands = [(l, x) for x in range(H) if (l, x) not in target_heads and (l, x) not in out]
        out.append(rng.choice(cands) if cands else (l, h))
    return out


# --- Metric Computations ---
@torch.no_grad()
def compute_standard_scores(model, tok, trials, L, H):
    # 1. Olsson Induction Score
    g = torch.Generator().manual_seed(0)
    vocab = min(tok.vocab_size, 30000)
    base = torch.randint(1000, vocab, (8, 48), generator=g)
    ids = torch.cat([base, base], dim=1).cuda()
    att = model(ids, output_attentions=True).attentions
    ind_sc = np.zeros((L, H))
    qi = torch.arange(48, 95).cuda()
    for l in range(L):
        a = att[l].float()
        ind_sc[l] = a[:, :, qi, qi - 48 + 1].mean((0, 2)).cpu().numpy()

    # 2. Wu Retrieval Hit Score
    ret_sc = np.zeros((L, H))
    for t in trials[:20]:
        att = model(t["ids"].cuda(), output_attentions=True).attentions
        for l in range(L):
            last = att[l][0, :, -1, :].float()
            ret_sc[l] += (last.argmax(-1) == t["needle_pos"]).float().cpu().numpy()
    ret_sc /= 20.0
    return ind_sc, ret_sc


def topk_heads(score_matrix, k):
    flat = score_matrix.flatten()
    idx = np.argsort(flat)[-k:]
    H = score_matrix.shape[1]
    return [(int(i // H), int(i % H)) for i in reversed(idx)]


# --- Activation Patching ---
@torch.no_grad()
def eval_activation_patching(model, hk, tok, words, target_head_sets, n_pairs=20):
    """Measures % recovery of clean target logit under corrupted context when candidate heads are patched."""
    rng = random.Random(42)
    pairs = []
    for _ in range(n_pairs):
        w_clean, id_clean = rng.choice(words)
        w_corr, id_corr = rng.choice([w for w in words if w[0] != w_clean])
        p_clean = niah_prompt(0.5, w_clean, rng.randint(0, 10**9), 30)
        p_corr = niah_prompt(0.5, w_corr, rng.randint(0, 10**9), 30)
        ids_clean = tok(p_clean, return_tensors="pt").input_ids
        ids_corr = tok(p_corr, return_tensors="pt").input_ids
        pairs.append((ids_clean, ids_corr, id_clean, id_corr))

    # Clean & Corrupted baselines
    clean_diffs, corr_diffs = [], []
    clean_acts_list = []
    for (ic, io, tc, to) in pairs:
        hk.off()
        hk.record = {}
        lg_c = model(ic.cuda()).logits[0, -1].float()
        clean_acts = {l: v.clone() for l, v in hk.record.items()}
        clean_acts_list.append(clean_acts)
        hk.record = None
        clean_diffs.append(float(lg_c[tc] - lg_c[to]))

        lg_o = model(io.cuda()).logits[0, -1].float()
        corr_diffs.append(float(lg_o[tc] - lg_o[to]))

    mean_clean_diff = float(np.mean(clean_diffs))
    mean_corr_diff = float(np.mean(corr_diffs))
    total_gap = mean_clean_diff - mean_corr_diff

    recovery = {}
    for name, heads in target_head_sets.items():
        patched_diffs = []
        for i, (ic, io, tc, to) in enumerate(pairs):
            hk.patch = clean_acts_list[i]
            hk.set(heads, mode="patch")
            lg_p = model(io.cuda()).logits[0, -1].float()
            hk.off()
            patched_diffs.append(float(lg_p[tc] - lg_p[to]))
        mean_p_diff = float(np.mean(patched_diffs))
        pct = (mean_p_diff - mean_corr_diff) / total_gap if abs(total_gap) > 1e-6 else 0.0
        recovery[name] = round(float(pct) * 100, 2)
    return recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n_draws", type=int, default=20)
    ap.add_argument("--n_trials_per_depth", type=int, default=20)
    a = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    mid, dt = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt, attn_implementation="eager").cuda().eval()
    cfg = model.config
    H = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    L = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    hid = getattr(cfg, "hidden_size", None) or cfg.n_embd
    hd = hid // H
    hk = HeadHooks(model, hd)

    canon = json.load(open(f"{ROOT}/outputs/canonical_labels.json"))["models"][a.model]["heads"]
    words = single_token_words(tok)
    trials = make_trials(tok, words, a.n_trials_per_depth, 70, 123)
    windows = wiki_windows(tok, 15, 512)
    calib = wiki_windows(tok, 15, 512, split="train")
    means = mean_activations(model, hk, calib, L)
    hk.means = means

    lab_map = {(l, h): canon.get(f"{l}_{h}", {}).get("label", "local") for l in range(L) for h in range(H)}
    ret = [(l, h) for (l, h), lab in lab_map.items() if lab == "retrieval"]
    indh = [(l, h) for (l, h), lab in lab_map.items() if lab == "induction"]
    sink = [(l, h) for (l, h), lab in lab_map.items() if lab == "sink"]
    local_pool = set((l, h) for (l, h), lab in lab_map.items() if lab == "local")

    # Compute independent scores for Wu and Olsson
    print("Computing Olsson and Wu scores...", flush=True)
    olsson_mat, wu_mat = compute_standard_scores(model, tok, trials, L, H)
    top_olsson = topk_heads(olsson_mat, len(indh))
    top_wu = topk_heads(wu_mat, max(len(ret), 4))

    results = {
        "model": a.model,
        "L": L, "H": H, "N_trials": len(trials), "N_draws": a.n_draws,
        "head_counts": {
            "canonical_retrieval": len(ret),
            "canonical_induction": len(indh),
            "canonical_both": len(ret) + len(indh),
            "canonical_sink": len(sink),
            "top_wu_retrieval": len(top_wu),
            "top_olsson_induction": len(top_olsson),
        }
    }
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING {a.model} (L={L}, H={H}, N={len(trials)})", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Dense Baseline
    dense_r = niah_acc(model, trials)
    dense_r["ppl"] = ppl(model, windows)
    results["dense_baseline"] = dense_r
    print(f"DENSE: NIAH={dense_r['acc']:.3f} {dense_r['ci_95']} PPL={dense_r['ppl']:.2f}\n", flush=True)

    # 2. Designated Ablations (Mean & Zero)
    results["ablations"] = {}
    eval_groups = {
        "canonical_retrieval": ret,
        "top_wu_retrieval": top_wu,
        "canonical_induction": indh,
        "top_olsson_induction": top_olsson,
        "canonical_both": ret + indh,
    }
    if sink:
        eval_groups["canonical_sink"] = sink

    for mode in ["mean", "zero"]:
        results["ablations"][mode] = {}
        print(f"--- Ablation Mode: {mode.upper()} ---", flush=True)
        for gname, hs in eval_groups.items():
            if not hs:
                continue
            hk.set(hs, mode)
            res = niah_acc(model, trials)
            res["ppl"] = ppl(model, windows)
            hk.off()
            results["ablations"][mode][gname] = res
            print(f"  [{mode:4s}] {gname:22s} (k={len(hs):2d}): NIAH={res['acc']:.3f} {res['ci_95']} PPL={res['ppl']:.2f}", flush=True)

    # 3. Rigorous Random Controls for Induction (k=len(indh)) under Mean Ablation
    k_ind = len(indh)
    print(f"\n--- Random Controls for Induction (k={k_ind} heads, mean mode) ---", flush=True)
    ctrl_results = {}

    # Control A: Unconfounded Globally Dispersed (Capped per layer)
    print("-> Running Control A: Globally Dispersed (max 2 heads/layer)...", flush=True)
    disp_accs = []
    for s in range(a.n_draws):
        rng = random.Random(2000 + s)
        hs = sample_globally_dispersed(k_ind, L, H, rng, max_per_layer=max(2, math.ceil(k_ind / L) + 1))
        hk.set(hs, "mean")
        acc = niah_acc(model, trials)["acc"]
        hk.off()
        disp_accs.append(acc)
    ctrl_results["dispersed_capped"] = {
        "mean": round(float(np.mean(disp_accs)), 4),
        "median": round(float(np.median(disp_accs)), 4),
        "std": round(float(np.std(disp_accs)), 4),
        "min": round(float(np.min(disp_accs)), 4),
        "max": round(float(np.max(disp_accs)), 4),
        "draws": disp_accs
    }
    print(f"   Dispersed: Mean={ctrl_results['dispersed_capped']['mean']:.3f} "
          f"Med={ctrl_results['dispersed_capped']['median']:.3f} "
          f"Range=[{ctrl_results['dispersed_capped']['min']:.3f}, {ctrl_results['dispersed_capped']['max']:.3f}]", flush=True)

    # Control B: Layer-Matched Unrestricted (tracks sinks hit & regression)
    print("-> Running Control B: Layer-Matched Unrestricted (tracking sinks hit)...", flush=True)
    unres_accs, unres_sinks = [], []
    for s in range(a.n_draws):
        rng = random.Random(1000 + s)
        hs = sample_layer_matched(indh, L, H, rng, allowed_pool=None)
        hk.set(hs, "mean")
        acc = niah_acc(model, trials)["acc"]
        hk.off()
        sinks_hit = sum(1 for (l, h) in hs if (l, h) in sink)
        unres_accs.append(acc)
        unres_sinks.append(sinks_hit)
    
    # Regression
    reg = stats.linregress(unres_sinks, unres_accs)
    spear = stats.spearmanr(unres_sinks, unres_accs)
    ctrl_results["layer_matched_unrestricted"] = {
        "mean": round(float(np.mean(unres_accs)), 4),
        "median": round(float(np.median(unres_accs)), 4),
        "std": round(float(np.std(unres_accs)), 4),
        "min": round(float(np.min(unres_accs)), 4),
        "max": round(float(np.max(unres_accs)), 4),
        "draws": unres_accs,
        "sinks_hit_per_draw": unres_sinks,
        "regression_on_sinks": {
            "slope": round(float(reg.slope), 4),
            "intercept": round(float(reg.intercept), 4),
            "r_value": round(float(reg.rvalue), 4),
            "r_squared": round(float(reg.rvalue**2), 4),
            "p_value": float(reg.pvalue),
            "spearman_rho": round(float(spear.correlation), 4),
            "spearman_p": float(spear.pvalue)
        }
    }
    print(f"   Unrestricted: Mean={ctrl_results['layer_matched_unrestricted']['mean']:.3f} "
          f"Med={ctrl_results['layer_matched_unrestricted']['median']:.3f} "
          f"Range=[{ctrl_results['layer_matched_unrestricted']['min']:.3f}, {ctrl_results['layer_matched_unrestricted']['max']:.3f}]", flush=True)
    print(f"   Sink Regression: R²={ctrl_results['layer_matched_unrestricted']['regression_on_sinks']['r_squared']:.4f}, "
          f"p={ctrl_results['layer_matched_unrestricted']['regression_on_sinks']['p_value']:.4e}", flush=True)

    # Control C: Layer-Matched Local-Only
    print("-> Running Control C: Layer-Matched Local-Only...", flush=True)
    loc_accs = []
    for s in range(a.n_draws):
        rng = random.Random(3000 + s)
        hs = sample_layer_matched(indh, L, H, rng, allowed_pool=local_pool)
        hk.set(hs, "mean")
        acc = niah_acc(model, trials)["acc"]
        hk.off()
        loc_accs.append(acc)
    ctrl_results["layer_matched_local"] = {
        "mean": round(float(np.mean(loc_accs)), 4),
        "median": round(float(np.median(loc_accs)), 4),
        "std": round(float(np.std(loc_accs)), 4),
        "min": round(float(np.min(loc_accs)), 4),
        "max": round(float(np.max(loc_accs)), 4),
        "draws": loc_accs
    }
    print(f"   Local-Only:   Mean={ctrl_results['layer_matched_local']['mean']:.3f} "
          f"Med={ctrl_results['layer_matched_local']['median']:.3f} "
          f"Range=[{ctrl_results['layer_matched_local']['min']:.3f}, {ctrl_results['layer_matched_local']['max']:.3f}]", flush=True)
    results["random_controls_induction"] = ctrl_results

    # 4. Activation Patching Causal Recovery (%)
    print("\n--- Running Activation Patching Causal Recovery (N=20 pairs) ---", flush=True)
    patch_sets = {
        "canonical_retrieval": ret,
        "top_wu_retrieval": top_wu,
        "canonical_induction": indh,
        "top_olsson_induction": top_olsson,
    }
    patch_recovery = eval_activation_patching(model, hk, tok, words, patch_sets, n_pairs=20)
    results["activation_patching_recovery_pct"] = patch_recovery
    for name, rec in patch_recovery.items():
        print(f"  Recovery % [{name:22s}]: {rec:6.2f}%", flush=True)

    results["total_runtime_s"] = round(time.time() - t0, 2)
    out_file = os.path.join(OUT, f"05_rigorous_{a.model}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file} (elapsed: {results['total_runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
