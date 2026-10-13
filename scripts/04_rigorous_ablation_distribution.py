"""
04_rigorous_ablation_distribution.py -- Rigorous Ablation & Random Control Distribution Suite:
  1. Evaluates N=100 NIAH trials (20 trials per depth across 5 depths) with exact Wilson Score 95% CIs.
  2. Evaluates N_draws=20 random head sets per condition:
     - rand_unrestricted: random layer-matched heads (may sample Sinks/Induction)
     - rand_local_only: random layer-matched heads strictly sampled from Local heads
       (guarantees NO Sink, NO Retrieval, NO Induction heads are ever touched).
  3. Reports full empirical distributions (mean, std, 5th percentile, median, 95th percentile).
  4. Runs both mean-ablation and zero-ablation.

Output: paper1_headgenome_mechinterp/results/04_rigorous_ablation_<model>.json
"""
import argparse, json, os, random, time, math
import numpy as np
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
        self.hd, self.sel, self.mode, self.means = hd, {}, None, None
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
    n = len(trials)
    acc = c / n
    return dict(
        acc=acc,
        n=n,
        ci_95=wilson_ci(c, n),
        by_depth={str(k): round(float(np.mean(v)), 3) for k, v in per.items()}
    )


@torch.no_grad()
def ppl(model, windows):
    nll, n = 0.0, 0
    for w in windows:
        w = w.cuda()
        out = model(w, labels=w)
        nll += out.loss.float().item() * (w.shape[1] - 1)
        n += w.shape[1] - 1
    return float(math.exp(nll / n)) if n > 0 else 0.0


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
def mean_activations(model, hk, windows, L):
    sums, cnt = {}, 0
    record = {}
    
    def rec_hook(l):
        def h(mod, args):
            record[l] = args[0].detach().clone()
        return h
    
    handles = []
    for l, layer in enumerate(hk.layers(model)):
        handles.append(hk.oproj(layer).register_forward_pre_hook(rec_hook(l)))
        
    for w in windows:
        model(w.cuda())
        for l, x in record.items():
            sums[l] = sums.get(l, 0) + x.float().sum((0, 1))
        cnt += w.shape[1]
        
    for h in handles:
        h.remove()
        
    return {l: (s / cnt) for l, s in sums.items()}


def sample_layer_matched(target_heads, L, H, rng, allowed_pool=None):
    """Samples layer-matched heads. If allowed_pool is provided, samples strictly from allowed_pool."""
    out = []
    for (l, h) in target_heads:
        if allowed_pool is not None:
            cands = [(l, x) for x in range(H) if (l, x) in allowed_pool and (l, x) not in out]
        else:
            cands = [(l, x) for x in range(H) if (l, x) not in target_heads and (l, x) not in out]
        out.append(rng.choice(cands) if cands else (l, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n_draws", type=int, default=20, help="Number of random draws per condition")
    ap.add_argument("--n_trials_per_depth", type=int, default=20, help="20 per depth * 5 depths = 100 trials")
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

    # Extract classified sets
    lab_map = {(l, h): canon.get(f"{l}_{h}", {}).get("label", "local") for l in range(L) for h in range(H)}
    ret = [(l, h) for (l, h), lab in lab_map.items() if lab == "retrieval"]
    indh = [(l, h) for (l, h), lab in lab_map.items() if lab == "induction"]
    sink = [(l, h) for (l, h), lab in lab_map.items() if lab == "sink"]
    local_pool = set((l, h) for (l, h), lab in lab_map.items() if lab == "local")

    results = {
        "model": a.model,
        "n_trials": len(trials),
        "n_draws": a.n_draws,
        "set_sizes": {"retrieval": len(ret), "induction": len(indh), "both": len(ret) + len(indh), "sink": len(sink), "local": len(local_pool)}
    }
    print(f"\n=== Evaluating {a.model} (N_trials={len(trials)}, N_draws={a.n_draws}) ===", flush=True)

    # 1. Dense Baseline
    hk.off()
    dense_r = niah_acc(model, trials)
    dense_r["ppl"] = ppl(model, windows)
    results["dense"] = dense_r
    print(f"DENSE: NIAH={dense_r['acc']:.3f} {dense_r['ci_95']} PPL={dense_r['ppl']:.2f}", flush=True)

    target_sets = {
        "retrieval": ret,
        "induction": indh,
        "both": ret + indh,
    }
    if sink:
        target_sets["sink"] = sink

    for mode in ["mean", "zero"]:
        results[mode] = {}
        print(f"\n--- Running Mode: {mode.upper()} ---", flush=True)
        for name, hs in target_sets.items():
            if not hs:
                continue
            # Evaluate target set
            hk.set(hs, mode)
            res = niah_acc(model, trials)
            res["ppl"] = ppl(model, windows)
            hk.off()
            print(f"[{mode}] {name.upper()} (n={len(hs)}): NIAH={res['acc']:.3f} {res['ci_95']} PPL={res['ppl']:.2f}", flush=True)

            # Evaluate Distribution: Unrestricted Random (can hit Sinks/Induction)
            unres_accs, unres_ppls = [], []
            print(f"   [Unrestricted Random] Running {a.n_draws} draws...", flush=True)
            for seed in range(a.n_draws):
                rng = random.Random(1000 + seed)
                r_hs = sample_layer_matched(hs, L, H, rng, allowed_pool=None)
                hk.set(r_hs, mode)
                r_niah = niah_acc(model, trials)["acc"]
                if seed < 3:
                    r_ppl = ppl(model, windows[:5])
                    unres_ppls.append(r_ppl)
                unres_accs.append(r_niah)
                hk.off()
                print(f"      unres {seed+1}/{a.n_draws}: niah={r_niah:.2f}", flush=True)

            # Evaluate Distribution: Local-Only Random (strictly drawn from Local heads)
            loc_accs, loc_ppls = [], []
            print(f"   [Local-Only Random] Running {a.n_draws} draws...", flush=True)
            for seed in range(a.n_draws):
                rng = random.Random(2000 + seed)
                r_hs = sample_layer_matched(hs, L, H, rng, allowed_pool=local_pool)
                hk.set(r_hs, mode)
                r_niah = niah_acc(model, trials)["acc"]
                if seed < 3:
                    r_ppl = ppl(model, windows[:5])
                    loc_ppls.append(r_ppl)
                loc_accs.append(r_niah)
                hk.off()
                print(f"      local {seed+1}/{a.n_draws}: niah={r_niah:.2f}", flush=True)

            results[mode][name] = {
                "target": res,
                "rand_unrestricted_dist": {
                    "mean_acc": round(float(np.mean(unres_accs)), 3),
                    "std_acc": round(float(np.std(unres_accs)), 3),
                    "median_acc": round(float(np.median(unres_accs)), 3),
                    "p5_acc": round(float(np.percentile(unres_accs, 5)), 3),
                    "p95_acc": round(float(np.percentile(unres_accs, 95)), 3),
                    "mean_ppl": round(float(np.mean(unres_ppls)), 2) if unres_ppls else None,
                    "std_ppl": round(float(np.std(unres_ppls)), 2) if unres_ppls else None,
                    "raw_accs": unres_accs
                },
                "rand_local_only_dist": {
                    "mean_acc": round(float(np.mean(loc_accs)), 3),
                    "std_acc": round(float(np.std(loc_accs)), 3),
                    "median_acc": round(float(np.median(loc_accs)), 3),
                    "p5_acc": round(float(np.percentile(loc_accs, 5)), 3),
                    "p95_acc": round(float(np.percentile(loc_accs, 95)), 3),
                    "mean_ppl": round(float(np.mean(loc_ppls)), 2) if loc_ppls else None,
                    "std_ppl": round(float(np.std(loc_ppls)), 2) if loc_ppls else None,
                    "raw_accs": loc_accs
                }
            }
            u_d = results[mode][name]["rand_unrestricted_dist"]
            l_d = results[mode][name]["rand_local_only_dist"]
            print(f"     => Unrestricted Rand ({a.n_draws} draws): Mean NIAH={u_d['mean_acc']:.3f} (med {u_d['median_acc']:.3f}) | Mean PPL={u_d['mean_ppl']}", flush=True)
            print(f"     => Local-Only   Rand ({a.n_draws} draws): Mean NIAH={l_d['mean_acc']:.3f} (med {l_d['median_acc']:.3f}) | Mean PPL={l_d['mean_ppl']}", flush=True)

    results["runtime_s"] = time.time() - t0
    out_file = f"{OUT}/04_rigorous_ablation_{a.model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file} in {results['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
