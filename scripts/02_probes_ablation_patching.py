"""
02_probes_ablation_patching.py -- GPU experiments for HeadGenome Paper I (run inside distrobox venv).

Per model, in one pass:
  Stage 1  independent head scores (NOT derived from the Delta probe):
            - induction score (Olsson et al.): repeated random-token sequence, attention to the token
              after the previous occurrence
            - retrieval score (Wu et al. style): fraction of NIAH trials in which the head's top-1
              attention at the answer position is the needle token (+ mean needle attention mass)
           -> validates / falsifies the canonical Delta-based labels.
  Stage 2  ablation grid on the SAME model: dense / Retrieval / Induction / both / Sink, each vs.
           layer-matched random head sets (3 seeds) and vs top-k by independent scores;
           zero-ablation AND mean-ablation; metrics = WikiText-2 PPL and NIAH accuracy (single-token
           needle, next-token exact match). 
  Stage 3  activation patching (noising + denoising) of head outputs between clean/corrupt NIAH pairs.

Output: results/02_<model>.json
"""
import argparse, json, os, random, time, math
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper1_headgenome_mechinterp", "results")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "GPT-2": ("openai-community/gpt2-medium", torch.float32),
    "Qwen-0.5B": ("Qwen/Qwen2.5-0.5B", torch.bfloat16),
    "Qwen-1.5B": ("Qwen/Qwen2.5-1.5B", torch.bfloat16),
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


class HeadHooks:
    """Forward-pre-hooks on the attention output projection; the input is the concatenation of head outputs."""
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


def load(name):
    mid, dt = MODELS[name]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt, attn_implementation="eager").cuda().eval()
    cfg = model.config
    H = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    L = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    hid = getattr(cfg, "hidden_size", None) or cfg.n_embd
    return tok, model, L, H, hid // H


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
    c, per, probs = 0, {}, []
    for t in trials:
        lg = model(t["ids"].cuda()).logits[0, -1].float()
        ok = int(lg.argmax().item() == t["target"])
        c += ok
        per.setdefault(t["depth"], []).append(ok)
        probs.append(float(torch.softmax(lg, -1)[t["target"]]))
    return dict(acc=c / len(trials), n=len(trials), by_depth={str(k): float(np.mean(v)) for k, v in per.items()},
                mean_target_prob=float(np.mean(probs)))


@torch.no_grad()
def ppl(model, windows):
    nll, n = 0.0, 0
    for w in windows:
        w = w.cuda()
        out = model(w, labels=w)
        nll += out.loss.float().item() * (w.shape[1] - 1)
        n += w.shape[1] - 1
    return math.exp(nll / n)


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
    hk.record = {}
    for w in windows:
        model(w.cuda())
        for l, x in hk.record.items():
            sums[l] = sums.get(l, 0) + x.float().sum((0, 1))
        cnt += w.shape[1]
    hk.record = None
    return {l: (s / cnt) for l, s in sums.items()}


# ------------------------------------------------------------------ Stage 1: independent scores
@torch.no_grad()
def induction_scores(model, tok, L, H, Lseq=48, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    vocab = tok.vocab_size
    base = torch.randint(1000, min(vocab, 30000), (B, Lseq), generator=g)
    ids = torch.cat([base, base], dim=1).cuda()
    att = model(ids, output_attentions=True).attentions
    sc = np.zeros((L, H))
    qi = torch.arange(Lseq, 2 * Lseq - 1).cuda()
    for l in range(L):
        a = att[l].float()  # B,H,T,T
        sc[l] = a[:, :, qi, qi - Lseq + 1].mean((0, 2)).cpu().numpy()
    return sc


@torch.no_grad()
def retrieval_scores(model, trials, L, H):
    hit, mass = np.zeros((L, H)), np.zeros((L, H))
    for t in trials:
        att = model(t["ids"].cuda(), output_attentions=True).attentions
        for l in range(L):
            last = att[l][0, :, -1, :].float()  # H,T
            hit[l] += (last.argmax(-1) == t["needle_pos"]).float().cpu().numpy()
            mass[l] += last[:, t["needle_pos"]].cpu().numpy()
    return hit / len(trials), mass / len(trials)


def validate_scores(canon, ind, rh, rm, L, H):
    lab = np.array([[canon.get(f"{l}_{h}", {}).get("label", "local") for h in range(H)] for l in range(L)])
    dl = np.array([[canon.get(f"{l}_{h}", {}).get("delta", 0.0) for h in range(H)] for l in range(L)])
    out = {}
    for cls, sc, nm in [("induction", ind, "induction_score"), ("retrieval", rh, "retrieval_hit"), ("retrieval", rm, "retrieval_mass")]:
        y = (lab == cls).flatten().astype(int)
        s = sc.flatten()
        out[f"AUC_{cls}_label_vs_{nm}"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
        out[f"mean_{nm}_by_label"] = {c: float(sc[lab == c].mean()) for c in ["sink", "local", "retrieval", "induction"] if (lab == c).any()}
    out["spearman_delta_vs_induction_score"] = float(stats.spearmanr(dl.flatten(), ind.flatten())[0])
    out["spearman_delta_vs_retrieval_hit"] = float(stats.spearmanr(dl.flatten(), rh.flatten())[0])
    # confusion between canonical labels and standard-metric labels
    std_ind = ind >= 0.3
    std_ret = rh >= 0.2
    out["standard_induction_heads(score>=0.3)"] = int(std_ind.sum())
    out["standard_retrieval_heads(hit>=0.2)"] = int(std_ret.sum())
    out["canonical_induction_in_standard_induction"] = int(((lab == "induction") & std_ind).sum())
    out["canonical_retrieval_in_standard_retrieval"] = int(((lab == "retrieval") & std_ret).sum())
    out["standard_induction_labelled_as"] = {c: int(((lab == c) & std_ind).sum()) for c in ["sink", "local", "retrieval", "induction"]}
    out["standard_retrieval_labelled_as"] = {c: int(((lab == c) & std_ret).sum()) for c in ["sink", "local", "retrieval", "induction"]}
    return out, lab


# ------------------------------------------------------------------ head sets
def heads_of(lab, cls):
    return [(l, h) for l in range(lab.shape[0]) for h in range(lab.shape[1]) if lab[l, h] == cls]


def layer_matched_random(target, L, H, seed, exclude=()):
    rng = random.Random(seed)
    excl = set(exclude) | set(target)
    out = []
    for (l, h) in target:
        cands = [(l, x) for x in range(H) if (l, x) not in excl and (l, x) not in out]
        out.append(rng.choice(cands) if cands else (l, h))
    return out


def topk(score, k):
    L, H = score.shape
    flat = np.argsort(-score.flatten())[:k]
    return [(int(i // H), int(i % H)) for i in flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    tok, model, L, H, hd = load(a.model)
    hk = HeadHooks(model, hd)
    canon = json.load(open(f"{ROOT}/outputs/canonical_labels.json"))["models"][a.model]["heads"]
    words = single_token_words(tok)
    n_pd = 4 if a.quick else 20
    trials = make_trials(tok, words, n_pd, 70, 123)
    score_trials = make_trials(tok, words, 3 if a.quick else 8, 35, 7)
    windows = wiki_windows(tok, 4 if a.quick else 20, 512)
    calib = wiki_windows(tok, 8 if a.quick else 24, 512, split="train")
    R = dict(model=a.model, L=L, H=H, head_dim=hd, n_trials=len(trials), n_words=len(words),
             prompt_tokens=int(trials[0]["ids"].shape[1]))
    print(a.model, R, flush=True)

    # Stage 1
    ind = induction_scores(model, tok, L, H)
    rh, rm = retrieval_scores(model, score_trials, L, H)
    val, lab = validate_scores(canon, ind, rh, rm, L, H)
    R["stage1_independent_scores"] = val
    R["stage1_scores"] = dict(induction=ind.tolist(), retrieval_hit=rh.tolist(), retrieval_mass=rm.tolist())
    print("stage1 done", json.dumps({k: v for k, v in val.items() if k.startswith("AUC")}), flush=True)

    # head sets
    ret, indh, sink = heads_of(lab, "retrieval"), heads_of(lab, "induction"), heads_of(lab, "sink")
    sets = {"retrieval": ret, "induction": indh, "both": ret + indh}
    if sink:
        sets["sink"] = sink
    R["set_sizes"] = {k: len(v) for k, v in sets.items()}
    configs = {"dense": []}
    for k, v in sets.items():
        configs[k] = v
        for s in range(3):
            excl = ret + indh if k in ("retrieval", "induction", "both") else sink
            configs[f"rand_{k}_s{s}"] = layer_matched_random(v, L, H, 100 + s, exclude=[] if k == "sink" else [])
    if ret:
        configs["topk_retrieval_score"] = topk(rh + 1e-6 * rm, len(ret))
    if indh:
        configs["topk_induction_score"] = topk(ind, len(indh))
    if ret and indh:
        configs["topk_both_scores"] = list(dict.fromkeys(topk(rh + 1e-6 * rm, len(ret)) + topk(ind, len(indh))))
    R["config_sizes"] = {k: len(v) for k, v in configs.items()}

    # Stage 2
    means = mean_activations(model, hk, calib, L)
    hk.means = means
    R["stage2_ablation"] = {}
    for mode in ["mean", "zero"]:
        R["stage2_ablation"][mode] = {}
        for name, hs in configs.items():
            hk.set(hs, mode)
            r = niah_acc(model, trials)
            r["ppl"] = ppl(model, windows)
            R["stage2_ablation"][mode][name] = r
            print(f"[{mode}] {name:24s} n={len(hs):3d} NIAH={r['acc']:.2f} PPL={r['ppl']:.2f}", flush=True)
            hk.off()
        json.dump(R, open(f"{OUT}/02_{a.model}.json", "w"), indent=1)

    # Stage 3: activation patching (clean = needle word A, corrupt = needle word B, same layout)
    rng = random.Random(5)
    pairs = []
    for _ in range(10 if a.quick else 60):
        d = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        (w1, i1), (w2, i2) = rng.sample(words, 2)
        s = rng.randint(0, 10**9)
        c = tok(niah_prompt(d, w1, s, 70), return_tensors="pt").input_ids
        k = tok(niah_prompt(d, w2, s, 70), return_tensors="pt").input_ids
        if c.shape == k.shape:
            pairs.append((c, k, i1, i2))
    patch_sets = {k: v for k, v in configs.items() if k in
                  ("retrieval", "induction", "both", "sink", "topk_retrieval_score", "topk_induction_score")
                  or k.startswith("rand_retrieval") or k.startswith("rand_induction") or k.startswith("rand_both")}
    R["stage3_patching"] = {}

    @torch.no_grad()
    def ld(ids, i1, i2):
        lg = model(ids.cuda()).logits[0, -1].float()
        return float(lg[i1] - lg[i2])
    for direction in ["noising", "denoising"]:
        res = {n: [] for n in patch_sets}
        for (c, k, i1, i2) in pairs:
            hk.off()
            hk.record = {}
            ld_c = ld(c, i1, i2)
            rec_c = hk.record
            hk.record = {}
            ld_k = ld(k, i1, i2)
            rec_k = hk.record
            hk.record = None
            if not (ld_c > 0 > ld_k):
                continue  # need both runs to be 'correct' for their own needle
            for n, hs in patch_sets.items():
                if direction == "noising":     # run clean, patch corrupt head outputs in
                    hk.patch = rec_k; hk.set(hs, "patch"); v = ld(c, i1, i2); hk.off()
                    res[n].append((ld_c - v) / (ld_c - ld_k))      # fraction of the clean->corrupt effect induced
                else:                           # run corrupt, patch clean head outputs in
                    hk.patch = rec_c; hk.set(hs, "patch"); v = ld(k, i1, i2); hk.off()
                    res[n].append((v - ld_k) / (ld_c - ld_k))      # fraction of the corrupt->clean effect recovered
        R["stage3_patching"][direction] = {n: dict(mean=float(np.mean(v)) if v else None,
                                                    sd=float(np.std(v)) if v else None, n_pairs=len(v))
                                           for n, v in res.items()}
        print(direction, {n: round(float(np.mean(v)), 3) for n, v in res.items() if v}, flush=True)
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(f"{OUT}/02_{a.model}.json", "w"), indent=1)
    print("saved", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
