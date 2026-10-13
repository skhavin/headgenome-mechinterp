"""
01_stats_audit.py  -- CPU-only audit of existing HeadGenome artifacts (Paper I).

Inputs  (read-only, from repo root):
  outputs/canonical_labels.json           canonical per-head labels + delta + match_entropy
  outputs/phase3/weight_features.json     static weight features per head
  outputs/phase1/robust_entropy_*.json    raw Delta values (single retrieval-style probe)
  outputs/phase6/ruler_comprehensive.json NIAH-by-depth under sparse windows
Output: paper1_headgenome_mechinterp/results/01_stats_audit.json
"""
import json, os, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             balanced_accuracy_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper1_headgenome_mechinterp", "results")
os.makedirs(OUT, exist_ok=True)
R = {}
rng = np.random.default_rng(0)

canon = json.load(open(f"{ROOT}/outputs/canonical_labels.json"))["models"]
wf = json.load(open(f"{ROOT}/outputs/phase3/weight_features.json"))["models"]
MODELS = ["GPT-2", "Qwen-0.5B", "Qwen-1.5B", "Llama-3.2-1B"]

rows = []
for m in MODELS:
    for hid, info in canon[m]["heads"].items():
        f = wf[m][hid]["features"]
        rows.append(dict(model=m, hid=hid, layer=info["layer"], head=info["head_idx"],
                         depth=info["relative_depth"], label=info["label"],
                         delta=info["delta"], match_entropy=info["match_entropy"], **f))
df = pd.DataFrame(rows)
df["cls5"] = df.label
df.loc[(df.label == "induction") & (df.depth >= 0.5), "cls5"] = "late_induction"
df.loc[(df.label == "induction") & (df.depth < 0.5), "cls5"] = "early_induction"
FEATS = ["entropy", "diag_off_ratio", "v_q_ratio", "q_k_ratio"] + [f"svd_{i}" for i in range(16)]
STATIC = [c for c in FEATS]  # purely weight-derived, no depth

# ---------------------------------------------------------------- A. label audit
A = {}
A["counts"] = {m: df[df.model == m].label.value_counts().to_dict() for m in MODELS}
A["total"] = df.label.value_counts().to_dict()
A["n_heads"] = len(df)
# Retrieval & Induction are both thresholds on the SAME delta -> disjoint by construction.
A["retrieval_and_induction_overlap_possible"] = False
A["both_from_single_delta"] = True
# sink priority shadows retrieval/induction?
shadow = df[(df.match_entropy < 0.1) & ((df.delta >= 0.3) | (df.delta <= -0.5))]
A["sink_shadowed_ret_or_ind"] = int(len(shadow))
A["sink_by_model_min_match_entropy"] = {m: float(df[df.model == m].match_entropy.min()) for m in MODELS}
A["frac_match_entropy_lt_0.5"] = {m: float((df[df.model == m].match_entropy < 0.5).mean()) for m in MODELS}
A["frac_match_entropy_lt_1.0"] = {m: float((df[df.model == m].match_entropy < 1.0).mean()) for m in MODELS}
R["A_label_audit"] = A

# ---------------------------------------------------------------- B. threshold sensitivity (independent axes)
B = {}
for m in MODELS:
    d = df[df.model == m].delta.values
    B[m] = {f"ret>={t}": int((d >= t).sum()) for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]}
    B[m].update({f"ind<={-t}": int((d <= -t).sum()) for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
sink_thr = {}
for m in MODELS:
    me = df[df.model == m].match_entropy.values
    sink_thr[m] = {f"sink<{t}": int((me < t).sum()) for t in [0.05, 0.1, 0.25, 0.5, 1.0]}
B["sink_threshold_sweep"] = sink_thr
R["B_threshold_sensitivity"] = B

# ---------------------------------------------------------------- C. static geometry: unsupervised + supervised
C = {}
def purity(y, c):
    ct = pd.crosstab(c, y)
    return float(ct.max(axis=1).sum() / ct.values.sum())

for scope, sub in [("GPT-2", df[df.model == "GPT-2"]), ("pooled", df)]:
    X = StandardScaler().fit_transform(sub[STATIC].values)
    km = KMeans(4, n_init=10, random_state=0).fit(X)
    y = sub.label.values
    ct = pd.crosstab(km.labels_, y)
    C[f"kmeans4_{scope}"] = dict(
        contingency=ct.to_dict(), ARI=float(adjusted_rand_score(y, km.labels_)),
        AMI=float(adjusted_mutual_info_score(y, km.labels_)), purity=purity(y, km.labels_),
        majority_baseline=float(pd.Series(y).value_counts(normalize=True).iloc[0]))
    # permutation null for AMI
    null = [adjusted_mutual_info_score(rng.permutation(y), km.labels_) for _ in range(200)]
    C[f"kmeans4_{scope}"]["AMI_perm_p"] = float((np.sum(np.array(null) >= C[f"kmeans4_{scope}"]["AMI"]) + 1) / 201)

def cv_eval(sub, cols, tag, binary=None):
    """5-fold CV; macro-AUC (OvR) and balanced accuracy. Rare classes limited by StratifiedKFold(n splits)."""
    y = sub.label.values
    classes = sorted(set(y))
    X = sub[cols].values
    min_cls = pd.Series(y).value_counts().min()
    k = min(5, int(min_cls))
    if k < 2:
        return None
    cv = StratifiedKFold(k, shuffle=True, random_state=0)
    out = {}
    for name, mdl in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                      ("gboost", GradientBoostingClassifier(random_state=0))]:
        proba = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")
        pred = np.array(sorted(classes))[proba.argmax(1)]
        try:
            auc = roc_auc_score(pd.get_dummies(y)[sorted(classes)].values, proba, average="macro")
        except Exception:
            auc = float("nan")
        out[name] = dict(balanced_acc=float(balanced_accuracy_score(y, pred)), macro_auc=float(auc), folds=k)
    return out

C["supervised"] = {}
for scope, sub in [(m, df[df.model == m]) for m in MODELS] + [("pooled", df)]:
    C["supervised"][scope] = dict(
        static_only=cv_eval(sub, STATIC, "static"),
        static_plus_depth=cv_eval(sub, STATIC + ["depth"], "static+depth"),
        depth_only=cv_eval(sub, ["depth"], "depth"))
# leave-one-model-out (static only)
loo = {}
for held in MODELS:
    tr, te = df[df.model != held], df[df.model == held]
    mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(tr[STATIC], tr.label)
    pred = mdl.predict(te[STATIC])
    loo[held] = float(balanced_accuracy_score(te.label, pred))
C["leave_one_model_out_balanced_acc_logreg_static"] = loo
C["chance_balanced_acc"] = 0.25
R["C_static_geometry"] = C

# ---------------------------------------------------------------- D. V/Q depth trend with proper statistics
D = {"per_model": {}}
for m in MODELS:
    s = df[df.model == m]
    r, p = stats.pearsonr(s.depth, s.v_q_ratio)
    rho, _ = stats.spearmanr(s.depth, s.v_q_ratio)
    # layer-level (n = n_layers): heads within a layer are not independent
    lm = s.groupby("layer").agg(depth=("depth", "first"), vq=("v_q_ratio", "mean"))
    rl, pl = stats.pearsonr(lm.depth, lm.vq)
    # cluster bootstrap over layers
    layers = s.layer.unique(); boots = []
    for _ in range(2000):
        pick = rng.choice(layers, len(layers), replace=True)
        ss = pd.concat([s[s.layer == l] for l in pick])
        boots.append(stats.pearsonr(ss.depth, ss.v_q_ratio)[0])
    # variance explained by layer
    grand = s.v_q_ratio.mean()
    ssb = sum(len(g) * (g.v_q_ratio.mean() - grand) ** 2 for _, g in s.groupby("layer"))
    sst = ((s.v_q_ratio - grand) ** 2).sum()
    # within-layer association with the dynamic axis delta (depth-controlled)
    s2 = s.copy()
    s2["vq_c"] = s2.v_q_ratio - s2.groupby("layer").v_q_ratio.transform("mean")
    s2["dl_c"] = s2.delta - s2.groupby("layer").delta.transform("mean")
    rw, pw = stats.pearsonr(s2.vq_c, s2.dl_c)
    D["per_model"][m] = dict(n_heads=len(s), n_layers=len(layers), head_level_r=float(r), head_level_p=float(p),
                             spearman=float(rho), layer_level_r=float(rl), layer_level_p=float(pl),
                             cluster_boot_r_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                             frac_variance_between_layers=float(ssb / sst),
                             within_layer_vq_vs_delta_r=float(rw), within_layer_vq_vs_delta_p=float(pw))
# pooled: head-level vs layer-level
r, p = stats.pearsonr(df.depth, df.v_q_ratio)
D["pooled_head_level"] = dict(r=float(r), p=float(p), n=len(df))
lay = df.groupby(["model", "layer"]).agg(depth=("depth", "first"), vq=("v_q_ratio", "mean")).reset_index()
rl, pl = stats.pearsonr(lay.depth, lay.vq)
D["pooled_layer_level"] = dict(r=float(rl), p=float(pl), n_layer_units=len(lay))
# mixed model: vq ~ depth + (1|model/layer)
import statsmodels.formula.api as smf
df["ml"] = df.model + "_" + df.layer.astype(str)
try:
    md = smf.mixedlm("v_q_ratio ~ depth", df, groups=df["model"], re_formula="~depth").fit(reml=True)
    D["mixedlm_model_random_slope"] = dict(slope=float(md.params["depth"]), se=float(md.bse["depth"]),
                                           p=float(md.pvalues["depth"]))
except Exception as e:
    D["mixedlm_model_random_slope"] = f"failed: {e}"
# OLS with layer-clustered SE (per model clusters across layer)
import statsmodels.api as sm
ols = sm.OLS(df.v_q_ratio, sm.add_constant(df[["depth"]])).fit(cov_type="cluster", cov_kwds={"groups": df.ml})
D["pooled_ols_layer_clustered"] = dict(slope=float(ols.params["depth"]), se=float(ols.bse["depth"]), p=float(ols.pvalues["depth"]))
R["D_vq_trend"] = D

# ---------------------------------------------------------------- E. depth enrichment / Early-Late split
E = {}
g5 = ["sink", "local", "retrieval", "early_induction", "late_induction"]
E["early_late_definition"] = "depth>=0.5 (hard threshold, see scripts/generate_real_headgenome_map.py L35-36), NOT k-means"
E["table3"] = {}
for c in g5:
    d = df[df.cls5 == c].depth.values
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if len(d) > 2 else (np.nan, np.nan)
    E["table3"][c] = dict(n=len(d), mean=float(d.mean()), ci95=[float(ci[0]), float(ci[1])])
H5, p5 = stats.kruskal(*[df[df.cls5 == c].depth.values for c in g5])
H4, p4 = stats.kruskal(*[df[df.label == c].depth.values for c in ["sink", "local", "retrieval", "induction"]])
E["kruskal_k5"] = dict(H=float(H5), p=float(p5)); E["kruskal_k4"] = dict(H=float(H4), p=float(p4))
pairs = list(itertools.combinations(g5, 2)); pv = []
for a, b in pairs:
    pv.append(stats.mannwhitneyu(df[df.cls5 == a].depth, df[df.cls5 == b].depth).pvalue)
holm = multipletests(pv, method="holm")[1]
E["pairwise_mwu_holm"] = {f"{a}__{b}": dict(p_raw=float(p), p_holm=float(h)) for (a, b), p, h in zip(pairs, pv, holm)}
E["circularity_note"] = "Early/Late are defined by depth; depth test between them is tautological."
# does a depth-FREE split exist? k-means(k=2) on induction heads using static weights only (no depth)
ind = df[df.label == "induction"]
X = StandardScaler().fit_transform(ind[STATIC].values)
kmi = KMeans(2, n_init=20, random_state=0).fit(X)
E["induction_depthfree_kmeans2"] = dict(
    n=len(ind), ARI_vs_depth_split=float(adjusted_rand_score((ind.depth >= 0.5).astype(int), kmi.labels_)),
    depth_means=[float(ind[kmi.labels_ == k].depth.mean()) for k in (0, 1)],
    sizes=[int((kmi.labels_ == k).sum()) for k in (0, 1)])
# bimodality of induction depth? (dip proxy: compare 1 vs 2 component GMM BIC)
from sklearn.mixture import GaussianMixture
dd = ind.depth.values.reshape(-1, 1)
E["induction_depth_gmm_bic"] = {k: float(GaussianMixture(k, random_state=0, n_init=5).fit(dd).bic(dd)) for k in (1, 2, 3)}
R["E_depth_enrichment"] = E

# ---------------------------------------------------------------- F. Table 6 / perplexity illusion CIs (n per depth)
ruler = json.load(open(f"{ROOT}/outputs/phase6/ruler_comprehensive.json"))
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, c - h)), float(min(1, c + h))]
F = {}
for cfg, v in ruler.items():
    d = v["success_by_depth"]
    nd = 20  # inferred: all depth values are multiples of 0.05 -> n=20 per depth
    F[cfg] = {k: dict(acc=a, k=int(round(a * nd)), n=nd, wilson95=wilson(int(round(a * nd)), nd)) for k, a in d.items()}
    F[cfg]["overall"] = dict(acc=v["overall_accuracy"], k=int(round(v["overall_accuracy"] * 100)), n=100,
                             wilson95=wilson(int(round(v["overall_accuracy"] * 100)), 100))
F["note"] = "n inferred from granularity (all accuracies are multiples of 0.05 => n=20/depth, 100 total); paper text says N=4,000 (that is context length, not trials)."
R["F_perplexity_illusion_CIs"] = F

json.dump(R, open(f"{OUT}/01_stats_audit.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved", f"{OUT}/01_stats_audit.json")
