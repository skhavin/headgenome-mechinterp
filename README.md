# How Far Does Dynamic Entropy-Collapse Probing Go?
### A Critical Evaluation of Behavioral Attention Head Specialization
**ACL ARR Submission — Mechanistic Interpretability Track**

Repository: [skhavin/headgenome-mechinterp](https://github.com/skhavin/headgenome-mechinterp)  
Anonymized Submission: [anonymous.4open.science/r/headgenome-mechinterp](https://anonymous.4open.science/r/headgenome-mechinterp)

---

## Overview

This repository contains the complete empirical code, auditing scripts, raw execution logs, and LaTeX source for the ACL ARR paper:
> **"How Far Does Dynamic Entropy-Collapse Probing Go? A Critical Evaluation of Behavioral Attention Head Specialization"**

### Key Findings
1. **Retrieval Head Causal Dissociation:** Heads labeled as retrieval via dynamic $\Delta$-probing are causally inert on long-context Needle-in-a-Haystack (NIAH, $100.0\% \to 100.0\%$, drop $0.0\%$). True task-critical heads (selected by attention to needle) collapse accuracy to $37.0\%$ [$28.2\%, 46.8\%$], while random $k=4$ controls preserve $98.5\%$. The dynamic probe achieves an AUC-ROC of only $0.43$ [$0.21, 0.65$].
2. **Circuit Dilution in Induction:** Probed induction heads cause a mild $24.0\%$ drop ($76.0\%$), whereas Olsson prefix-matching heads collapse NIAH to $3.0\%$ [$1.0\%, 8.5\%$]. On Indirect Object Identification (IOI), activation patching recovers $+67.7\%$ logit difference for Olsson heads versus only $+17.8\%$ for probed heads and $+1.2\%$ for random.
3. **The Perplexity Illusion:** Perplexity is insensitive to factual retrieval ablation ($\Delta\text{PPL} = +0.24$ despite a $63\%$ drop in accuracy), masking severe sub-circuit failure, whereas induction ablation impairs perplexity by $+4.86$ ($24\%$).
4. **Weight Frobenius Ratio as Depth Gradient:** $\|W_V\|_F / \|W_Q\|_F$ is an emergent architectural depth gradient ($r = 0.635\text{--}0.734$, $p = 1.9 \times 10^{-11}$) driven by residual backpropagation, rather than a functional class separator ($r \approx -0.01$ to $+0.03$ within layers).
5. **Bottleneck Layer Choking:** Prior random control collapses were artifacts of sampling multiple heads from the same layer, starving representation rank, rather than attention sink collisions ($R^2 = 0.156, p = 0.085$).

---

## Directory Structure

```text
├── FINDINGS.md                # Detailed audit dossier and raw verified metrics
├── paper1_headgenome.pdf      # Compiled camera-ready / review PDF
├── latex_paper/               # Official ACL template and paper source
│   ├── paper1_headgenome.tex  # LaTeX main document
│   ├── references.bib         # Bibliography
│   └── figures/               # High-resolution vector/raster figures
├── results/                   # Raw JSON logs and stdout evaluation traces
│   ├── 01_stats_audit.json
│   ├── 02_GPT-2.json
│   ├── 02_Qwen-0.5B.json
│   ├── 03_extra_audits.json
│   ├── 04_qwen05.log
│   └── 05_qwen05.log
└── scripts/                   # Standalone, reproducible evaluation scripts
    ├── 01_stats_audit.py
    ├── 02_probes_ablation_patching.py
    ├── 03_paper1_extra_audits.py
    ├── 04_rigorous_ablation_distribution.py
    └── 05_rigorous_unconfounded_suite.py
```

---

## Reproducing Experiments

```bash
# 1. Install dependencies
pip install torch transformers datasets accelerate

# 2. Run statistical audit and threshold sensitivity
python scripts/01_stats_audit.py

# 3. Run random-init V/Q null, GQA query variance, and GMM bimodality audit
python scripts/03_paper1_extra_audits.py

# 4. Run unconfounded causal ablation and activation patching suite
python scripts/05_rigorous_unconfounded_suite.py
```
