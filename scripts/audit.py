"""Recompute every headline number of the paper from the released per-dataset scores.

Run it from this folder:

    cd scripts && python3 audit.py

Each check prints OK or XX against the value quoted in the manuscript.  No GPU and
no raw expression data are needed: the scores in ../data/ are what the runs produced,
and every table in the paper is a summary of them.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
METHODS = ["scICE", "tSNE", "UMAP", "KPCA", "PHATE", "SIMLR",
           "scDCCA", "scGAC", "scAGCL", "scGPT(KM)", "scGPT(Le)"]
SHEETS = {"NMI": ("nmi_raw.csv", "nmi_pp.csv"), "AMI": ("ami_raw.csv", "ami_pp.csv")}

# What the manuscript states, table by table.
PAPER_MEAN = {            # Table "summary", columns 2-5
    "c-UMAP":    (0.618, 0.696, 0.657, 0.728),
    "scICE":     (0.499, 0.632, 0.573, 0.707),
    "scAGCL":    (0.579, 0.579, 0.618, 0.626),
    "UMAP":      (0.473, 0.646, 0.509, 0.695),
    "PHATE":     (0.402, 0.661, 0.430, 0.697),
    "tSNE":      (0.464, 0.563, 0.502, 0.608),
    "scGPT(Le)": (0.490, 0.514, 0.543, 0.573),
    "scGPT(KM)": (0.449, 0.480, 0.489, 0.521),
    "scDCCA":    (0.363, 0.594, 0.381, 0.633),
    "scGAC":     (0.378, 0.505, 0.413, 0.557),
    "SIMLR":     (0.365, 0.581, 0.378, 0.597),
    "KPCA":      (0.163, 0.521, 0.168, 0.561),
}
PAPER_WINS = {            # Table "summary", the two "vs. c-UMAP" blocks
    "scICE":     ((42, 57), (39, 57)), "scAGCL": ((45, 54), (47, 54)),
    "UMAP":      ((45, 58), (48, 58)), "PHATE":  ((46, 58), (47, 58)),
    "tSNE":      ((48, 58), (50, 58)), "scGPT(Le)": ((47, 59), (46, 59)),
    "scGPT(KM)": ((49, 59), (52, 59)), "scDCCA": ((52, 58), (53, 58)),
    "scGAC":     ((39, 46), (40, 46)), "SIMLR":  ((52, 55), (53, 55)),
    "KPCA":      ((55, 58), (55, 58)),
}
PAPER_P = {               # same table, the two p columns
    "scICE":     (2.5e-5, 9.5e-4), "scAGCL": (1.5e-6, 2.0e-7),
    "UMAP":      (8.7e-8, 3.1e-8), "PHATE":  (1.3e-7, 2.6e-8),
    "tSNE":      (9.0e-9, 2.0e-9), "scGPT(Le)": (2.6e-6, 1.6e-6),
    "scGPT(KM)": (4.7e-9, 1.7e-9), "scDCCA": (7.1e-9, 1.8e-9),
    "scGAC":     (8.3e-8, 4.7e-8), "SIMLR":  (1.2e-9, 5.6e-10),
    "KPCA":      (6.7e-11, 6.4e-11),
}

ok_count = bad_count = 0


def check(passed, message):
    global ok_count, bad_count
    print(("  OK  " if passed else "  XX  ") + message)
    if passed:
        ok_count += 1
    else:
        bad_count += 1


def sheet(name):
    """One protocol sheet, summary rows dropped, scores numeric."""
    d = pd.read_csv(os.path.join(DATA, name))
    d = d[d["dataset"].astype(str) != "Mean (this sheet)"]
    for column in ["CUMAP"] + METHODS:
        d[column] = pd.to_numeric(d[column], errors="coerce")
    return d[d["CUMAP"].notna()]


def paired(index, method):
    """c-UMAP against one baseline, Raw and PP pooled, entries the baseline produced."""
    raw, pp = (sheet(f) for f in SHEETS[index])
    ours = np.concatenate([raw["CUMAP"].values, pp["CUMAP"].values])
    theirs = np.concatenate([raw[method].values, pp[method].values])
    usable = np.isfinite(ours) & np.isfinite(theirs)
    return ours[usable], theirs[usable]


print("Explainable t-SNE, cell-driven information fusion:\n"
      "every number in the paper, recomputed from the released scores\n")

# ---- the benchmark itself
raw = sheet("nmi_raw.csv")
check(len(raw) == 31, f"scored rows per protocol = {len(raw)} (29 datasets + 2 fine-resolution cortex readings)")
check(2 * len(raw) == 62, f"evaluation entries = {2 * len(raw)}, text says 62")
check(len(METHODS) == 11, f"baselines = {len(METHODS)}, text says 11")

cells = pd.to_numeric(raw["Cells"], errors="coerce").dropna()
spars = pd.to_numeric(raw["Sparsity%"], errors="coerce").dropna()
klass = pd.to_numeric(raw["K"], errors="coerce").dropna()
check(int(cells.min()) == 225 and int(cells.max()) == 1136218,
      f"cells span {int(cells.min())}--{int(cells.max())}, text says 225--1,136,218")
check(abs(spars.min() - 45.0) < 0.05 and abs(spars.max() - 98.0) < 0.05,
      f"sparsity spans {spars.min():.1f}--{spars.max():.1f}%, text says 45.0--98.0%")
check(int(klass.min()) == 3 and int(klass.max()) == 180,
      f"classes span {int(klass.min())}--{int(klass.max())}, text says 3--180")
check(round(cells.max() / cells.min()) in range(4900, 5200),
      f"cell-count range is {cells.max() / cells.min():.0f}-fold, text says 5,000-fold")

# ---- Table "summary": the sixteen mean scores plus the eleven baselines' means
print()
columns = [("NMI", 0, "nmi_raw.csv"), ("NMI", 1, "nmi_pp.csv"),
           ("AMI", 2, "ami_raw.csv"), ("AMI", 3, "ami_pp.csv")]
for method, quoted in PAPER_MEAN.items():
    column = "CUMAP" if method == "c-UMAP" else method
    got = [sheet(f)[column].mean() for _, _, f in columns]
    agree = all(abs(g - q) < 5e-4 for g, q in zip(got, quoted))
    check(agree, "mean %-10s %s vs text %s" % (method,
          " ".join("%.3f" % g for g in got), " ".join("%.3f" % q for q in quoted)))

# ---- c-UMAP is highest in each of the four columns
print()
for label, _, filename in columns:
    d = sheet(filename)
    best = max(METHODS, key=lambda m: d[m].mean())
    check(d["CUMAP"].mean() > d[best].mean(),
          "%s %-7s c-UMAP %.3f is above the best baseline %s %.3f"
          % (label, filename.split("_")[1][:-4], d["CUMAP"].mean(), best, d[best].mean()))

# ---- the 22 paired tests
print()
pvalues = []
for index, slot in (("NMI", 0), ("AMI", 1)):
    for method in METHODS:
        ours, theirs = paired(index, method)
        wins = int((ours > theirs).sum())
        p = wilcoxon(ours, theirs, alternative="two-sided").pvalue
        pvalues.append(p)
        qw, qn = PAPER_WINS[method][slot]
        qp = PAPER_P[method][slot]
        agree = (wins == qw and len(ours) == qn
                 and abs(np.log10(p) - np.log10(qp)) < 0.03)
        check(agree, "%s %-10s wins %d/%d (text %d/%d), p %.1e (text %.1e)"
              % (index, method, wins, len(ours), qw, qn, p, qp))

print()
check(len(pvalues) == 22, f"paired tests run = {len(pvalues)}, text says 22")
check(max(pvalues) < 1e-3, f"largest p = {max(pvalues):.1e}, text says every test is below 0.001")
check(max(pvalues) < 0.05 / 22,
      f"largest p = {max(pvalues):.1e} is below the Bonferroni level {0.05 / 22:.1e}, text says all 22 survive")

print("\n%d OK, %d XX" % (ok_count, bad_count))
sys.exit(1 if bad_count else 0)
