# Cell-driven information fusion — code and results

Reference implementation and released scores for *Cell-Driven Information Fusion:
Explainable, Preprocessing-Free Embedding of Single-Cell RNA-seq Data at Atlas Scale*.

Author: Henry Han. MIT licensed.

This is a **light release**. It carries the method and every number the paper reports,
so any claim can be checked on a CPU in under a minute. It does not carry the raw
expression matrices (all 29 datasets are public; accessions are in the paper) or the
figure-generation code.

## Check every number in the paper

```bash
cd scripts
python3 audit.py
```

48 checks, each printed `OK` or `XX` against the value quoted in the manuscript.
The script recomputes, from `../data/` alone: the benchmark's span (datasets, entries,
cells, sparsity, classes), all 48 mean NMI and AMI scores of Table 8, that c-UMAP is
highest in each of the four columns, and all 22 paired Wilcoxon tests with their win
counts and p-values, including the Bonferroni check.

Requires `numpy`, `pandas` and `scipy`.

## Run the method

```bash
cd scripts
python3 quickstart.py
```

Or, in your own code:

```python
from cumap import CTSNE
model = CTSNE(distances=("yule", "fractional", "l_chebyshev"), weights="auto")
Y = model.fit_transform(X)   # X is a raw count matrix, cells by genes
```

Pass an explicit triple to reproduce a protocol, for example
`weights=(0.90, 0.00, 0.10)` for Raw.  Note that `weights="auto"` in the package is a
grid search scored by NMI and therefore needs labels; the label-free eigengap
selection of Section 3.11 is a separate script, `scripts/paper/fusion_select.py`.

The fusion needs no normalization, log-transform, imputation or gene selection.
It reads raw counts.

## Layout

```
cumap/        the method. fusion.py is the sparsity-adaptive operator; distances.py
              holds the three biological views (Yule's Y, fractional, L-Chebyshev);
              search.py is the label-free (alpha, w) selection; embedding.py is the
              UMAP backbone; accel.py and gpu/ are the lean large-scale paths.
data/         the released scores, one row per dataset, one column per method.
              nmi_raw.csv / nmi_pp.csv / ami_raw.csv / ami_pp.csv are the two indices
              under the two protocols. 31 scored rows per file: the 29 benchmark
              datasets plus the two cortex datasets read at fine label resolution,
              giving 62 evaluation entries in all.
scripts/      audit.py recomputes the paper's numbers; quickstart.py runs the method.
scripts/paper/
              the experiment scripts behind the tables the released scores do not
              cover: fusion_select.py is the label-free eigengap selection of
              Section 3.11 (Table 6), power_mean_fusion.py the power-mean operator,
              fast_ctsne.py the near-linear c-TSNE, and the three snf_*.py the SNF
              comparison.  These read the raw expression matrices, which are not
              shipped here; every dataset is public and its accession is in the paper.
tests/        unit tests for the distances, the fusion, the search and the pipeline.
```

Each score file also carries the per-dataset cell count, gene count, sparsity, class
count and the fixed weight vector used, so the benchmark table can be rebuilt from it.

## The two fixed weight settings

One weight vector per protocol, applied unchanged to every dataset:

| Protocol | `[w_yule, w_frac, w_L-cheb]` |
|---|---|
| Raw (no preprocessing) | `[0.90, 0.00, 0.10]` |
| PP (standard preprocessing) | `[0.80, 0.10, 0.10]` |

One entry triggers the documented degenerate-case safeguard and uses `[0.00, 0.20, 0.80]`.
