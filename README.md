# scD2P

**scD2P** is an interpretable framework for modeling drug-induced perturbation programs from single-cell drug perturbation transcriptomic data.

The model separates drug-associated transcriptional changes from intrinsic basal cellular states by using a basal state-corrected residual decomposition strategy. It learns basal programs from control cells, estimates expected unperturbed expression, and decomposes drug-induced residual expression into low-dimensional perturbation programs.

## Key features

- Learn robust non-negative basal programs from control cells.
- Estimate basal full-gene expression using a regularized expression model.
- Extract signed perturbation programs from residual expression using Incremental PCA.
- Project external datasets into a fitted reference program space.
- Annotate basal and perturbation programs using pathway enrichment.
- Evaluate program reproducibility across experimental replicates.
- Support downstream tasks such as mechanism-of-action-labeled drug clustering, perturbation expression prediction, and drug sensitivity classification.

## Repository structure

```text
scD2P-github/
├── scD2P/
│   ├── __init__.py
│   ├── model.py              # Core scD2P fitting and projection
│   ├── enrichment.py         # Program-to-pathway annotation
│   ├── pathway_overlap.py    # Replicate-level pathway reproducibility
│   ├── benchmarking.py       # Lightweight downstream benchmarking helpers
│   ├── metrics.py            # Regression, expression, and classification metrics
│   └── utils.py              # General utilities
├── scripts/
│   ├── fit_scD2P.py          # Command-line model fitting
│   ├── project_dataset.py    # Command-line external projection
│   └── run_enrichment.py     # Command-line pathway enrichment
├── examples/
│   └── quick_start.py
├── tests/
│   └── test_import.py
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/QiGuan1920/scD2P.git
cd scD2P
pip install -e .
```

Install optional development tools:

```bash
pip install -e ".[dev]"
```

## Input data format

scD2P expects an `AnnData` object with:

- `adata.layers['lognorm']`: normalized expression matrix.
- `adata.obs['perturbation']`: perturbation label for each cell.
- `adata.var_names`: gene symbols or gene identifiers.
- `adata.obs['cell_line']`
- `adata.obs['is_ctrl']`

For replicate-level reproducibility analysis, the following columns are also used:

- `adata.obs['replicate']`

## Python quick start

```python
import scanpy as sc
from scD2P import fit_scD2P, save_model

adata = sc.read_h5ad("data/example.h5ad")

model, outputs = fit_scD2P(
    adata,
    n_basal=16,
    n_pert=16,
    ctrl_label="control",
    perturbation_col="perturbation",
    layer="lognorm",
)

save_model(model, "outputs/scD2P_model.joblib")
adata.write_h5ad("outputs/example_with_scD2P_embeddings.h5ad")
```

After fitting, scD2P writes the following representations:

- `adata.obsm['X_zb']`: basal program scores.
- `adata.obsm['X_zp']`: perturbation program scores.
- `adata.obsm['X_z']`: concatenated basal and perturbation scores.

## Command-line usage

Fit scD2P:

```bash
python scripts/fit_scD2P.py \
  --adata data/example.h5ad \
  --out-model outputs/scD2P_model.joblib \
  --out-adata outputs/example_with_scD2P_embeddings.h5ad \
  --layer lognorm \
  --perturbation-col perturbation \
  --ctrl-label control \
  --n-basal 16 \
  --n-pert 16
```

Project an external dataset:

```bash
python scripts/project_dataset.py \
  --adata data/external.h5ad \
  --model outputs/scD2P_model.joblib \
  --out-adata outputs/external_projected.h5ad \
  --layer lognorm
```

Run pathway enrichment:

```bash
python scripts/run_enrichment.py \
  --model outputs/scD2P_model.joblib \
  --gmt data/h.all.v2026.1.Hs.symbols.gmt \
  --out-dir outputs/enrichment
```
