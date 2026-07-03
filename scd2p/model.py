"""Core scD2P model.

scD2P decomposes single-cell drug perturbation data into basal programs and
basal-state-corrected perturbation programs. The model is intentionally modular:

1. Learn robust non-negative basal programs from control cells.
2. Project all cells onto the learned basal programs.
3. Predict basal full-gene expression from basal program scores.
4. Compute drug-associated residual expression.
5. Extract signed perturbation programs from the residual space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import IncrementalPCA, NMF, non_negative_factorization
from sklearn.linear_model import Ridge

from .utils import get_layer_matrix, make_batches, to_dense_array, validate_obs_columns


@dataclass
class ScD2PModel:
    """Container for a fitted scD2P reference model."""

    h_basal: np.ndarray
    h_basal_filtered: np.ndarray
    gene_mask: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: np.ndarray
    ipca_components: np.ndarray
    ipca_mean: np.ndarray
    ipca_var: np.ndarray | None
    ipca_var_ratio: np.ndarray | None
    var_names: np.ndarray
    n_basal: int
    n_pert: int
    layer: str | None = "lognorm"

    @property
    def n_genes(self) -> int:
        """Return the number of reference genes."""
        return len(self.var_names)


def robust_nmf(
    x_counts: np.ndarray,
    n_components: int = 16,
    n_runs: int = 15,
    seed: int = 42,
    max_ctrl: int = 5000,
    max_iter: int = 100,
) -> np.ndarray:
    """Learn consensus basal programs using repeated NMF and clustering.

    Parameters
    ----------
    x_counts:
        Non-negative control-cell expression matrix with shape ``n_cells x n_genes``.
    n_components:
        Number of basal programs.
    n_runs:
        Number of NMF runs used to build the program pool.
    seed:
        Random seed.
    max_ctrl:
        Maximum number of control cells used for each fitting run.
    max_iter:
        Maximum NMF iterations.

    Returns
    -------
    np.ndarray
        Consensus basal program matrix with shape ``n_components x n_genes``.
    """
    if np.any(x_counts < 0):
        raise ValueError("NMF input must be non-negative.")

    if x_counts.shape[0] > max_ctrl:
        rng_sub = np.random.RandomState(seed)
        idx = rng_sub.choice(x_counts.shape[0], max_ctrl, replace=False)
        x_fit = x_counts[idx].astype(np.float32, copy=False)
        print(f"Subsampled control cells: {x_counts.shape[0]} -> {max_ctrl}")
    else:
        x_fit = x_counts.astype(np.float32, copy=False)

    rng = np.random.RandomState(seed)
    all_h = []

    for _ in range(n_runs):
        nmf = NMF(
            n_components=n_components,
            init="nndsvda",
            random_state=int(rng.randint(0, 100000)),
            max_iter=max_iter,
        )
        nmf.fit(x_fit)
        h = nmf.components_
        norms = np.linalg.norm(h, axis=1)
        for j in range(h.shape[0]):
            if norms[j] > 1e-10:
                all_h.append(h[j] / norms[j])

    all_programs = np.vstack(all_h).astype(np.float32)
    finite_mask = np.all(np.isfinite(all_programs), axis=1)
    all_programs = all_programs[finite_mask]
    print(f"Valid program vectors: {len(all_programs)} / {n_runs * n_components}")

    dist = pdist(all_programs, metric="euclidean")
    if not np.all(np.isfinite(dist)):
        finite_dist = dist[np.isfinite(dist)]
        fill_value = np.nanmax(finite_dist) if finite_dist.size else 0.0
        dist = np.where(np.isfinite(dist), dist, fill_value)

    z_link = linkage(dist, method="average")
    clusters = fcluster(z_link, t=n_components, criterion="maxclust")

    consensus_h = np.zeros((n_components, x_fit.shape[1]), dtype=np.float32)
    for cluster_id in range(1, n_components + 1):
        members = all_programs[clusters == cluster_id]
        if len(members) > 0:
            consensus_h[cluster_id - 1] = members.mean(axis=0)

    return consensus_h


def project_basal_programs(
    adata: Any,
    h_basal_filtered: np.ndarray,
    gene_mask: np.ndarray,
    layer: str | None = "lognorm",
    batch_size: int = 10000,
    max_iter: int = 100,
) -> np.ndarray:
    """Project cells onto fixed non-negative basal programs."""
    x_matrix = get_layer_matrix(adata, layer)
    n_cells = adata.shape[0]
    n_components = h_basal_filtered.shape[0]
    z_basal = np.zeros((n_cells, n_components), dtype=np.float32)
    h64 = h_basal_filtered.astype(np.float64, copy=False)

    n_batches = int(np.ceil(n_cells / batch_size))
    for batch_idx, (start, end) in enumerate(make_batches(n_cells, batch_size), start=1):
        x_batch = to_dense_array(x_matrix[start:end], dtype=np.float32)
        x_batch = np.maximum(x_batch[:, gene_mask], 0).astype(np.float64, copy=False)
        w_batch, _, _ = non_negative_factorization(
            x_batch,
            W=None,
            H=h64,
            n_components=n_components,
            init="custom",
            update_H=False,
            max_iter=max_iter,
        )
        z_basal[start:end] = w_batch.astype(np.float32, copy=False)
        if batch_idx % 5 == 0 or batch_idx == n_batches:
            print(f"Projected basal programs: batch {batch_idx}/{n_batches}")

    return z_basal


def extract_perturbation_programs(
    adata: Any,
    z_basal: np.ndarray,
    is_control: np.ndarray,
    n_pert: int = 16,
    layer: str | None = "lognorm",
    batch_size: int = 10000,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, IncrementalPCA, Ridge]:
    """Fit a basal expression model and extract residual perturbation programs."""
    x_matrix = get_layer_matrix(adata, layer)
    n_cells = adata.shape[0]
    drug_mask = ~is_control

    x_ctrl = to_dense_array(x_matrix[is_control], dtype=np.float32)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(z_basal[is_control], x_ctrl)

    ipca = IncrementalPCA(n_components=n_pert)
    n_batches = int(np.ceil(n_cells / batch_size))
    print(f"IncrementalPCA fit pass: {n_batches} batches")

    for batch_idx, (start, end) in enumerate(make_batches(n_cells, batch_size), start=1):
        x_batch = to_dense_array(x_matrix[start:end], dtype=np.float32)
        pred_batch = ridge.predict(z_basal[start:end]).astype(np.float32, copy=False)
        residual = x_batch - pred_batch
        drug_in_batch = drug_mask[start:end]
        if int(drug_in_batch.sum()) > n_pert:
            ipca.partial_fit(residual[drug_in_batch])
        if batch_idx % 5 == 0 or batch_idx == n_batches:
            print(f"IncrementalPCA fit: batch {batch_idx}/{n_batches}")

    z_pert = np.zeros((n_cells, n_pert), dtype=np.float32)
    print(f"IncrementalPCA transform pass: {n_batches} batches")

    for batch_idx, (start, end) in enumerate(make_batches(n_cells, batch_size), start=1):
        x_batch = to_dense_array(x_matrix[start:end], dtype=np.float32)
        pred_batch = ridge.predict(z_basal[start:end]).astype(np.float32, copy=False)
        residual = x_batch - pred_batch
        z_pert[start:end] = ipca.transform(residual).astype(np.float32, copy=False)
        if batch_idx % 5 == 0 or batch_idx == n_batches:
            print(f"IncrementalPCA transform: batch {batch_idx}/{n_batches}")

    if hasattr(ipca, "explained_variance_ratio_"):
        cumvar = np.cumsum(ipca.explained_variance_ratio_)
        print(f"Cumulative explained variance: {cumvar[-1]:.3f}")

    return z_pert, ipca.components_, ipca, ridge


def fit_scd2p(
    adata: Any,
    n_basal: int = 16,
    n_pert: int = 16,
    ctrl_label: str = "control",
    perturbation_col: str = "perturbation",
    layer: str | None = "lognorm",
    nmf_runs: int = 15,
    max_ctrl: int = 5000,
    batch_size: int = 10000,
    ridge_alpha: float = 1.0,
    seed: int = 42,
) -> tuple[ScD2PModel, dict[str, np.ndarray]]:
    """Fit scD2P and store cell-level representations in ``adata.obsm``.

    The function writes three matrices:
    ``adata.obsm['X_zb']`` for basal scores,
    ``adata.obsm['X_zp']`` for perturbation scores, and
    ``adata.obsm['X_z']`` for their concatenation.
    """
    validate_obs_columns(adata, [perturbation_col])
    is_control = (adata.obs[perturbation_col].astype(str).values == ctrl_label)
    adata.obs["is_ctrl"] = is_control

    x_matrix = get_layer_matrix(adata, layer)
    x_ctrl = to_dense_array(x_matrix[is_control], dtype=np.float32)
    x_ctrl_nonneg = np.maximum(x_ctrl, 0)
    gene_mask = x_ctrl_nonneg.sum(axis=0) > 0
    x_ctrl_filtered = x_ctrl_nonneg[:, gene_mask]
    print(f"Retained genes for basal NMF: {int(gene_mask.sum())} / {len(gene_mask)}")

    print("Step 1/3: learning basal programs")
    h_basal_filtered = robust_nmf(
        x_ctrl_filtered,
        n_components=n_basal,
        n_runs=nmf_runs,
        seed=seed,
        max_ctrl=max_ctrl,
    )
    h_basal = np.zeros((n_basal, adata.shape[1]), dtype=np.float32)
    h_basal[:, gene_mask] = h_basal_filtered

    print("Step 2/3: projecting basal program scores")
    z_basal = project_basal_programs(
        adata,
        h_basal_filtered,
        gene_mask,
        layer=layer,
        batch_size=batch_size,
    )
    adata.obsm["X_zb"] = z_basal

    print("Step 3/3: extracting perturbation programs")
    z_pert, pert_components, ipca, ridge = extract_perturbation_programs(
        adata,
        z_basal,
        is_control=is_control,
        n_pert=n_pert,
        layer=layer,
        batch_size=batch_size,
        ridge_alpha=ridge_alpha,
    )
    adata.obsm["X_zp"] = z_pert
    adata.obsm["X_z"] = np.hstack([z_basal, z_pert])

    model = ScD2PModel(
        h_basal=h_basal,
        h_basal_filtered=h_basal_filtered,
        gene_mask=gene_mask,
        ridge_coef=ridge.coef_.astype(np.float32, copy=False),
        ridge_intercept=ridge.intercept_.astype(np.float32, copy=False),
        ipca_components=pert_components.astype(np.float32, copy=False),
        ipca_mean=ipca.mean_.astype(np.float32, copy=False),
        ipca_var=getattr(ipca, "var_", None),
        ipca_var_ratio=getattr(ipca, "explained_variance_ratio_", None),
        var_names=np.asarray(adata.var_names.astype(str)),
        n_basal=n_basal,
        n_pert=n_pert,
        layer=layer,
    )

    outputs = {
        "z_basal": z_basal,
        "z_pert": z_pert,
        "z": adata.obsm["X_z"],
        "is_control": is_control,
    }
    return model, outputs


def save_model(model: ScD2PModel, path: str | Path) -> None:
    """Save a fitted scD2P model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> ScD2PModel:
    """Load a fitted scD2P model saved by :func:`save_model`."""
    return joblib.load(path)


def _align_to_reference_genes(adata: Any, model: ScD2PModel, layer: str | None) -> np.ndarray:
    """Align an AnnData expression matrix to the reference gene order."""
    x_matrix = get_layer_matrix(adata, layer)
    x = to_dense_array(x_matrix, dtype=np.float32)
    query_genes = np.asarray(adata.var_names.astype(str))
    query_index = {gene: idx for idx, gene in enumerate(query_genes)}

    aligned = np.zeros((adata.shape[0], model.n_genes), dtype=np.float32)
    matched = 0
    for ref_idx, gene in enumerate(model.var_names):
        src_idx = query_index.get(str(gene))
        if src_idx is not None:
            aligned[:, ref_idx] = x[:, src_idx]
            matched += 1
    print(f"Aligned genes: {matched} / {model.n_genes}")
    return aligned


def project_adata(
    adata: Any,
    model: ScD2PModel,
    layer: str | None = "lognorm",
    batch_size: int = 10000,
) -> dict[str, np.ndarray]:
    """Project a new AnnData object into a fitted scD2P reference space."""
    x_aligned = _align_to_reference_genes(adata, model, layer)
    n_cells = x_aligned.shape[0]

    z_basal = np.zeros((n_cells, model.n_basal), dtype=np.float32)
    h64 = model.h_basal_filtered.astype(np.float64, copy=False)

    for start, end in make_batches(n_cells, batch_size):
        x_batch = np.maximum(x_aligned[start:end, model.gene_mask], 0).astype(np.float64)
        w_batch, _, _ = non_negative_factorization(
            x_batch,
            W=None,
            H=h64,
            n_components=model.n_basal,
            init="custom",
            update_H=False,
            max_iter=100,
        )
        z_basal[start:end] = w_batch.astype(np.float32, copy=False)

    basal_pred = z_basal @ model.ridge_coef.T + model.ridge_intercept
    residual = x_aligned - basal_pred
    z_pert = (residual - model.ipca_mean) @ model.ipca_components.T
    z_pert = z_pert.astype(np.float32, copy=False)

    adata.obsm["X_zb"] = z_basal
    adata.obsm["X_zp"] = z_pert
    adata.obsm["X_z"] = np.hstack([z_basal, z_pert])

    return {"z_basal": z_basal, "z_pert": z_pert, "z": adata.obsm["X_z"]}
