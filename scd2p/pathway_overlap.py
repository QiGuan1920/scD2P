"""Pathway-level reproducibility analysis for scD2P programs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_program_to_pathways(df_sig: pd.DataFrame, pathway_col: str = "geneset_short") -> dict[int, set[str]]:
    """Build a mapping from program index to significant pathway names."""
    mapping: dict[int, set[str]] = {}
    if df_sig is None or len(df_sig) == 0:
        return mapping
    for program_idx, sub_df in df_sig.groupby("program_idx"):
        mapping[int(program_idx)] = set(sub_df[pathway_col].astype(str))
    return mapping


def active_programs(
    z_mean: np.ndarray,
    mode: str = "top_k",
    top_k: int = 5,
    z_thresh: float = 0.0,
    use_abs: bool = True,
) -> list[int]:
    """Select active programs from a group-level mean program score vector."""
    score = np.abs(z_mean) if use_abs else z_mean
    if mode == "top_k":
        k = min(top_k, len(score))
        return list(np.argsort(score)[::-1][:k])
    if mode == "threshold":
        return list(np.where(score > z_thresh)[0])
    raise ValueError(f"Unsupported active program selection mode: {mode}")


def pathways_of(program_indices: list[int], program_to_pathways: dict[int, set[str]]) -> set[str]:
    """Return the union of pathways associated with selected programs."""
    pathways: set[str] = set()
    for program_idx in program_indices:
        pathways |= program_to_pathways.get(int(program_idx), set())
    return pathways


def overlap_rates(set1: set[str], set2: set[str]) -> dict[str, float | int]:
    """Compute Jaccard index and overlap coefficient between two pathway sets."""
    intersection = set1 & set2
    union = set1 | set2
    n1 = len(set1)
    n2 = len(set2)
    n_intersection = len(intersection)
    n_union = len(union)
    min_size = min(n1, n2)
    return {
        "jaccard": n_intersection / n_union if n_union > 0 else np.nan,
        "overlap_coef": n_intersection / min_size if min_size > 0 else np.nan,
        "n_path_rep1": n1,
        "n_path_rep2": n2,
        "n_overlap": n_intersection,
        "n_union": n_union,
    }


def pathway_overlap_reproducibility(
    adata,
    sig_csv_basal: str | Path,
    sig_csv_pert: str | Path,
    export_dir: str | Path = "model_export/pathway_overlap",
    pathway_col: str = "geneset_short",
    active_mode: str = "top_k",
    top_k: int = 10,
    z_thresh: float = 0.0,
    use_abs: bool = True,
    min_cells: int = 5,
    cell_line_col: str = "cell_line",
    perturbation_col: str = "perturbation",
    replicate_col: str = "replicate",
    is_ctrl_col: str = "is_ctrl",
) -> dict[str, pd.DataFrame | str]:
    """Assess pathway-level reproducibility between two replicates.

    Control cells are evaluated at the cell-line level using basal scores.
    Drug-treated cells are evaluated at the cell-line-drug level using
    perturbation scores.
    """
    obs = adata.obs.copy().reset_index(drop=True)
    required_cols = [cell_line_col, perturbation_col, replicate_col, is_ctrl_col]
    missing = [col for col in required_cols if col not in obs.columns]
    if missing:
        raise KeyError(f"Missing required columns in adata.obs: {missing}")

    replicates = sorted(obs[replicate_col].unique())
    if len(replicates) != 2:
        raise ValueError(f"Expected exactly two replicates, found: {replicates}")
    rep1, rep2 = replicates

    is_control = obs[is_ctrl_col].values.astype(bool)
    is_drug = ~is_control
    cell_lines = obs[cell_line_col].values
    perturbations = obs[perturbation_col].values
    reps = obs[replicate_col].values

    z_basal = np.asarray(adata.obsm["X_zb"], dtype=np.float32)
    z_pert = np.asarray(adata.obsm["X_zp"], dtype=np.float32)

    program_to_path_basal = build_program_to_pathways(pd.read_csv(sig_csv_basal), pathway_col)
    program_to_path_pert = build_program_to_pathways(pd.read_csv(sig_csv_pert), pathway_col)

    def group_pathways(z_all: np.ndarray, mask: np.ndarray, program_to_pathways: dict[int, set[str]]):
        z_mean = z_all[mask].mean(axis=0)
        programs = active_programs(
            z_mean,
            mode=active_mode,
            top_k=top_k,
            z_thresh=z_thresh,
            use_abs=use_abs,
        )
        return pathways_of(programs, program_to_pathways), programs

    ctrl_rows = []
    for cell_line in sorted(obs.loc[is_control, cell_line_col].unique()):
        mask_r1 = is_control & (cell_lines == cell_line) & (reps == rep1)
        mask_r2 = is_control & (cell_lines == cell_line) & (reps == rep2)
        if mask_r1.sum() < min_cells or mask_r2.sum() < min_cells:
            continue
        set1, programs1 = group_pathways(z_basal, mask_r1, program_to_path_basal)
        set2, programs2 = group_pathways(z_basal, mask_r2, program_to_path_basal)
        metrics = overlap_rates(set1, set2)
        ctrl_rows.append(
            {
                "cell_line": cell_line,
                "n_rep1": int(mask_r1.sum()),
                "n_rep2": int(mask_r2.sum()),
                "programs_rep1": ";".join(map(str, programs1)),
                "programs_rep2": ";".join(map(str, programs2)),
                "shared_pathways": ";".join(sorted(set1 & set2)),
                **metrics,
            }
        )

    drug_rows = []
    for cell_line in sorted(obs.loc[is_drug, cell_line_col].unique()):
        cell_line_mask = is_drug & (cell_lines == cell_line)
        for drug in np.unique(perturbations[cell_line_mask]):
            mask_r1 = is_drug & (cell_lines == cell_line) & (perturbations == drug) & (reps == rep1)
            mask_r2 = is_drug & (cell_lines == cell_line) & (perturbations == drug) & (reps == rep2)
            if mask_r1.sum() < min_cells or mask_r2.sum() < min_cells:
                continue
            set1, programs1 = group_pathways(z_pert, mask_r1, program_to_path_pert)
            set2, programs2 = group_pathways(z_pert, mask_r2, program_to_path_pert)
            metrics = overlap_rates(set1, set2)
            drug_rows.append(
                {
                    "cell_line": cell_line,
                    "drug": drug,
                    "n_rep1": int(mask_r1.sum()),
                    "n_rep2": int(mask_r2.sum()),
                    "programs_rep1": ";".join(map(str, programs1)),
                    "programs_rep2": ";".join(map(str, programs2)),
                    "shared_pathways": ";".join(sorted(set1 & set2)),
                    **metrics,
                }
            )

    df_ctrl = pd.DataFrame(ctrl_rows)
    df_drug = pd.DataFrame(drug_rows)
    summary = pd.DataFrame(
        [
            {
                "type": "ctrl/basal",
                "level": "cell_line",
                "n": len(df_ctrl),
                "jaccard_mean": df_ctrl["jaccard"].mean() if len(df_ctrl) else np.nan,
                "jaccard_std": df_ctrl["jaccard"].std() if len(df_ctrl) else np.nan,
                "overlap_coef_mean": df_ctrl["overlap_coef"].mean() if len(df_ctrl) else np.nan,
            },
            {
                "type": "drug/pert",
                "level": "cell_line,drug",
                "n": len(df_drug),
                "jaccard_mean": df_drug["jaccard"].mean() if len(df_drug) else np.nan,
                "jaccard_std": df_drug["jaccard"].std() if len(df_drug) else np.nan,
                "overlap_coef_mean": df_drug["overlap_coef"].mean() if len(df_drug) else np.nan,
            },
        ]
    )

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    df_ctrl.to_csv(export_dir / "ctrl_overlap.csv", index=False)
    df_drug.to_csv(export_dir / "drug_overlap.csv", index=False)
    summary.to_csv(export_dir / "summary.csv", index=False)

    return {
        "df_ctrl": df_ctrl,
        "df_drug": df_drug,
        "summary": summary,
        "rep1": str(rep1),
        "rep2": str(rep2),
    }
