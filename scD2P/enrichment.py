"""Pathway enrichment and program annotation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def load_gmt(path: str | Path) -> dict[str, list[str]]:
    """Load a GMT file as ``{gene_set_name: [genes]}``.

    The second field in a GMT row is a description and is ignored.
    Gene symbols are converted to uppercase for robust matching.
    """
    gene_sets: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [gene.upper() for gene in parts[2:] if gene]
            gene_sets[name] = genes
    return gene_sets


def _as_upper_array(values: Iterable[str]) -> np.ndarray:
    """Convert an iterable of gene names to an uppercase NumPy array."""
    return np.asarray([str(value).upper() for value in values])


def annotate_programs_enrichment(
    gene_loadings: np.ndarray,
    gene_names: Iterable[str],
    gene_sets: dict[str, list[str]],
    program_prefix: str = "prog",
    top_n_genes: int = 100,
    top_k_display: int = 5,
    pval_cutoff: float = 0.05,
    min_geneset_size: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    """Annotate programs using Fisher exact test-based gene set enrichment.

    For each program, genes with the largest absolute loadings are selected as
    representative program genes. Each program gene set is then tested against
    predefined pathway gene sets using a one-sided Fisher exact test. P-values
    are adjusted by the Benjamini-Hochberg procedure.
    """
    gene_names_upper = _as_upper_array(gene_names)
    n_programs = gene_loadings.shape[0]
    n_background = len(gene_names_upper)
    background = set(gene_names_upper)

    gene_set_names = list(gene_sets.keys())
    gene_sets_in_background = {
        name: set(_as_upper_array(genes)) & background for name, genes in gene_sets.items()
    }

    all_results = []

    for program_idx in range(n_programs):
        abs_loading = np.abs(gene_loadings[program_idx])
        top_idx = np.argsort(abs_loading)[::-1][:top_n_genes]
        program_genes = set(gene_names_upper[top_idx])

        loading_signs = gene_loadings[program_idx][top_idx]
        program_genes_pos = set(gene_names_upper[top_idx[loading_signs > 0]])
        program_genes_neg = set(gene_names_upper[top_idx[loading_signs < 0]])

        pvals = []
        odds_ratios = []
        overlaps = []
        geneset_sizes = []
        overlap_gene_text = []
        directions = []

        for gene_set_name in gene_set_names:
            pathway_genes = gene_sets_in_background[gene_set_name]
            pathway_size = len(pathway_genes)

            if pathway_size < min_geneset_size:
                pvals.append(1.0)
                odds_ratios.append(0.0)
                overlaps.append(0)
                geneset_sizes.append(pathway_size)
                overlap_gene_text.append("")
                directions.append("none")
                continue

            overlap = program_genes & pathway_genes
            n_overlap = len(overlap)

            a = n_overlap
            b = len(program_genes) - n_overlap
            c = pathway_size - n_overlap
            d = n_background - len(program_genes) - c
            table = [[a, b], [c, d]]
            odds, pval = fisher_exact(table, alternative="greater")

            pvals.append(pval)
            odds_ratios.append(100.0 if np.isinf(odds) else odds)
            overlaps.append(n_overlap)
            geneset_sizes.append(pathway_size)
            overlap_gene_text.append(";".join(sorted(overlap)[:20]))

            n_pos = len(overlap & program_genes_pos)
            n_neg = len(overlap & program_genes_neg)
            if n_pos > n_neg:
                directions.append("up")
            elif n_neg > n_pos:
                directions.append("down")
            else:
                directions.append("mixed")

        _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")

        for gene_set_idx, gene_set_name in enumerate(gene_set_names):
            all_results.append(
                {
                    "program": f"{program_prefix}_{program_idx}",
                    "program_idx": program_idx,
                    "geneset": gene_set_name,
                    "geneset_short": gene_set_name.replace("HALLMARK_", ""),
                    "pval": pvals[gene_set_idx],
                    "pval_adj": pvals_adj[gene_set_idx],
                    "neg_log10_padj": -np.log10(pvals_adj[gene_set_idx] + 1e-300),
                    "odds_ratio": odds_ratios[gene_set_idx],
                    "n_overlap": overlaps[gene_set_idx],
                    "n_prog_genes": len(program_genes),
                    "n_geneset_in_bg": geneset_sizes[gene_set_idx],
                    "n_background": n_background,
                    "direction": directions[gene_set_idx],
                    "overlap_genes": overlap_gene_text[gene_set_idx],
                }
            )

    enrich_df = pd.DataFrame(all_results)
    annotations = summarize_program_annotations(
        enrich_df,
        program_prefix=program_prefix,
        top_k_display=top_k_display,
        pval_cutoff=pval_cutoff,
    )
    return annotations, enrich_df


def summarize_program_annotations(
    enrich_df: pd.DataFrame,
    program_prefix: str = "prog",
    top_k_display: int = 5,
    pval_cutoff: float = 0.05,
) -> list[str]:
    """Create compact pathway annotations for each program."""
    annotations = []
    program_ids = sorted(enrich_df["program_idx"].unique())

    for program_idx in program_ids:
        df_program = enrich_df[enrich_df["program_idx"] == program_idx].copy()
        df_sig = (
            df_program[df_program["pval_adj"] < pval_cutoff]
            .sort_values(["pval_adj", "odds_ratio"], ascending=[True, False])
            .head(top_k_display)
        )

        if len(df_sig) == 0:
            annotations.append(f"{program_prefix}_{program_idx} | no significant enrichment")
            continue

        parts = []
        for _, row in df_sig.iterrows():
            parts.append(
                f"{row['geneset_short']}(OR={row['odds_ratio']:.1f}, "
                f"q={row['pval_adj']:.1e}, "
                f"n={int(row['n_overlap'])}/{int(row['n_geneset_in_bg'])}, "
                f"{row['direction']})"
            )
        annotations.append(f"{program_prefix}_{program_idx} | {' / '.join(parts)}")

    return annotations


def export_enrichment_results(
    enrich_df: pd.DataFrame,
    annotations: list[str],
    out_dir: str | Path,
    prefix: str,
    pval_cutoff: float = 0.05,
) -> dict[str, Path]:
    """Export full enrichment results, significant hits, annotations, and heatmap matrix."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / f"{prefix}_enrichment_full.csv"
    sig_path = out_dir / f"{prefix}_enrichment_significant.csv"
    annot_path = out_dir / f"{prefix}_annotations.csv"
    heatmap_path = out_dir / f"{prefix}_enrichment_heatmap.csv"

    enrich_df.to_csv(full_path, index=False)
    sig_df = enrich_df[enrich_df["pval_adj"] < pval_cutoff].sort_values(
        ["program_idx", "pval_adj"]
    )
    sig_df.to_csv(sig_path, index=False)
    pd.DataFrame(
        {
            "program": [f"{prefix}_{idx}" for idx in range(len(annotations))],
            "annotation": annotations,
        }
    ).to_csv(annot_path, index=False)
    heatmap = enrich_df.pivot_table(
        index="program", columns="geneset_short", values="neg_log10_padj", fill_value=0
    )
    heatmap.to_csv(heatmap_path)

    return {
        "full": full_path,
        "significant": sig_path,
        "annotations": annot_path,
        "heatmap": heatmap_path,
    }
