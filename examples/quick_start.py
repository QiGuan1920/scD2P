"""Minimal scD2P workflow example."""

import scanpy as sc

from scD2P import fit_scD2P, save_model
from scD2P.enrichment import (
    annotate_programs_enrichment,
    export_enrichment_results,
    load_gmt,
)

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

gene_sets = load_gmt("data/h.all.v2026.1.Hs.symbols.gmt")
basal_annotations, basal_df = annotate_programs_enrichment(
    model.h_basal,
    model.var_names,
    gene_sets,
    program_prefix="basal",
)
export_enrichment_results(basal_df, basal_annotations, "outputs/enrichment", prefix="basal")

pert_annotations, pert_df = annotate_programs_enrichment(
    model.ipca_components,
    model.var_names,
    gene_sets,
    program_prefix="pert",
)
export_enrichment_results(pert_df, pert_annotations, "outputs/enrichment", prefix="pert")
