"""Utility functions for scD2P."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_dense_array(x: Any, dtype=np.float32) -> np.ndarray:
    """Convert a dense or sparse matrix-like object to a NumPy array."""
    if hasattr(x, "toarray"):
        return x.toarray().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def get_layer_matrix(adata: Any, layer: str | None = "lognorm") -> Any:
    """Return the requested AnnData layer, or ``adata.X`` when ``layer`` is None."""
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' was not found in adata.layers.")
    return adata.layers[layer]


def validate_obs_columns(adata: Any, columns: list[str]) -> None:
    """Validate that required columns exist in ``adata.obs``."""
    missing = [col for col in columns if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required columns in adata.obs: {missing}")


def make_batches(n_items: int, batch_size: int):
    """Yield ``(start, end)`` index ranges."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)
