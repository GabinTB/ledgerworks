"""Parquet helpers: vectors in one list column, a description in every file.

A vector is stored in a single ``List(Float32)`` column (``embedding``, ``vector``).
Every file carries a ``description`` entry in its parquet key-value metadata saying
what a row and a column are:

    polars.read_parquet_metadata(path)["description"]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

VECTOR = pl.List(pl.Float32)


def vector_series(name: str, vectors: np.ndarray) -> pl.Series:
    """A (n, d) array as one ``List(Float32)`` column."""
    return pl.Series(name, [v.tolist() for v in np.asarray(vectors, np.float32)], dtype=VECTOR)


def vectors_of(frame: pl.DataFrame, column: str) -> np.ndarray:
    """A ``List(Float32)`` column as a (n, d) float32 array."""
    if frame.height == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.array(frame[column].to_list(), dtype=np.float32)


def write_table(frame: pl.DataFrame, path: Path, description: str) -> None:
    """Write atomically (tmp + replace) with a ``description`` in the parquet metadata."""
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, metadata={"description": description})
    tmp.replace(path)
