"""Flat parquet tables: one column per vector dimension, a description in the file.

Vectors are stored as plain Float32 columns ``e0000``, ``e0001``, ... so any tool
(polars, pandas, DuckDB, a spreadsheet) shows them without unpacking a list column.
Every file carries a ``description`` entry in its parquet key-value metadata saying
what a row and a column are:

    polars.read_parquet_metadata(path)["description"]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

VECTOR_PREFIX = "e"


def vector_columns(dim: int) -> list[str]:
    """``e0000`` ... ``e{dim-1}``."""
    return [f"{VECTOR_PREFIX}{i:04d}" for i in range(dim)]


def is_vector_column(name: str) -> bool:
    return name.startswith(VECTOR_PREFIX) and name[1:].isdigit()


def vector_frame(vectors: np.ndarray) -> pl.DataFrame:
    """A (n, d) float32 array as d Float32 columns."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-d array, got shape {vectors.shape}")
    return pl.DataFrame(
        {name: vectors[:, i] for i, name in enumerate(vector_columns(vectors.shape[1]))},
        schema={name: pl.Float32 for name in vector_columns(vectors.shape[1])},
    )


def vectors_of(frame: pl.DataFrame) -> np.ndarray:
    """The ``e*`` columns of a frame as a (n, d) float32 array, in column order."""
    cols = [c for c in frame.columns if is_vector_column(c)]
    if not cols:
        return np.zeros((frame.height, 0), dtype=np.float32)
    return frame.select(cols).to_numpy().astype(np.float32, copy=False)


def write_table(frame: pl.DataFrame, path: Path, description: str) -> None:
    """Write atomically (tmp + replace) with a ``description`` in the parquet metadata."""
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, metadata={"description": description})
    tmp.replace(path)
