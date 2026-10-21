"""fomc_embeddings: one vector per statement paragraph, one parquet per year.

One row per paragraph: ``date``, ``doc_id``, ``para_idx`` (0-based position in the
statement), ``text`` (the paragraph that was embedded), ``text_sha256`` and
``embedding`` (the unit-norm vector, a list of 1024 Float32). Paragraphs are the
unit because FOMC statements are written as short, self-contained paragraphs and
the model was trained at 512 tokens.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from ledgerworks import Artifact, Ledger, ModelCard
from ledgerworks.jobs import Job, JobContext, Unit, register_job
from ledgerworks.layout import Layout, layout_of
from ledgerworks.verify import Finding, Severity

from codecheck_fomc import PIPELINE, PIPELINE_REPO, PIPELINE_VERSION
from codecheck_fomc.backends import Backend, make_backend
from codecheck_fomc.tables import VECTOR, vector_series, vectors_of, write_table

KEYS = {
    "date": pl.Date,
    "doc_id": pl.String,
    "para_idx": pl.Int64,
    "text": pl.String,
    "text_sha256": pl.String,
    "embedding": VECTOR,
}
DESCRIPTION = (
    "fomc_embeddings: one row per statement paragraph. date, doc_id and para_idx "
    "(0-based position in the statement) identify it, text is the paragraph that was "
    "embedded, embedding is its unit-norm vector (list of Float32)."
)


def backend_params(backend: Backend) -> dict[str, Any]:
    """The backend settings that define the artifact (resume rebuilds the backend from them)."""
    ident = backend.identity()
    if ident["backend"] == "hashing":
        return {"backend": "hashing", "dim": ident["dim"]}
    return {
        "backend": "finmodernbert",
        "model_id": ident["model_id"],
        "revision": ident["revision"],
        "max_seq_length": ident["max_seq_length"],
    }


def backend_from_params(hp: dict[str, Any], **overrides: Any) -> Backend:
    hp = dict(hp)
    name = hp.pop("backend")
    if name == "hashing":
        return make_backend("hashing", dim=int(hp["dim"]))
    return make_backend(
        "finmodernbert",
        model_id=hp["model_id"],
        revision=hp["revision"],
        max_seq_length=int(hp["max_seq_length"]),
        **overrides,
    )


@register_job
class EmbedJob(Job):
    kind = "fomc_embeddings"
    pipeline = PIPELINE
    pipeline_repo = PIPELINE_REPO
    pipeline_version = PIPELINE_VERSION

    def __init__(self, source: Artifact, backend: Backend) -> None:
        self.source, self.backend = source, backend
        self.layout: Layout = layout_of(source)

    def params(self) -> dict[str, Any]:
        return {**self.layout.hyperparams(), **backend_params(self.backend), "unit": "paragraph"}

    def sources(self) -> list[Any]:
        return [self.source]

    def model_card(self) -> ModelCard | None:
        return self.backend.model_card()

    def backends(self) -> list[Any]:
        return [self.backend]

    def units(self) -> list[Unit]:
        return [Unit(key) for key in self.layout.existing(self.source.path)]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / f"{unit.key}.parquet").exists()

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        docs = pl.read_parquet(self.source.path / f"{unit.key}.parquet")
        rows = docs.select("doc_id", "date", "paragraphs").explode("paragraphs")
        rows = rows.with_columns(
            pl.int_range(pl.len()).over("doc_id").alias("para_idx"),
            pl.col("paragraphs").alias("text"),
        )
        texts = rows["text"].to_list()
        vectors = self.backend.encode(texts) if texts else np.zeros((0, 0), np.float32)
        out = pl.DataFrame(
            {
                "date": rows["date"],
                "doc_id": rows["doc_id"],
                "para_idx": rows["para_idx"],
                "text": texts,
                "text_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
                "embedding": vector_series("embedding", vectors),
            },
            schema=KEYS,
        )
        write_table(out, ctx.out_dir / f"{unit.key}.parquet", DESCRIPTION)
        ctx.log.info("%s: %d paragraphs from %d statements", unit.key, out.height, docs.height)

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: Ledger, **kwargs: Any) -> EmbedJob:
        source = index.get(artifact.meta.sources[0])
        return cls(source, backend_from_params(artifact.meta.hyperparams, **kwargs))


def verify_artifact(artifact: Artifact) -> list[Finding]:
    findings: list[Finding] = []
    dim = artifact.meta.model_card.dim if artifact.meta.model_card else None
    for path in artifact.files("*.parquet"):
        frame = pl.read_parquet(path)
        if dim and (frame["embedding"].list.len() != dim).any():
            findings.append(
                Finding(
                    Severity.ERROR, artifact.artifact_id, f"{path.name}: vectors not of dim {dim}"
                )
            )
        norms = np.linalg.norm(vectors_of(frame, "embedding"), axis=1)
        if frame.height and not np.allclose(norms, 1.0, atol=1e-3):
            findings.append(
                Finding(Severity.ERROR, artifact.artifact_id, f"{path.name}: vectors not unit norm")
            )
    return findings
