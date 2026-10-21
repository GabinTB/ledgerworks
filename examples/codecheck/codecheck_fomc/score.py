"""fomc_narrative_scores: how close every statement is to every narrative.

The taxonomy (``taxonomy/*.yaml``) gives each narrative a description and seed
phrases. They are embedded with the backend recorded on the embeddings artifact
(same model, same revision), averaged into one unit vector per narrative, and
written to ``narratives.parquet`` (one row per narrative: id, label, vector).

Scores, one parquet per year, one row per statement (``date``), one column per narrative:

    score[date, narrative] = mean over the statement's paragraphs p of cos(p, narrative)

Paragraph and narrative vectors are unit norm, so each cosine is a dot product.
These are raw scores; calibration or z-scoring is downstream work.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from ledgerworks import Artifact, Ledger
from ledgerworks.jobs import Job, JobContext, Unit, register_job
from ledgerworks.layout import Layout, layout_of
from ledgerworks.verify import Finding, Severity

from codecheck_fomc import PIPELINE, PIPELINE_REPO, PIPELINE_VERSION, TAXONOMY_DIR
from codecheck_fomc.backends import Backend, normalise
from codecheck_fomc.embed import backend_from_params
from codecheck_fomc.tables import VECTOR, vector_series, vectors_of, write_table

NARRATIVES_FILE = "narratives.parquet"
DESCRIPTION = (
    "fomc_narrative_scores: one row per FOMC statement (its date), one column per "
    "narrative of the taxonomy. Each value is the mean, over the statement's paragraphs, "
    "of the cosine between the paragraph embedding and the narrative vector."
)
NARRATIVES_DESCRIPTION = (
    "Narrative vectors: one row per narrative of the taxonomy (narrative_id, label); "
    "vector is the unit-norm mean of the embeddings of its description and seed phrases."
)


@dataclass(frozen=True)
class Taxonomy:
    name: str
    version: str
    narratives: list[dict[str, Any]]
    sha256: str
    path: Path

    @classmethod
    def load(cls, path: Path) -> Taxonomy:
        body = path.read_bytes()
        doc = yaml.safe_load(body)
        return cls(
            doc["name"],
            str(doc["version"]),
            list(doc["narratives"]),
            hashlib.sha256(body).hexdigest(),
            path,
        )

    def texts(self, narrative: dict[str, Any]) -> list[str]:
        return [narrative["description"].strip(), *narrative.get("seeds", [])]


def narrative_vectors(taxonomy: Taxonomy, backend: Backend) -> tuple[list[str], np.ndarray]:
    ids, vectors = [], []
    for narrative in taxonomy.narratives:
        vecs = backend.encode(taxonomy.texts(narrative))
        ids.append(narrative["id"])
        vectors.append(vecs.mean(axis=0))
    return ids, normalise(np.stack(vectors))


def narratives_frame(taxonomy: Taxonomy, ids: list[str], vectors: np.ndarray) -> pl.DataFrame:
    labels = {n["id"]: n.get("label", n["id"]) for n in taxonomy.narratives}
    return pl.DataFrame(
        {
            "narrative_id": ids,
            "label": [labels[i] for i in ids],
            "vector": vector_series("vector", vectors),
        },
        schema={"narrative_id": pl.String, "label": pl.String, "vector": VECTOR},
    )


def read_narratives(path: Path) -> tuple[list[str], np.ndarray]:
    """Narrative ids and their (n, d) vectors from a ``narratives.parquet``."""
    frame = pl.read_parquet(path)
    return frame["narrative_id"].to_list(), vectors_of(frame, "vector")


def score_frame(paragraphs: pl.DataFrame, ids: list[str], nv: np.ndarray) -> pl.DataFrame:
    """One row per statement (``date``), one Float32 column per narrative.

    ``paragraphs`` is a fomc_embeddings table (``date``, ``doc_id``, ``para_idx``,
    ``embedding``); ``nv`` the (n, d) unit narrative vectors in the order of ``ids``.
    """
    paragraphs = paragraphs.sort("date", "doc_id", "para_idx")
    pv = vectors_of(paragraphs, "embedding")  # (p, d), unit norm
    if pv.shape[1] != nv.shape[1]:
        raise ValueError(
            f"paragraph vectors of dim {pv.shape[1]}, narrative vectors of dim {nv.shape[1]}: "
            "they must come from the same model"
        )
    sims = pv @ nv.T  # (p, n): cosine of every paragraph with every narrative
    per_para = pl.concat(
        [
            paragraphs.select("date"),
            pl.DataFrame(
                {nid: sims[:, j] for j, nid in enumerate(ids)},
                schema={nid: pl.Float32 for nid in ids},
            ),
        ],
        how="horizontal",
    )
    return (
        per_para.group_by("date", maintain_order=True)
        .agg(pl.col(ids).mean())
        .with_columns(pl.col(ids).cast(pl.Float32))
    )


@register_job
class ScoreJob(Job):
    kind = "fomc_narrative_scores"
    pipeline = PIPELINE
    pipeline_repo = PIPELINE_REPO
    pipeline_version = PIPELINE_VERSION

    def __init__(self, source: Artifact, taxonomy: Taxonomy, backend: Backend) -> None:
        self.source, self.taxonomy, self.backend = source, taxonomy, backend
        self.layout: Layout = layout_of(source)
        self._ids: list[str] | None = None
        self._vectors: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {
            **self.layout.hyperparams(),
            "taxonomy": self.taxonomy.name,
            "taxonomy_version": self.taxonomy.version,
            "taxonomy_sha256": self.taxonomy.sha256,
            "narrative_vector": "mean of description and seeds",
        }

    def sources(self) -> list[Any]:
        return [self.source]

    def model_card(self):
        return self.source.meta.model_card

    def backends(self) -> list[Any]:
        return [self.backend]

    def units(self) -> list[Unit]:
        return [Unit(key) for key in self.layout.existing(self.source.path)]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / f"{unit.key}.parquet").exists()

    def _narratives(self, ctx: JobContext) -> tuple[list[str], np.ndarray]:
        if self._vectors is None:
            self._ids, self._vectors = narrative_vectors(self.taxonomy, self.backend)
            frame = narratives_frame(self.taxonomy, self._ids, self._vectors)
            write_table(frame, ctx.out_dir / NARRATIVES_FILE, NARRATIVES_DESCRIPTION)
        assert self._ids is not None
        return self._ids, self._vectors

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        ids, nv = self._narratives(ctx)
        paragraphs = pl.read_parquet(self.source.path / f"{unit.key}.parquet")
        out = score_frame(paragraphs, ids, nv)
        write_table(out, ctx.out_dir / f"{unit.key}.parquet", DESCRIPTION)
        ctx.log.info("%s: %d statements x %d narratives", unit.key, out.height, len(ids))

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: Ledger, **kwargs: Any) -> ScoreJob:
        source = index.get(artifact.meta.sources[0])
        hp = artifact.meta.hyperparams
        taxonomy = Taxonomy.load(TAXONOMY_DIR / f"{hp['taxonomy']}.yaml")
        if taxonomy.sha256 != hp["taxonomy_sha256"]:
            raise ValueError("the taxonomy file changed since the artifact was started")
        return cls(source, taxonomy, backend_from_params(source.meta.hyperparams, **kwargs))


def verify_artifact(artifact: Artifact) -> list[Finding]:
    findings: list[Finding] = []
    if not (artifact.path / NARRATIVES_FILE).exists():
        findings.append(Finding(Severity.ERROR, artifact.artifact_id, f"{NARRATIVES_FILE} missing"))
        return findings
    ids, _ = read_narratives(artifact.path / NARRATIVES_FILE)
    for path in artifact.files("*.parquet"):
        if path.name == NARRATIVES_FILE:
            continue
        frame = pl.read_parquet(path)
        if frame.columns != ["date", *ids]:
            findings.append(
                Finding(
                    Severity.ERROR,
                    artifact.artifact_id,
                    f"{path.name}: columns are not date and the {len(ids)} narratives",
                )
            )
            continue
        for col in ids:
            if frame.height and ((frame[col] < -1.001) | (frame[col] > 1.001)).any():
                findings.append(
                    Finding(
                        Severity.ERROR, artifact.artifact_id, f"{path.name}: {col} out of [-1, 1]"
                    )
                )
    return findings
