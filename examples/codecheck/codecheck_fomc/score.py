"""fomc_narrative_scores: cosine scores of every statement against every narrative.

The taxonomy (``taxonomy/*.yaml``) gives each narrative a description and seed
phrases. They are embedded with the backend recorded on the embeddings artifact
(same model, same revision), averaged into one unit vector per narrative, and
written to ``narratives.parquet``. Then, per statement and narrative:

    cos_doc         cosine between the narrative and the mean of the statement's
                    paragraph vectors (re-normalised)
    cos_para_mean   mean over paragraphs of the paragraph/narrative cosine
    cos_para_max    the best paragraph

These are the raw scores; calibration or z-scoring is downstream work.
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

SCHEMA = {
    "doc_id": pl.String,
    "date": pl.Date,
    "narrative_id": pl.String,
    "cos_doc": pl.Float32,
    "cos_para_mean": pl.Float32,
    "cos_para_max": pl.Float32,
}


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
            frame = pl.DataFrame(
                {"narrative_id": self._ids, "vector": [v.tolist() for v in self._vectors]},
                schema={"narrative_id": pl.String, "vector": pl.List(pl.Float32)},
            )
            tmp = ctx.out_dir / "narratives.parquet.tmp"
            frame.write_parquet(tmp)
            tmp.replace(ctx.out_dir / "narratives.parquet")
        assert self._ids is not None
        return self._ids, self._vectors

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        ids, nv = self._narratives(ctx)
        paras = pl.read_parquet(self.source.path / f"{unit.key}.parquet").sort("doc_id", "para_idx")
        rows: list[dict[str, Any]] = []
        for (doc_id, date), group in paras.group_by(["doc_id", "date"], maintain_order=True):
            pv = np.array(group["embedding"].to_list(), dtype=np.float32)  # (p, d)
            doc = normalise(pv.mean(axis=0, keepdims=True))[0]
            sims = pv @ nv.T  # (p, n), both unit norm
            doc_sims = nv @ doc
            for j, nid in enumerate(ids):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "date": date,
                        "narrative_id": nid,
                        "cos_doc": float(doc_sims[j]),
                        "cos_para_mean": float(sims[:, j].mean()),
                        "cos_para_max": float(sims[:, j].max()),
                    }
                )
        out = pl.DataFrame(rows, schema=SCHEMA)
        tmp = ctx.out_dir / f"{unit.key}.parquet.tmp"
        out.write_parquet(tmp)
        tmp.replace(ctx.out_dir / f"{unit.key}.parquet")
        ctx.log.info("%s: %d scores", unit.key, out.height)

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
    if not (artifact.path / "narratives.parquet").exists():
        findings.append(Finding(Severity.ERROR, artifact.artifact_id, "narratives.parquet missing"))
    for path in artifact.files("*.parquet"):
        if path.name == "narratives.parquet":
            continue
        frame = pl.read_parquet(path)
        for col in ("cos_doc", "cos_para_mean", "cos_para_max"):
            if frame.height and ((frame[col] < -1.001) | (frame[col] > 1.001)).any():
                findings.append(
                    Finding(
                        Severity.ERROR, artifact.artifact_id, f"{path.name}: {col} out of [-1, 1]"
                    )
                )
    return findings
