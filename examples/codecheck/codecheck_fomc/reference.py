"""The committed reference run (``expected/``) as inputs: rescoring the author's embeddings.

``expected/{kind}/`` is a verbatim copy of each artifact directory of the author's
run, ``meta.json`` included. Two checks start from it instead of from the raw HTML:

    rescore --no-model   recompute the scores from the committed embeddings and the
                         committed narrative vectors (numpy, seconds, no model)
    rescore              re-embed the 10 narratives with the model and score the
                         committed embeddings; the new scores artifact records the
                         author's embeddings as its input (lineage)

Both first re-hash every committed file against its ``meta.json``: the inputs are
provably the author's, byte for byte.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from ledgerworks import Artifact, Ledger, hash_file, read_meta

from codecheck_fomc import PIPELINE_VERSION
from codecheck_fomc.compare import Report, _compare_scores
from codecheck_fomc.corpus import IngestJob
from codecheck_fomc.score import NARRATIVES_FILE, read_narratives, score_frame

LAYERS = {"fomc_statements": IngestJob.layer}  # the others: the ledger's default layer


@dataclass
class Reference:
    """One committed artifact directory, checked against its own meta.json."""

    kind: str
    artifact: Artifact
    checked: list[str]  # files whose blake2b matched meta.json
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def load_reference(expected_dir: Path, kind: str, layer: str = "derived") -> Reference:
    """Read ``expected/{kind}`` in place and re-hash every file it records."""
    directory = expected_dir / kind
    meta, hashes = read_meta(directory)
    checked, problems = [], []
    for name, recorded in sorted(hashes.items()):
        path = directory / name
        if not path.exists():
            problems.append(f"{kind}/{name}: missing")
            continue
        digest, _ = hash_file(path)
        if digest == recorded["digest"]:
            checked.append(name)
        else:
            problems.append(f"{kind}/{name}: blake2b differs from meta.json")
    artifact = Artifact(
        meta.artifact_id,
        LAYERS.get(kind, layer),
        directory,
        meta,
        {name: h["digest"] for name, h in hashes.items()},
    )
    return Reference(kind, artifact, checked, problems)


def short_commit(commit: str | None) -> str:
    """The first 10 characters of a commit, keeping a ``-dirty`` marker."""
    if not commit:
        return "(none)"
    return commit[:10] + ("-dirty" if commit.endswith("-dirty") else "")


def describe(ref: Reference) -> str:
    run = ref.artifact.meta.runs[-1]
    env = run.env or {}
    where = f"{env.get('platform', '?')}, Python {env.get('python', '?')}"
    return (
        f"{ref.kind}: {ref.artifact.artifact_id}\n"
        f"        made by commit {short_commit(run.pipeline_commit)} on {where}\n"
        f"        {len(ref.checked)}/{len(ref.artifact.file_hashes)} files match the "
        f"blake2b digests in its meta.json"
    )


def rescore_without_model(expected_dir: Path, show_year: str | None = None) -> Report:
    """Recompute every score table from the committed embeddings and narrative vectors."""
    report = Report()
    embeddings = load_reference(expected_dir, "fomc_embeddings")
    scores = load_reference(expected_dir, "fomc_narrative_scores")
    for ref in (embeddings, scores):
        report.lines.append(describe(ref))
        for problem in ref.problems:
            report.fail(problem)
    if report.passed is False:
        return report
    ids, nv = read_narratives(scores.artifact.path / NARRATIVES_FILE)
    if scores.artifact.meta.pipeline_version != PIPELINE_VERSION:
        report.fail(
            f"the reference was written by pipeline {scores.artifact.meta.pipeline_version}, "
            f"this code is {PIPELINE_VERSION} (different table layout); rebuild it with "
            "`all` and `freeze`"
        )
        return report
    report.lines.append(
        f"narrative vectors: {len(ids)} x {nv.shape[1]} read from "
        f"{expected_dir.name}/fomc_narrative_scores/{NARRATIVES_FILE} (not recomputed: no model)"
    )
    years = sorted(p.name for p in embeddings.artifact.files() if p.name[:4].isdigit())
    for name in years:
        paragraphs = pl.read_parquet(embeddings.artifact.path / name)
        recomputed = score_frame(paragraphs, ids, nv)
        committed = pl.read_parquet(scores.artifact.path / name)
        _compare_scores(report, "fomc_narrative_scores", name, recomputed, committed, 1e-6)
        if name == show_year:
            report.lines.append(f"\nrecomputed {name} (one row per statement):\n{recomputed}\n")
    return report


def register_reference(ledger: Ledger, expected_dir: Path, kinds: list[str]) -> list[Reference]:
    """Put the committed artifacts of ``kinds`` into ``ledger`` under their own ids.

    Each directory is re-hashed first and copied unchanged (meta.json included) to
    the place the ledger would have created it; the index is then rebuilt from the
    sidecars. An id already in the ledger is left as it is.
    """
    refs = [load_reference(expected_dir, kind, ledger.layer_for(kind)) for kind in kinds]
    bad = [p for ref in refs for p in ref.problems]
    if bad:
        raise ValueError("committed reference does not match its meta.json: " + "; ".join(bad))
    for ref in refs:
        art = ref.artifact
        if ledger.exists(art.artifact_id):
            continue
        target = ledger.artifact_dir(art.layer, ref.kind, art.artifact_id, art.meta.group)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(art.path, target)
    ledger.reindex()
    return refs
