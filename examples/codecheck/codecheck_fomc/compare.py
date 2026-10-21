"""Compare a fresh run against the committed reference outputs in ``expected/``.

``expected/{kind}/`` is a verbatim copy of the author's artifact directory
(``meta.json`` included, so the reviewer also sees the author's commit, model
card and file hashes). ``compare`` joins each table on its keys and checks:

    fomc_statements          identical rows (text and HTML digest)
    fomc_embeddings          same paragraphs (text digest); vectors equal within
                             ``atol`` element-wise and cosine >= ``min_cos``
    fomc_narrative_scores    scores equal within ``atol``

Floating-point outputs of a neural model are reproducible to tolerance across
machines, not bit for bit, hence the tolerances. The hashing backend is exact.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from ledgerworks import Artifact, Ledger, read_meta

from codecheck_fomc.tables import vectors_of

KINDS = ("fomc_statements", "fomc_embeddings", "fomc_narrative_scores")

# What each table is checked for, printed above its results.
CHECKS = {
    "fomc_statements": "check: every row identical (statement text and HTML digest); exact, "
    "parsing is deterministic",
    "fomc_embeddings": "check: same paragraphs (text digest), vectors within atol element-wise "
    "and cosine >= min_cos; float tolerance, the model's sums are reordered across CPUs",
    "fomc_narrative_scores": "check: same statements and narratives, every score within atol; "
    "narratives.parquet vectors as for the embeddings",
}


@dataclass
class Report:
    passed: bool = True
    no_reference: bool = False
    lines: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.lines.append(f"  ok    {msg}")

    def fail(self, msg: str) -> None:
        self.passed = False
        self.lines.append(f"  FAIL  {msg}")

    def __str__(self) -> str:
        return "\n".join(self.lines)


def freeze(ledger: Ledger, expected_dir: Path) -> dict[str, str]:
    """Copy the latest artifact of every kind into ``expected/`` (the reference run)."""
    ids: dict[str, str] = {}
    for kind in KINDS:
        art = ledger.latest(kind)
        target = expected_dir / kind
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(art.path, target, ignore=shutil.ignore_patterns("job.*", "PAUSE"))
        ids[kind] = art.artifact_id
    (expected_dir / "RUN.json").write_text(json.dumps(ids, indent=2) + "\n")
    return ids


def _files(path: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(path.glob("*.parquet"))}


def compare(
    ledger: Ledger,
    expected_dir: Path,
    *,
    atol: float = 1e-4,
    min_cos: float = 0.9999,
    kinds: tuple[str, ...] = KINDS,
) -> Report:
    report = Report()
    if not any((expected_dir / kind).exists() for kind in kinds):
        # first run: nothing frozen yet, so nothing to fail against
        report.no_reference = True
        report.lines.append(f"no reference yet in {expected_dir}, run freeze")
        return report
    for kind in kinds:
        exp_dir = expected_dir / kind
        if not exp_dir.exists():
            report.fail(f"{kind}: no reference in {expected_dir}")
            continue
        try:
            art: Artifact = ledger.latest(kind)
        except Exception as exc:  # noqa: BLE001
            report.fail(f"{kind}: no complete artifact in the ledger ({exc})")
            continue
        exp_meta, exp_hashes = read_meta(exp_dir)
        report.lines.append(
            f"{kind}: {art.artifact_id}\n"
            f"        reference {exp_meta.artifact_id} (commit {exp_meta.pipeline_commit})\n"
            f"        {CHECKS[kind]}"
        )
        if exp_meta.pipeline_version != art.meta.pipeline_version:
            report.fail(
                f"{kind}: reference written by pipeline {exp_meta.pipeline_version}, this run "
                f"by {art.meta.pipeline_version}: the table layouts differ, rebuild the reference"
            )
            continue
        got, exp = _files(art.path), _files(exp_dir)
        if set(got) != set(exp):
            report.fail(
                f"{kind}: files {sorted(set(got) ^ set(exp))} differ between run and reference"
            )
            continue
        for name in sorted(exp):
            a, b = pl.read_parquet(got[name]), pl.read_parquet(exp[name])
            if a.height != b.height:
                report.fail(f"{kind}/{name}: {a.height} rows vs {b.height} in reference")
                continue
            if kind == "fomc_statements":
                _compare_statements(report, kind, name, a, b)
            elif kind == "fomc_embeddings":
                _compare_embeddings(report, kind, name, a, b, atol, min_cos)
            elif name == "narratives.parquet":
                _compare_vectors(report, kind, name, a, b, "narrative_id", atol, min_cos)
            else:
                _compare_scores(report, kind, name, a, b, atol)
        exact = [n for n, h in exp_hashes.items() if art.file_hashes.get(n) == h["digest"]]
        report.lines.append(
            f"        {len(exact)}/{len(exp_hashes)} files hash-identical to the reference"
        )
    return report


def _compare_statements(report, kind, name, a, b) -> None:
    cols = ["doc_id", "date", "html_sha256", "text"]
    if a.select(cols).sort("doc_id").equals(b.select(cols).sort("doc_id")):
        report.ok(f"{kind}/{name}: {a.height} statements identical")
    else:
        report.fail(f"{kind}/{name}: statement rows differ")


def _aligned(a: pl.DataFrame, b: pl.DataFrame, keys: list[str]):
    """Both frames sorted on ``keys``, or None if their key sets differ."""
    a, b = a.sort(keys), b.sort(keys)
    return (a, b) if a.select(keys).equals(b.select(keys)) else None


def _compare_embeddings(report, kind, name, a, b, atol, min_cos) -> None:
    pair = _aligned(a, b, ["doc_id", "para_idx"])
    if pair is None:
        report.fail(f"{kind}/{name}: paragraph keys differ")
        return
    a, b = pair
    if not a["text_sha256"].equals(b["text_sha256"]):
        report.fail(f"{kind}/{name}: paragraph text differs from the reference")
        return
    x, y = vectors_of(a, "embedding"), vectors_of(b, "embedding")
    _vectors_close(report, kind, name, x, y, atol, min_cos)


def _compare_vectors(report, kind, name, a, b, key, atol, min_cos) -> None:
    pair = _aligned(a, b, [key])
    if pair is None:
        report.fail(f"{kind}/{name}: {key} sets differ")
        return
    x, y = vectors_of(pair[0], "vector"), vectors_of(pair[1], "vector")
    _vectors_close(report, kind, name, x, y, atol, min_cos)


def _vectors_close(report, kind, name, x, y, atol, min_cos) -> None:
    if x.shape != y.shape:
        report.fail(f"{kind}/{name}: vector shape {x.shape} vs {y.shape}")
        return
    max_abs = float(np.abs(x - y).max()) if x.size else 0.0
    x64, y64 = x.astype(np.float64), y.astype(np.float64)  # float64: exact for equal vectors
    norms = np.linalg.norm(x64, axis=1) * np.linalg.norm(y64, axis=1)
    cos = float(((x64 * y64).sum(axis=1) / norms).min()) if x.size else 1.0
    msg = f"{kind}/{name}: {x.shape[0]} vectors, max|diff| {max_abs:.2e}, min cosine {cos:.6f}"
    if max_abs <= atol and cos >= min_cos:
        report.ok(msg)
    else:
        report.fail(msg + f" (atol {atol:g}, min_cos {min_cos})")


def _compare_scores(report, kind, name, a, b, atol) -> None:
    if a.columns != b.columns:
        report.fail(f"{kind}/{name}: columns {a.columns} vs {b.columns} in reference")
        return
    pair = _aligned(a, b, ["date"])
    if pair is None:
        report.fail(f"{kind}/{name}: statement dates differ")
        return
    a, b = pair
    narratives = [c for c in a.columns if c != "date"]
    diff = np.abs(a.select(narratives).to_numpy() - b.select(narratives).to_numpy())
    worst = float(diff.max()) if diff.size else 0.0
    msg = (
        f"{kind}/{name}: {a.height} statements x {len(narratives)} narratives, "
        f"max|diff| {worst:.2e}"
    )
    if worst <= atol:
        report.ok(msg)
    else:
        report.fail(msg + f" (atol {atol:g})")
