"""Plain-text accounts of artifacts: what a step read, what it wrote, where, with what.

``summary`` is the few lines every step prints; ``explain`` is the full account
behind ``codecheck-fomc explain`` (inputs, hyperparams, model, every run with its
commit and environment, every file with its rows, columns and description).
Everything comes from the artifact's ``meta.json`` and its parquet files.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from ledgerworks import Artifact

from codecheck_fomc.reference import short_commit


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _columns(schema) -> str:
    return ", ".join(f"{name} ({dtype})" for name, dtype in schema.items())


def _rows(path: Path) -> int:
    return pl.scan_parquet(path).select(pl.len()).collect().item()


def _where(env: dict) -> str:
    if not env:
        return "environment not recorded"
    python = f"{env.get('implementation', 'Python')} {env.get('python', '?')}"
    return f"{python} on {env.get('platform', '?')}"


def summary(art: Artifact) -> str:
    """What a step wrote: id, inputs, files and rows, model, commit, machine."""
    meta = art.meta
    files = art.files("*.parquet")
    years = [p for p in files if p.name[:4].isdigit()]
    extra = "".join(f" + {p.name}" for p in files if p not in years)
    rows = sum(_rows(p) for p in years)
    run = meta.runs[-1]
    lines = [f"{meta.kind:24s} {art.artifact_id}"]
    for source in meta.sources:
        lines.append(f"        from   {source}")
    lines.append(f"        wrote  {len(years)} year files ({rows} rows){extra}")
    lines.append(f"        in     {_rel(art.path)}")
    if meta.model_card is not None:
        card = meta.model_card
        lines.append(f"        model  {card.model_id} @ {card.version} ({card.dim} dimensions)")
    lines.append(f"        code   commit {short_commit(run.pipeline_commit)}, {_where(run.env)}")
    return "\n".join(lines)


def explain(art: Artifact, label: str = "") -> str:
    """Everything recorded about one artifact, in reading order."""
    meta = art.meta
    out = [f"== {meta.kind}{label}", f"id        {art.artifact_id}", f"folder    {_rel(art.path)}"]
    out.append(f"pipeline  {meta.pipeline} {meta.pipeline_version}")
    if meta.sources:
        out.append("computed from (inputs, by artifact id):")
        out += [f"    {s}" for s in meta.sources]
    else:
        out.append("computed from: data/raw/ (no input artifact: this is the raw layer)")
    out.append("hyperparameters (they define the id):")
    out += [f"    {k} = {v!r}" for k, v in meta.hyperparams.items()]
    if meta.model_card is not None:
        card = meta.model_card
        out.append(f"model     {card.model_id}, revision {card.commit or card.version}")
        out.append(f"          {card.dim} dimensions, {card.pooling} pooling, {card.backend}")
        serving = card.serving or {}
        if serving:
            out.append(
                "          ran on "
                + ", ".join(
                    f"{k} {serving[k]}" for k in ("device", "dtype", "threads") if k in serving
                )
            )
    for i, run in enumerate(meta.runs, 1):
        state = "partial" if run.partial else "complete"
        out.append(f"run {i}     {run.run_start} -> {run.run_end or '(unfinished)'} ({state})")
        out.append(f"          commit {run.pipeline_commit or '(none)'}")
        out.append(f"          {_where(run.env)}")
        if run.env.get("uv_lock_sha256"):
            out.append(f"          uv.lock sha256 {run.env['uv_lock_sha256']}")
        deps = run.env.get("dependencies", {})
        if deps:
            out.append("          " + ", ".join(f"{k} {v}" for k, v in deps.items()))
    out.append("files (blake2b digests in meta.json):")
    shown: set[tuple[str, str]] = set()
    for path in art.files("*.parquet"):
        digest = art.file_hashes.get(path.name, "")
        out.append(f"    {path.name}: {_rows(path)} rows, blake2b {digest[:16]}...")
        layout = (
            _columns(pl.read_parquet_schema(path)),
            pl.read_parquet_metadata(path).get("description", ""),
        )
        if layout not in shown:  # say once what the files of this shape hold
            shown.add(layout)
            out.append(f"        columns: {layout[0]}")
            if layout[1]:
                out.append(f"        {layout[1]}")
    return "\n".join(out)
