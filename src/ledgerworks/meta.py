"""Sidecar writers: meta.json, README.md, and file hashing.

Every artifact directory carries two sidecars:

  meta.json   machine-readable RunMeta plus file hashes.  The index is a cache
              over these; `ledgerworks reindex` rebuilds index.db from them.

  README.md   the same content rendered for a human browsing the tree in a
              file manager or on Google Drive, where no tooling is available.

Both are written atomically (tmp + replace) so a crash never leaves a sidecar
that parses but lies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

from ledgerworks.artifact import RunMeta

log = logging.getLogger(__name__)

META_FILENAME = "meta.json"
README_FILENAME = "README.md"

# blake2b over xxhash: stdlib, no new dependency, ~1GB/s which is fast enough
# for a once-per-artifact operation.  Swap here if hashing ever dominates.
HASH_ALGORITHM = "blake2b"
_HASH_DIGEST_SIZE = 32  # 256-bit digest, hex-encodes to 64 chars
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_file(path: Path) -> tuple[str, int]:
    """Return (hex digest, size in bytes) for one file, streamed in chunks."""
    digest = hashlib.blake2b(digest_size=_HASH_DIGEST_SIZE)
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


# ledgerworks.jobs control files: run state, not outputs (never hashed)
JOB_CONTROL_FILES = ("job.json", "job.lock", "job.log", "PAUSE")


def hash_directory(
    directory: Path,
    pattern: str = "*",
    exclude: frozenset[str] = frozenset({META_FILENAME, README_FILENAME, *JOB_CONTROL_FILES}),
) -> dict[str, dict[str, Any]]:
    """Hash every matching file directly inside `directory` (non-recursive).

    Sidecars and job control files (ledgerworks.jobs) are excluded by default.  Returns
    {filename: {"digest", "size_bytes", "algorithm"}}, sorted by filename.
    """
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(pattern)):
        if not path.is_file() or path.name in exclude:
            continue
        if path.name.endswith(".tmp"):
            log.warning("skipping stray tmp file while hashing: %s", path)
            continue
        digest, size = hash_file(path)
        results[path.name] = {
            "algorithm": HASH_ALGORITHM,
            "digest": digest,
            "size_bytes": size,
        }
    return results


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------


def git_commit(repo_dir: Path | None = None) -> str | None:
    """Current HEAD SHA, with '-dirty' appended if the tree has changes.

    Returns None when git is unavailable or the directory is not a repo.
    """
    cwd = str(repo_dir) if repo_dir else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = ""

    return f"{sha}-dirty" if dirty else sha


# ---------------------------------------------------------------------------
# Execution environment
# ---------------------------------------------------------------------------

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def run_environment(repo_dir: Path | None = None) -> dict[str, Any]:
    """Interpreter, platform and dependency pins of the current process.

    ``uv_lock_sha256`` is the sha256 of the nearest ``uv.lock`` found walking up
    from `repo_dir` (default cwd), or None.  ``dependencies`` maps each direct
    dependency declared in the ``pyproject.toml`` beside that lock (base and
    optional) to its installed version; dependencies that are not installed
    are left out, and so is the key when there is no lock or nothing is found.
    """
    lock = _nearest(Path(repo_dir or Path.cwd()).resolve(), "uv.lock")
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock else None,
    }
    deps = _direct_dependency_versions(lock.parent / "pyproject.toml") if lock else {}
    if deps:
        env["dependencies"] = deps
    return env


def _nearest(start: Path, name: str) -> Path | None:
    for directory in (start, *start.parents):
        if (directory / name).is_file():
            return directory / name
    return None


def _direct_dependency_versions(pyproject: Path) -> dict[str, str]:
    try:
        project = tomllib.loads(pyproject.read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    versions: dict[str, str] = {}
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement)
        if not match:
            continue
        try:
            versions[match.group(1)] = metadata.version(match.group(1))
        except metadata.PackageNotFoundError:
            continue
    return dict(sorted(versions.items()))


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------


def write_meta(
    directory: Path,
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write meta.json atomically.  Returns the written path."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "artifact_id": meta.artifact_id,
        **meta.to_dict(),
        "files": file_hashes or {},
    }
    path = directory / META_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)
    return path


def read_meta(directory: Path) -> tuple[RunMeta, dict[str, dict[str, Any]]]:
    """Read meta.json back into (RunMeta, file_hashes).

    Raises FileNotFoundError when the sidecar is absent.
    """
    path = directory / META_FILENAME
    payload = json.loads(path.read_text())
    files = payload.pop("files", {})
    payload.pop("schema_version", None)
    payload.pop("artifact_id", None)
    return RunMeta.from_dict(payload), files


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------


def _format_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PB"


def render_readme(
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render RunMeta as human-readable Markdown."""
    files = file_hashes or {}
    lines: list[str] = [f"# {meta.artifact_id}", ""]

    if meta.partial:
        lines += [
            "> **INCOMPLETE RUN.** This artifact was never finalised; the "
            "producing run crashed or was interrupted. Do not consume it "
            "downstream.",
            "",
        ]
    if meta.deprecated:
        reason = meta.deprecation_reason or "no reason recorded"
        lines += [f"> **DEPRECATED.** {reason}", ""]

    lines += [
        f"- **Kind**: `{meta.kind}`",
        f"- **Pipeline**: `{meta.pipeline}` version `{meta.pipeline_version}`",
    ]
    if meta.group:
        lines.append(f"- **Group**: `{meta.group}`")
    if meta.pipeline_repo:
        lines.append(f"- **Repo**: {meta.pipeline_repo}")
    if meta.verifier:
        lines.append(f"- **Verifier**: `{meta.verifier}`")
    lines.append(f"- **Created**: {meta.created}")
    lines.append(f"- **Executions**: {len(meta.runs)}")
    lines.append("")

    if meta.hyperparams:
        lines += ["## Hyperparameters", ""]
        for key in sorted(meta.hyperparams):
            lines.append(f"- `{key}` = `{meta.hyperparams[key]!r}`")
        lines.append("")

    if meta.runs:
        lines += ["## Execution history", ""]
        lines += [
            "| # | Started | Finished | Commit | Version | Outputs | State |",
            "| --: | --- | --- | --- | --- | --: | --- |",
        ]
        for i, r in enumerate(meta.runs, 1):
            commit = (r.pipeline_commit[:10] + "...") if r.pipeline_commit else "(none)"
            state = "partial" if r.partial else "complete"
            lines.append(
                f"| {i} | {r.run_start} | {r.run_end or '(unfinished)'} | "
                f"`{commit}` | {r.pipeline_version} | {len(r.produced)} | {state} |"
            )
        lines.append("")

    if any(r.env for r in meta.runs):
        lines += ["## Environment", ""]
        lines += [
            "| # | Python | Platform | Machine | uv.lock (sha256) | Dependencies |",
            "| --: | --- | --- | --- | --- | --- |",
        ]
        for i, r in enumerate(meta.runs, 1):
            if not r.env:
                continue
            e = r.env
            lock = (e["uv_lock_sha256"][:12] + "...") if e.get("uv_lock_sha256") else "(none)"
            deps = ", ".join(f"{k} {v}" for k, v in e.get("dependencies", {}).items())
            lines.append(
                f"| {i} | {e.get('implementation', '')} {e.get('python', '')} | "
                f"{e.get('platform', '')} | {e.get('machine', '')} | `{lock}` | {deps or '-'} |"
            )
        lines.append("")

    if meta.model_card is not None:
        card = meta.model_card
        lines += ["## Model", ""]
        lines.append(f"- **Model**: `{card.model_id}` version `{card.version}`")
        if card.architecture:
            lines.append(f"- **Architecture**: {card.architecture}")
        if card.dim is not None:
            lines.append(f"- **Dimension**: {card.dim}")
        if card.pooling:
            lines.append(f"- **Pooling**: {card.pooling}")
        if card.trained_on:
            lines.append(f"- **Trained on**: {card.trained_on}")
        lines.append(f"- **Repo**: {card.repo}")
        if card.commit:
            lines.append(f"- **Model commit**: `{card.commit}`")
        if card.weights_public:
            lines.append("- **Weights**: public, fetchable from the repo above")
        else:
            digest = card.weights_sha256 or "not recorded"
            lines.append(f"- **Weights**: private, not distributable (sha256 `{digest}`)")
        if card.notes:
            lines.append(f"- **Notes**: {card.notes}")
        if card.backend:
            lines.append(f"- **Backend**: `{card.backend}`")
        lines.append("")
        if card.serving:
            lines += [
                "### Serving",
                "",
                "```json",
                json.dumps(card.serving, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]

    if meta.sources:
        lines += ["## Inputs", ""]
        for source in meta.sources:
            lines.append(f"- `{source}`")
        lines.append("")

    if meta.relink:
        rl = meta.relink
        lines += [
            "## Relinked",
            "",
            f"Same files as `{rl.get('from')}` (hard links), registered under new "
            "inputs or code after the equivalence checks below.",
            "",
        ]
        asserted = set(rl.get("asserted") or [])
        for old, new in sorted((rl.get("replace") or {}).items()):
            mark = " (asserted: never recorded as a source)" if old in asserted else ""
            lines.append(f"- `{old}` -> `{new}`{mark}")
        checks = rl.get("checks") or []
        if checks:
            lines += ["", "| Check | Result | Details |", "| --- | --- | --- |"]
            for c in checks:
                details = str(c.get("details", "")).replace("|", "/").replace("\n", " ")
                lines.append(
                    f"| `{c.get('name')}` | {'pass' if c.get('passed') else 'FAIL'} | {details} |"
                )
        lines.append("")

    if files:
        total = sum(entry["size_bytes"] for entry in files.values())
        algorithm = next(iter(files.values()))["algorithm"]
        lines += [
            "## Files",
            "",
            f"{len(files)} files, {_format_bytes(total)} total. Digests are `{algorithm}`.",
            "",
            "| File | Size | Digest |",
            "| --- | ---: | --- |",
        ]
        for name in sorted(files):
            entry = files[name]
            short = entry["digest"][:16]
            lines.append(f"| `{name}` | {_format_bytes(entry['size_bytes'])} | `{short}...` |")
        lines.append("")

    if meta.notes:
        lines += ["## Notes", "", meta.notes, ""]

    lines += [
        "---",
        "",
        "Generated by ledgerworks. Do not edit by hand: this file is "
        "regenerated from `meta.json` and any manual change will be lost.",
        "",
    ]
    return "\n".join(lines)


def write_readme(
    directory: Path,
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write README.md atomically.  Returns the written path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / README_FILENAME
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_readme(meta, file_hashes))
    tmp.replace(path)
    return path


def write_sidecars(
    directory: Path,
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write meta.json and README.md together (meta first, it is truth)."""
    write_meta(directory, meta, file_hashes)
    write_readme(directory, meta, file_hashes)
