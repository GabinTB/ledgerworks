# ledgerworks

Run, record, verify and trace pipeline artifacts: immutable artifact directories
with `meta.json` provenance sidecars, lineage, per-file hashes and a SQLite index
that is always rebuildable from the sidecars.

## Type

Python library (pip-installable, src-layout). Core is stdlib only; optional
extras: `cli` (python-dotenv), `pandas`, `relink` (numpy, polars, pyarrow).

## Layout

```
src/ledgerworks/
  artifact.py      RunMeta / RunRecord / ModelCard / Artifact, the artifact id
  ledger.py        Ledger: run() context manager, lookups, lineage, reindex, move
  meta.py          meta.json + README.md sidecars, hashing, git commit
  verify.py        hash / partial / lineage / content checks (never mutates)
  lineage.py       require_lineage: were these inputs the ones the child was built from?
  jobs.py          Job contract + JobRunner: resume, update, pause, lock, status
  periods.py       the one calendar: D/W/M/Q/Y period keys and spans
  layout.py        partition layout of an artifact (which file holds which days)
  relink.py        re-register files under new inputs / code (needs `relink` extra)
  equivalence.py   the checks relink runs before registering anything
  plugins.py       entry-point discovery (ledgerworks.* groups + legacy aliases)
  cli.py           `ledgerworks` command line
  jobs_cli.py      `ledgerworks-jobs` command line
tests/             pytest, mirrors src (test_ledger, test_jobs, test_relink, ...)
examples/codecheck UT CodeCheck 2026 reproducibility package (not in the wheel)
```

## Commands

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
uv build
```

## Invariants (do not break)

- An artifact is never mutated or deleted by this layer: `extend` appends a
  RunRecord, `deprecate` flags, `relink` registers a new artifact.
- A run is partial until it exits cleanly; `list()` / `latest()` hide partial
  and deprecated artifacts by default.
- `meta.json` is truth; `index.db` is a cache rebuilt by `reindex()`.
- The artifact id is derived from kind, model, pipeline_version, hyperparams
  and creation minute. Hyperparams are frozen at creation.
- Core modules import nothing outside the stdlib (CI checks it). Heavy deps go
  behind lazy imports and an extra.

## Releasing

Bump `version` in `pyproject.toml` and `__version__` in `__init__.py`, commit,
tag `vX.Y.Z`, push the tag. `release.yml` runs CI, publishes to TestPyPI,
smoke-installs from there, then publishes to PyPI (trusted publishing).

## Conventions

- src-layout, tests mirror source, type hints and Google-style docstrings on
  public APIs, ruff before every commit.
- When adding a module, add its test file. When fixing a bug, add a regression test.
