# ledgerworks

Run, record, verify and trace pipeline artifacts.

Every output of a pipeline becomes an immutable artifact directory with a
`meta.json` sidecar that says what it is, what it was built from, which code
and model produced it, and what every file hashed to when it was written.
A SQLite index makes lookups fast and can always be rebuilt from the sidecars,
so the files are the source of truth and the index is a cache.

| | |
|---|---|
| PyPI | `pip install ledgerworks` |
| Python | >= 3.11 |
| Core dependencies | none (stdlib only) |
| License | MIT |

## What it does

| Layer | What you get |
|---|---|
| **Record** | A deterministic artifact id (`{kind}__{model}__{version}__{params}__{minute}`), a `meta.json` and `README.md` sidecar per artifact, one `RunRecord` per execution (start, end, git commit, files produced, Python, platform, `uv.lock` digest and direct dependency versions), blake2b hash and size of every output file |
| **Query** | `latest(kind)`, `get(id)`, `list(...)`, `parents` / `children` / `ancestors` / `descendants`, DuckDB views by kind |
| **Verify** | Re-hash files, flag crashed (partial) runs and stray temp files, find lineage edges that point to nothing, run per-kind content checks you register |
| **Run** | `ledgerworks.jobs`: long pipelines as ordered work units with resume, update, pause, a heartbeat lock and a status file |
| **Migrate** | `ledgerworks.relink`: re-register an artifact's files under new inputs or new code without recomputing, after equivalence checks (needs the `relink` extra) |

Guarantees that hold everywhere:

- a run is **partial** until it exits cleanly, and `latest()` never returns a partial artifact;
- nothing is mutated or deleted: new data appends an execution, a wrong artifact is deprecated;
- the index holds nothing the sidecars lack, so `reindex` recovers it from disk at any time.

## Installation

```bash
pip install ledgerworks              # core, stdlib only
pip install "ledgerworks[cli]"       # + .env support for the command lines
pip install "ledgerworks[relink]"    # + numpy / polars / pyarrow for relink checks
pip install "ledgerworks[all]"
```

For development:

```bash
git clone https://github.com/GabinTB/ledgerworks
cd ledgerworks
uv sync --group dev
uv run pytest
```

## Quick start

```python
from ledgerworks import Ledger, ModelCard

ledger = Ledger("/data/ledger")

# 1. register a raw input
with ledger.run(kind="documents", pipeline="my-project", pipeline_version="v0.1.0",
                hyperparams={"source": "fomc", "start": "2021-01-01"},
                layer="raw") as run:
    write_parquet(run.out_dir / "documents.parquet", docs)

# 2. derive something from it, with lineage and a model card
source = ledger.latest("documents")
card = ModelCard(model_id="my-embedder", version="1.0", repo="https://hf.co/...",
                 weights_public=True, dim=768)

with ledger.run(kind="document_embeddings", pipeline="my-project", pipeline_version="v0.1.0",
                hyperparams={"pooling": "mean"}, sources=[source], model_card=card,
                verifier="document_embeddings") as run:
    embed(source.path, run.out_dir)

# 3. query
art = ledger.latest("document_embeddings")
print(art.artifact_id, art.meta.pipeline_commit, art.file_hashes)
print(ledger.ancestors(art.artifact_id))
```

On clean exit the run's record is marked complete, the outputs are hashed and
the sidecars written. On an exception the record stays partial, so the crash is
visible instead of silent.

The artifact directory:

```
{root}/derived/document_embeddings/document_embeddings__my-embedder-1.0__v0.1.0__poolingmean__20261014T1530/
    meta.json        # the record: identity, hyperparams, sources, model card, runs[], file hashes
    README.md        # the same, rendered for humans
    2021-01.parquet
    ...
```

## Command lines

```
ledgerworks init | ls | show ID | verify | lineage ID | deprecate ID --reason ... | reindex | move | relink
ledgerworks-jobs start KIND | resume ID | update ID | pause ID | status | list | check ID
```

The root comes from `--root`, else `$LEDGER_ROOT` (also read from `.env` when
the `cli` extra is installed).

## Plugging in your own kinds

A package contributes jobs, content verifiers and equivalence checks through
entry points in its `pyproject.toml`:

```toml
[project.entry-points."ledgerworks.jobs"]
document_embeddings = "my_project.embed:EmbedJob"

[project.entry-points."ledgerworks.verifiers"]
document_embeddings = "my_project.embed:verify_artifact"
```

A `Job` declares what it computes (`kind`, `pipeline`, `pipeline_version`,
`params()`, `sources()`, `units()`, `run_unit()`); `JobRunner` owns how it
runs. See `ledgerworks/jobs.py` for the contract and `examples/codecheck` for
a complete job.

## Examples

- [`examples/codecheck`](examples/codecheck): a self-contained reproducibility
  package (FOMC statements, an open embedding model, a macro-narrative
  taxonomy) submitted to the University of Twente CodeCheck 2026. Run it with
  one command and compare the outputs to the committed reference.

## Development

```bash
uv run pytest             # tests
uv run ruff check .       # lint
uv run ruff format .      # format
uv build                  # wheel + sdist
```

Releases: push a tag `vX.Y.Z` matching the version in `pyproject.toml`. The
release workflow runs CI, publishes to TestPyPI, installs from there as a
smoke test, then publishes to PyPI and creates a GitHub release.

## License

MIT
