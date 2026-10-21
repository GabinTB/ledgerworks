# Reviewing this package

This page is for a reviewer: what is claimed, three ways to check it (from seconds
without the model to the full pipeline), how to go through the pipeline by hand, and
what each output means. Background: [ARTIFACTS.md](ARTIFACTS.md) (what each file
holds), [PIPELINE.md](PIPELINE.md) (what each step computes),
[LEDGERWORKS.md](LEDGERWORKS.md) (why the outputs carry a provenance record).

## What is claimed

1. From the 46 committed HTML pages in `data/raw/`, the code rebuilds the three
   tables in `expected/`: statements, paragraph embeddings, narrative scores.
2. The statements are rebuilt exactly. The embeddings and scores are rebuilt within
   floating-point tolerance on any machine: every element within `1e-4`, every
   vector with cosine at least `0.9999` to the reference. They are not bit-identical
   across machines, because a different CPU adds the model's numbers in a
   different order (see [PIPELINE.md](PIPELINE.md), step 2).
3. Every committed file is the one the author's run wrote, and each table was
   computed from the one before it: the scores from the embeddings, the embeddings
   from the statements, the statements from the committed HTML. The records in
   each `meta.json` (file digests, input ids, code commit, model revision,
   environment) let you check this.

There is no published paper table to match: this is the "ongoing project" route of
CodeCheck. The taxonomy is a public stand-in for the unpublished research one.

## Setup

Python 3.12 is pinned (`.python-version`); uv installs it if needed.

```bash
git clone https://github.com/GabinTB/ledgerworks
cd ledgerworks/examples/codecheck
uv sync --extra model
```

`uv sync --extra model` installs the pinned model stack (torch, sentence-transformers;
on Linux the CPU-only torch build). `uv sync` alone is enough for check A and the
tests. Disk: about 1.2 GB for `.venv` and 1.5 GB for the model, which is downloaded
from the Hugging Face Hub the first time it is needed.

## Three ways to check

| | What it checks | Model | Time (Apple M4 Pro) |
|---|---|---|---|
| **A** `rescore --no-model` | The committed scores follow from the committed embeddings | no | under 1 s |
| **B** `rescore` | Same, with the narrative vectors recomputed by the model on your machine | yes | about 5 s, plus the first download |
| **C** `all` | The whole pipeline from the raw HTML on your machine | yes | about 15 s; 2 to 3 min on GitHub's runners |

Each one ends with `REPRODUCED` or `NOT REPRODUCED` and exits 0 or 1.

### A. Recompute the scores, no model

```bash
uv run codecheck-fomc rescore --no-model
```

It reads `expected/` where it is. It re-hashes every committed embedding and score
file against the digest in its `meta.json`, then recomputes the 46 × 10 score table
from the committed paragraph vectors and the committed narrative vectors, with the
same function the pipeline uses, and compares:

```
fomc_embeddings: fomc_embeddings__BatuhanECB-FinModernBERT-embed-large-v1-4fd33d2335e7__v0.3.0__params030c4bc3__…
        made by commit … on macOS-27.0.1-arm64-arm-64bit, Python 3.12.13
        6/6 files match the blake2b digests in its meta.json
fomc_narrative_scores: fomc_narrative_scores__BatuhanECB-FinModernBERT-embed-large-v1-4fd33d2335e7__v0.3.0__paramsfd478237__…
        made by commit … on macOS-27.0.1-arm64-arm-64bit, Python 3.12.13
        7/7 files match the blake2b digests in its meta.json
narrative vectors: 10 x 1024 read from expected/fomc_narrative_scores/narratives.parquet (not recomputed: no model)
  ok    fomc_narrative_scores/2021.parquet: 8 statements x 10 narratives, max|diff| 0.00e+00
  ...
  ok    fomc_narrative_scores/2026.parquet: 6 statements x 10 narratives, max|diff| 0.00e+00

recomputed 2026.parquet (one row per statement):
(the table)

REPRODUCED: the committed scores follow from the committed embeddings
```

**What it tells you:** the scoring code does what [PIPELINE.md](PIPELINE.md) says,
and the committed scores were produced from the committed embeddings, not from
something else. Same machine arithmetic, so the match is exact. **What it does not
test:** the model. `--year 2024` prints another year's table.

### B. Rescore the author's embeddings with the model

```bash
uv run codecheck-fomc rescore
```

It checks the committed statements and embeddings against their `meta.json`, adds
them to your `ledger/` under their original ids, embeds the 10 narratives with the
model on your machine, and scores the author's paragraph vectors with them. The new
scores artifact is yours (your commit, your environment), and its recorded input is
the author's embeddings:

```
fomc_narrative_scores    fomc_narrative_scores__…__v0.3.0__paramsfd478237__…
        from   fomc_embeddings__…__v0.3.0__params030c4bc3__…
        wrote  6 year files (46 rows) + narratives.parquet
        in     ledger/derived/fomc_narrative_scores/…
        model  BatuhanECB/FinModernBERT-embed-large-v1 @ 4fd33d2335e7 (1024 dimensions)
        code   commit …, CPython 3.12.… on …
        (its input is the committed embeddings)

  ok    fomc_narrative_scores/2021.parquet: 8 statements x 10 narratives, max|diff| …
  ...
  ok    fomc_narrative_scores/narratives.parquet: 10 vectors, max|diff| …, min cosine 1.000000

REPRODUCED
```

Then ask ledgerworks what your scores were computed from:

```bash
uv run ledgerworks --root ledger ls
uv run ledgerworks --root ledger lineage <the scores id printed above>
uv run ledgerworks --root ledger verify
```

`lineage` lists the author's embeddings and statements ids; `verify` re-hashes
every file in your ledger, the author's included.

**What it tells you:** the model on your machine produces the same narrative
vectors, and with the author's paragraph vectors the same scores. If B passes and C
fails, the difference is in the paragraph embeddings; if B fails, look at the
narrative vectors line.

### C. The whole pipeline

```bash
uv run codecheck-fomc all
```

Runs ingest, embed, score into `ledger/`, then `verify` on the ledger, then
`compare` with `expected/`. Each step prints what it read and wrote:

```
fomc_statements          fomc_statements__v0.3.0__end2026-12-31_manifest_sha25612103e8d_n_files46_parser_version3_partition_freqY_start2021-01-01__…
        wrote  6 year files (46 rows)
        in     ledger/raw/fomc_statements/…
        code   commit …, CPython 3.12.… on …
fomc_embeddings          fomc_embeddings__BatuhanECB-FinModernBERT-embed-large-v1-4fd33d2335e7__v0.3.0__params030c4bc3__…
        from   fomc_statements__…
        wrote  6 year files (190 rows)
        ...
fomc_narrative_scores    ...
        wrote  6 year files (46 rows) + narratives.parquet
        ...
0 error(s), 0 warning(s) across 3 artifacts (19 files checked)
fomc_statements: …
        reference … (commit …)
        check: every row identical (statement text and HTML digest); exact, parsing is deterministic
  ok    fomc_statements/2021.parquet: 8 statements identical
  ...
fomc_embeddings: …
        check: same paragraphs (text digest), vectors within atol element-wise and cosine >= min_cos; ...
  ok    fomc_embeddings/2021.parquet: 40 vectors, max|diff| …, min cosine …
  ...
fomc_narrative_scores: …
  ok    fomc_narrative_scores/2021.parquet: 8 statements x 10 narratives, max|diff| …
  ...

REPRODUCED
```

`max|diff|` is the largest absolute difference between your value and the
reference over the whole file; `min cosine` the lowest cosine between your vector and
the reference vector over all rows. On the author's machine both are exact (0 and
1); on other machines expect `max|diff|` around 1e-6 to 1e-5 (below).

## By hand, one step at a time

Every command below prints what it did; `-v` before the command adds the per-year
logs (e.g. `uv run codecheck-fomc -v embed`).

**1. Look at what the author committed** (no computation):

```bash
uv run codecheck-fomc explain --reference
```

For each committed artifact: its id, what it was computed from, its hyperparameters,
the model, the commit, start and end times, the Python version, platform and library
versions, and every file with its rows, columns and description.

**2. Parse the HTML:**

```bash
uv run codecheck-fomc ingest
uv run codecheck-fomc explain fomc_statements
```

Then open the table, in `uv run python`:

```python
import polars as pl
from ledgerworks import Ledger

ledger = Ledger("ledger")
s = pl.read_parquet(ledger.latest("fomc_statements").path / "2024.parquet")
print(s.select("date", "doc_id", "n_paragraphs"))
print(s["paragraphs"][0].to_list())
```

**3. Embed:**

```bash
uv run codecheck-fomc embed
uv run codecheck-fomc explain fomc_embeddings
```

```python
e = pl.read_parquet(ledger.latest("fomc_embeddings").path / "2024.parquet")
print(e.select("date", "para_idx", "text", "embedding"))
```

**4. Score:**

```bash
uv run codecheck-fomc score
uv run codecheck-fomc explain fomc_narrative_scores
```

```python
sc = pl.read_parquet(ledger.latest("fomc_narrative_scores").path / "2024.parquet")
print(sc)
```

**5. Check the records and compare:**

```bash
uv run ledgerworks --root ledger ls
uv run ledgerworks --root ledger show <an id from ls>
uv run ledgerworks --root ledger lineage <the scores id>
uv run ledgerworks --root ledger verify
uv run codecheck-fomc compare
```

Each artifact folder also has a `README.md` rendering of its record; `ledger/` can be
deleted at any time and is rebuilt by the commands.

To see a check fail, change something and re-run. Edit one seed phrase in
`taxonomy/monetary_policy.yaml`, then:

```bash
uv run codecheck-fomc --allow-dirty score
uv run codecheck-fomc compare
```

The scores and narrative vectors no longer match (`--allow-dirty` because the
repository now has an uncommitted change, which the run records). Or overwrite a
byte of a file in `expected/fomc_embeddings/` and run `rescore --no-model`: the
digest check refuses it. Restore both with `git checkout -- taxonomy expected`.

## Cross-machine results

`expected/` was produced on an Apple M4 Pro (macOS, arm64, CPython 3.12.13, torch
2.14.1). The manual workflow
[`reproduce.yml`](../../../.github/workflows/reproduce.yml) runs check C on GitHub's
Linux and Windows runners and uploads every `meta.json` and the comparison output.
Run of 21 October 2026 at commit `8b47b0f`:

| | Linux (ubuntu-latest) | Windows (windows-latest) |
|---|---|---|
| Machine | x86_64, CPython 3.12.3, torch 2.14.1+cpu | AMD64, CPython 3.12.10, torch 2.14.1 |
| Verdict | REPRODUCED | REPRODUCED |
| statements | identical, 6/6 files hash-identical | identical, 6/6 files hash-identical |
| embeddings, max\|diff\| | 6.97e-06 | 5.65e-06 |
| embeddings, min cosine | 1.000000 | 1.000000 |
| scores, max\|diff\| | 1.43e-06 | 1.19e-06 |
| narrative vectors, max\|diff\| | 3.73e-07 | 3.28e-07 |
| Wall time, `all`, model cached | 139 s | 116 s |

The largest difference is about 14 times below the tolerance. The artifact ids (up
to the creation minute) and the recorded `uv.lock` digest are the same on all three
machines.

## If something goes wrong

| Message | Meaning, what to do |
|---|---|
| `error: refusing a non-TEMP run from a dirty git tree (…-dirty)` | You changed or added a file in the repository, so the recorded commit would not describe the code that ran. Commit the change, or pass `--allow-dirty` (the run is then recorded as `…-dirty` and `verify` warns about it). |
| `error: committed reference does not match its meta.json: …` | A file in `expected/` is not the one the author's run wrote. `git status` shows what changed; `git checkout -- expected` restores it. |
| `NOT REPRODUCED` with `FAIL` lines | The line says which file and by how much. Run A and B to locate it (scoring, narrative vectors, or paragraph embeddings), and compare the `env` blocks of `explain` and `explain --reference`. |
| Slow first run | The model (1.6 GB) is being downloaded to the Hugging Face cache (`HF_HOME`). Later runs reuse it; `HF_HUB_OFFLINE=1` then forbids network access. |
| `--threads N` | Sets torch's CPU threads; results stay within tolerance. |

Questions and issues: https://github.com/GabinTB/ledgerworks/issues.
