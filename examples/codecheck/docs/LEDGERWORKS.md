# Why ledgerworks, and what it adds to this package

This package is built on [`ledgerworks`](../../../README.md), a small Python library
written for the PhD project this example comes from. This page explains the problem
it solves, how it works, and what it lets a reviewer check here that a script and a
folder of expected outputs could not.

## The problem

A text-to-signal research pipeline runs in stages: collect documents, parse them,
embed them with a language model, score them against a set of narratives, aggregate,
forecast. Each stage takes hours to weeks of compute and is re-run many times while
the research moves: a new parser rule, a new model revision, a different taxonomy,
more years of data. After a few months the questions that matter for a paper or a
review become hard to answer:

- **Which inputs produced this table?** A score file sits next to three embedding
  files of different ages. Which one was it computed from?
- **With which code, model and environment?** The repository moved on since; the
  model on the Hub has a new revision; a library was upgraded.
- **Is this output complete?** A run crashed at year 2019 of 2026. The half-written
  folder looks like any other, and the next stage reads it.
- **Was it overwritten?** The same path was re-run with different settings. The
  numbers in last month's draft no longer match the file.

Writing these facts in a notebook or a file name does not survive contact with a
long project. They have to be recorded by the code that runs the stage, at the moment
it runs, next to the output.

## What ledgerworks does

| Idea | What it means in practice |
|---|---|
| **An artifact is a folder** | Every run of a stage writes one folder: its output files plus a `meta.json` record and a human-readable `README.md`. |
| **`meta.json` is the truth** | The record holds the stage's hyperparameters, the ids of its input artifacts, the model card, the git commit, start and end times, the environment (Python, platform, `uv.lock` digest, library versions) and a blake2b digest of every output file. The SQLite index (`index.db`) is only a cache, rebuilt from the `meta.json` files at any time (`ledgerworks reindex`). |
| **The id says what it is** | An artifact id is built from its kind, model, pipeline version, hyperparameters and creation minute, e.g. `fomc_embeddings__BatuhanECB-FinModernBERT-embed-large-v1-4fd33d2335e7__v0.3.0__params030c4bc3__20261021T0915`. Changing a hyperparameter gives a new id, never a silent overwrite. |
| **Artifacts are never changed** | ledgerworks never mutates or deletes an artifact. Re-running makes a new one; an old one can be flagged `deprecated`, not edited. |
| **Lineage** | Each artifact lists the ids of its inputs (`sources`). `ledgerworks lineage ID` walks the graph up to the raw data. |
| **Partial runs are visible** | A run is marked partial from its first second until it exits cleanly. A crash leaves a folder that says `INCOMPLETE RUN`, and lookups (`latest`, `ls`) skip it by default, so the next stage cannot consume it by accident. |
| **Verification** | `ledgerworks verify` re-hashes every file against its `meta.json`, checks the lineage resolves, flags partial runs and code run from a dirty git tree, and runs per-kind content checks (here: vectors have the model's dimension and unit norm, scores are within [-1, 1]). It reports, it never repairs. |

The core library uses only the Python standard library; heavy dependencies (pandas,
polars, numpy) are optional. Its pipeline jobs (`ledgerworks.jobs`) add resumable
units (one file per year here), locking, and pause/resume on top of the same record.

## What it adds in this package

Without ledgerworks, a reproducibility package is a script and a folder of expected
outputs: run the script, diff the outputs. That checks one thing (same numbers) and
leaves the reviewer to trust everything else. Here every output in `expected/` is an
artifact folder with its `meta.json`, so the reviewer can check more, and check it
piece by piece:

1. **That the committed files are the ones the author's run wrote.** Every file's
   blake2b digest is in its `meta.json`; `codecheck-fomc rescore` and
   `ledgerworks verify` re-hash them.
2. **What each output was computed from.** The scores record the id of the
   embeddings they were computed from, which record the id of the statements table,
   which records the digest of `data/raw/MANIFEST.json`, which lists the SHA-256 of
   every HTML page. `codecheck-fomc explain --reference` prints the chain.
3. **With what.** Each record names the model and its exact Hub revision, the git
   commit of the code, the Python version, the platform and every direct dependency
   version, so a mismatch has somewhere to be traced to.
4. **Each stage separately.** Because the author's embeddings are a self-describing
   artifact, the reviewer can rescore *them* (check A without the model, check B
   with it) instead of only re-running everything (check C). A disagreement can then
   be located: in the scoring code, in the narrative vectors, or in the embeddings.
   After check B, `ledgerworks lineage` on the reviewer's new scores shows the
   author's embeddings id as their input: the reviewer's result is provably built on
   the author's files, byte for byte.

See [REVIEWING.md](REVIEWING.md) for the commands and [ARTIFACTS.md](ARTIFACTS.md)
for what each artifact contains.

## What it does not do

- It does not make computations deterministic. Neural-network outputs still differ
  across CPUs in the last floating-point digits; ledgerworks records enough to explain
  a difference, and `compare` judges it against stated tolerances.
- It does not store or version large data itself: artifacts are ordinary folders on
  disk (or a shared drive), and the raw data here is committed to git.
- It does not sandbox anything or capture the full operating system. The environment
  record covers Python, platform, the lock file and direct dependencies.
- A `meta.json` is evidence, not a signature: someone with write access could edit
  both a file and its digest. The git history of `expected/` is what anchors the
  committed reference.
