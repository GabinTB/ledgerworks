# CodeCheck 2026: FOMC statements scored against a monetary-policy narrative taxonomy

A self-contained reproducibility package, submitted to the University of Twente
CodeCheck / Reproducibility Hackathon (20 October 2026). It is one slice of an
ongoing PhD project on narrative dynamics in financial markets (Gabin Taibi,
University of Twente, BMS-IEBIS), rebuilt on open data and an open model so that
every step can be re-run and checked.

**What it computes.** 46 FOMC statements (2021-01-27 to 2026-09-16) are split into
paragraphs, each paragraph is embedded with a finance-adapted language model, and
each statement gets a score on each of 10 monetary-policy narratives: the mean
cosine similarity between its paragraphs and the narrative. Output: a table with one
row per statement date and one column per narrative.

**What is claimed.** From the committed HTML pages, the code rebuilds the committed
tables in `expected/`: the statements exactly, the embeddings and scores within
floating-point tolerance on any machine. Every committed file carries a provenance
record (`meta.json`, written by [`ledgerworks`](../../README.md)) that says what it
was computed from, with which code, model revision and environment, and its digest.

## Start here

```bash
git clone https://github.com/GabinTB/ledgerworks
cd ledgerworks/examples/codecheck
uv sync --extra model
```

Then pick a check. Each one prints what it reads and writes and ends with
`REPRODUCED` or `NOT REPRODUCED`.

| Check | Command | Model | Time |
|---|---|---|---|
| **A.** The committed scores follow from the committed embeddings | `uv run codecheck-fomc rescore --no-model` | no | under 1 s |
| **B.** Rescore the committed embeddings with the model on your machine | `uv run codecheck-fomc rescore` | yes | ~5 s |
| **C.** The whole pipeline from the raw HTML | `uv run codecheck-fomc all` | yes | ~15 s (M4 Pro), 2-3 min (CI runners) |

`uv run codecheck-fomc explain --reference` describes every committed artifact: what
it was computed from, the model, the commit, the environment, the files and their
columns. Times exclude the first download of the model (1.6 GB).

## Documentation

| Page | For |
|---|---|
| [docs/REVIEWING.md](docs/REVIEWING.md) | The reviewer's guide: the claim, checks A/B/C with their output, a step-by-step session by hand, cross-machine results, troubleshooting |
| [docs/ARTIFACTS.md](docs/ARTIFACTS.md) | Every artifact: what it is computed from, its files, every column, its hyperparameters, its `meta.json` |
| [docs/PIPELINE.md](docs/PIPELINE.md) | The method: parsing rules, the model and its pinning, narrative vectors, the score formula |
| [docs/LEDGERWORKS.md](docs/LEDGERWORKS.md) | Why the outputs carry a provenance record, and what that lets you check here |

## What is in the box

| Path | What | Produced by |
|---|---|---|
| `data/raw/html/*.htm` | The 46 FOMC statement pages, byte for byte as served by federalreserve.gov | author, `fetch_fomc.py` |
| `data/raw/MANIFEST.json` | URL, date, SHA-256 and retrieval time of every page | `fetch_fomc.py` |
| `taxonomy/monetary_policy.yaml` | The 10 narratives: label, description, seed phrases | author |
| `codecheck_fomc/` | The pipeline (three `ledgerworks` jobs), the comparison, `rescore`, `explain` | author |
| `expected/` | The reference run: the three artifact folders with their `meta.json`, and `RUN.json` | author, `codecheck-fomc freeze` |
| `uv.lock`, `.python-version` | Every Python dependency pinned, model stack included; Python 3.12 | `uv` |
| `tests/` | Offline tests of every step on a small synthetic corpus | author |
| `docs/` | The pages above | author |

`ledger/` is created by the commands (your runs) and can be deleted at any time.

## Data, model and licences

- **Data:** FOMC statements, Board of Governors of the Federal Reserve System, from
  the [FOMC calendars page](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm);
  public domain ([disclaimer](https://www.federalreserve.gov/disclaimer.htm)). Only
  statements issued on a meeting's decision day are included.
- **Model:** [`BatuhanECB/FinModernBERT-embed-large-v1`](https://huggingface.co/BatuhanECB/FinModernBERT-embed-large-v1),
  Apache-2.0, pinned to Hub revision `4fd33d2335e75c79df57b345aee06d49c5f85bc8`.
- **Taxonomy:** this package's own wording, a public stand-in for the unpublished
  research taxonomy. Labels marked `bkmx: true` coincide with topics of Bybee, Kelly,
  Manela and Xiu (2024), *Business News and Business Cycles*, Journal of Finance
  79(5), [doi:10.1111/jofi.13377](https://doi.org/10.1111/jofi.13377).

## Rebuilding the reference (author's procedure)

```bash
python fetch_fomc.py
git add data/raw
git commit -m "data: FOMC statements"
uv sync --extra model
uv run codecheck-fomc all
uv run codecheck-fomc freeze
git add expected
git commit -m "expected: reference run"
```

`fetch_fomc.py` needs the network; `python fetch_fomc.py --check` confirms the
committed pages still match the live site without writing anything. The run must
start from a clean git tree, so the recorded commit describes the code exactly.

## Contact

Gabin Taibi, University of Twente (BMS-IEBIS), gabin.taibi@utwente.nl.
Issues and questions: https://github.com/GabinTB/ledgerworks/issues.
