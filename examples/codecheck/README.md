# CodeCheck 2026: FOMC statements scored against a monetary-policy narrative taxonomy

A self-contained reproducibility package, submitted to the University of Twente
CodeCheck / Reproducibility Hackathon (20 October 2026). It is one slice of an
ongoing PhD project on narrative dynamics in financial markets (Gabin Taibi,
University of Twente, BMS-IEBIS), rebuilt on open data and an open model so
that every step can be re-run and checked by a reviewer.

**What is reproduced:** the two computational steps at the core of the research
pipeline, (1) embedding a corpus of documents with a finance-adapted language
model and (2) scoring every document against a taxonomy of narratives, applied
to FOMC statements. The outputs are the embedding tables and the score tables
under `expected/`. There is no paper table to match: this is the "ongoing
project" route of the CodeCheck call, and the research taxonomy itself is not
published here (the file in `taxonomy/` is a public stand-in with the same
structure).

**What the reviewer checks:** that `codecheck-fomc all` rebuilds, from the files
in this directory, outputs equal to `expected/` within the stated tolerances,
and that every artifact's provenance record (`meta.json`) is intact. The
package is also the end-to-end example of
[`ledgerworks`](../../README.md), the artifact-provenance library the project
is built on.

## Quick start

Requirements: Python >= 3.11, [uv](https://docs.astral.sh/uv/), about 2.7 GB of
disk (measured on macOS, Apple silicon: 1.2 GB for `.venv`, 1.5 GB for the model
in the Hugging Face cache), a network connection the first time (to
download the model from the Hugging Face Hub). CPU is enough. On Linux, `uv.lock`
takes torch from the PyTorch CPU index (`2.14.1+cpu`), not PyPI, whose Linux
wheel pulls several GB of CUDA libraries a CPU run never loads; macOS and
Windows use PyPI's torch, which is CPU-only there.

```bash
git clone https://github.com/GabinTB/ledgerworks
cd ledgerworks/examples/codecheck
uv sync --extra model          # creates .venv from uv.lock (pinned)
uv run codecheck-fomc all      # ingest, embed, score, verify, compare
```

The corpus is 46 statements from 2021-01-27 to 2026-09-16, split into 190
paragraphs. Measured runtime on an Apple M4 Pro (CPU), model already cached:
about 15 seconds wall clock, almost all of it in the embedding step (a
395M-parameter model over 190 paragraphs). The first run also downloads the
model weights (1.6 GB). The last lines of the output are the comparison with
`expected/` and a verdict (here, on the machine that produced `expected/`):

```
fomc_narrative_scores: ...
  ok    fomc_narrative_scores/2021.parquet: 80 scores, max|diff| 0.00e+00
  ...
  ok    fomc_narrative_scores/2026.parquet: 60 scores, max|diff| 0.00e+00
  ok    fomc_narrative_scores/narratives.parquet: 10 vectors, max|diff| 0.00e+00, min cosine 1.000000
        7/7 files hash-identical to the reference

REPRODUCED
```

The score tables hold 460 rows in total (46 statements × 10 narratives).

`uv run codecheck-fomc --help` lists the individual steps. Everything lands in
`./ledger/` (deleted and rebuilt freely; it is in `.gitignore`).

To check the plumbing without the model (seconds, no download):

```bash
uv sync
uv run codecheck-fomc --backend hashing all     # NOT REPRODUCED is expected here:
                                                # expected/ was produced with the real model
uv run pytest                                   # offline tests on a fixture corpus
```

## What is in the box

| Path | What | Who produced it |
|---|---|---|
| `data/raw/html/*.htm` | The FOMC statement pages, byte for byte as served by federalreserve.gov | author, with `fetch_fomc.py` |
| `data/raw/MANIFEST.json` | URL, date, SHA-256 and retrieval time of every page | `fetch_fomc.py` |
| `taxonomy/monetary_policy.yaml` | Ten monetary-policy narratives: label, description, seed phrases | author |
| `codecheck_fomc/` | The pipeline: three `ledgerworks` jobs, the embedding backends, the comparison | author |
| `expected/` | The reference run: the three artifact directories with their `meta.json` | author, with `codecheck-fomc freeze` |
| `uv.lock` | Every Python dependency pinned, model stack included | `uv` |
| `tests/` | Offline tests of the whole pipeline on a fixture corpus (hashing backend) | author |

### Data

FOMC statements from January 2021 to the retrieval date, one HTML page per
meeting statement (other FOMC press releases, such as the August 2025 update of
the Statement on Longer-Run Goals and Monetary Policy Strategy, are excluded), downloaded from the Board's
[FOMC calendars page](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).
Content on the Board's website is in the public domain and may be copied and
distributed without permission
([disclaimer](https://www.federalreserve.gov/disclaimer.htm)); the Board of
Governors of the Federal Reserve System is the source. The committed HTML is the
immutable input: parsing it into a table is the first job of the pipeline, so
it is recorded and re-run like every other step. `fetch_fomc.py` is kept for
provenance; the reviewer does not need to run it.

### Model

[`BatuhanECB/FinModernBERT-embed-large-v1`](https://huggingface.co/BatuhanECB/FinModernBERT-embed-large-v1),
a ModernBERT-large encoder domain-adapted on financial text and contrastively
fine-tuned for embeddings (Apache-2.0, 395M parameters, 1024 dimensions). It is
pinned to Hub commit `4fd33d2335e75c79df57b345aee06d49c5f85bc8` in
`codecheck_fomc/backends.py` and loaded through `sentence-transformers`; the
exact library versions are in `uv.lock` and in every artifact's model card
(`meta.json` → `model_card.serving`). It runs on CPU in float32 with
`torch.use_deterministic_algorithms(True)`.

### Taxonomy

`taxonomy/monetary_policy.yaml` holds ten narratives. Where a label coincides
with a topic of Bybee, Kelly, Manela and Xiu (2024), *Business News and
Business Cycles*, Journal of Finance 79(5),
[doi:10.1111/jofi.13377](https://doi.org/10.1111/jofi.13377), the file says so
(`bkmx: true`); the descriptions and seed phrases are this package's own and are
the part a researcher swaps for their own taxonomy.

## The pipeline

Three jobs, each producing one `ledgerworks` artifact (an immutable directory
with a `meta.json` record), each partitioned one file per year:

| Step | Kind | Input | Output |
|---|---|---|---|
| 1 ingest | `fomc_statements` (raw layer) | `data/raw/` | one row per statement: `doc_id`, `date`, `url`, `html_sha256`, `paragraphs`, `text` |
| 2 embed | `fomc_embeddings` | step 1 | one row per paragraph: `doc_id`, `para_idx`, `text_sha256`, `embedding` (unit norm) |
| 3 score | `fomc_narrative_scores` | step 2 + taxonomy | one row per statement × narrative: `cos_doc`, `cos_para_mean`, `cos_para_max`; plus `narratives.parquet` (the narrative vectors) |

Paragraphs are the embedding unit because FOMC statements are written as short
self-contained paragraphs and the model was trained at 512 tokens. The parser
(`codecheck_fomc/corpus.py`, `PARSER_VERSION = "3"`, recorded in the
`fomc_statements` hyperparameters) keeps only what the Committee says. From the
`<p>` elements of the press release it drops:

- page furniture ("For release at", "Share", "Implementation Note", "Last Update", ...);
- the release-date line printed above the title ("January 27, 2021");
- sidebar panels (`<div class="panel ...">`, "Related Information" links);
- vote paragraphs, in both formats: "Voting for/against the monetary policy
  action ..." (with any dissents inside it), the approval line that opens
  statements from June 2026 ("The Federal Open Market Committee approved the
  following statement for release by a 12 – 0 vote:"), and standalone dissent
  paragraphs ("<Name>, who preferred ...", "<Name> dissented").
 A narrative
vector is the mean of the embeddings of its description and seed phrases, from
the same model. `cos_doc` compares the narrative with the mean of a statement's
paragraph vectors; `cos_para_mean` and `cos_para_max` aggregate paragraph-level
cosines. These are raw scores; calibration is downstream work and out of scope.

Every artifact records: its hyperparameters (which define its id), its input
artifact (lineage), the model card (id, revision, dimension, pooling, library
versions, device), the git commit of the code, start and end times, the
execution environment (Python, platform, `uv.lock` digest, versions of the
direct dependencies), and the blake2b digest of every output file. `ledgerworks verify --root ledger` re-hashes
everything and checks the lineage; `ledgerworks lineage ID` prints the graph.

## How the comparison works

`codecheck-fomc compare` joins each table of the fresh run with the reference
on its keys and checks:

| Table | Check |
|---|---|
| statements | rows identical (text and HTML digest) |
| embeddings | same paragraphs (text digest); vectors within `--atol 1e-4` element-wise and cosine >= `--min-cos 0.9999` |
| scores | within `--atol 1e-4` |

Neural-network outputs are reproducible to floating-point tolerance across
machines, not bit for bit (different BLAS kernels, instruction sets and thread
counts reorder floating-point sums), hence the tolerances. The ingest step and
the hashing test backend are exact, and `compare` reports how many files are
hash-identical to the reference. The comparison exits non-zero on any failure;
with no reference in `expected/` yet (first run) it says so and exits 0.

## Cross-machine check

`expected/` was produced on an Apple M4 Pro (macOS, arm64, CPython 3.14.5, PyPI
torch 2.14.1). The manual workflow
[`reproduce.yml`](../../.github/workflows/reproduce.yml) rebuilds everything on
GitHub-hosted runners with `uv sync --extra model` and `codecheck-fomc all`, and
uploads every `meta.json` and the comparison output. Run of 14 October 2026 at
commit `556285d`:

| | Linux (ubuntu-latest) | Windows (windows-latest) |
|---|---|---|
| Machine | x86_64, CPython 3.14.8, torch 2.14.1+cpu | AMD64, CPython 3.14.7, torch 2.14.1 |
| Verdict | REPRODUCED | REPRODUCED |
| statements | 46 identical, 6/6 files hash-identical | 46 identical, 6/6 files hash-identical |
| embeddings, max\|diff\| | 6.97e-06 | 5.95e-06 |
| embeddings, min cosine | 1.000000 | 1.000000 |
| scores, max\|diff\| | 5.01e-06 | 4.11e-06 |
| narrative vectors, max\|diff\| / min cosine | 3.73e-07 / 1.000000 | 3.58e-07 / 1.000000 |
| Wall time, `all` (model download included) | 147 s | 148 s |

The largest difference is about 14 times below the tolerance (`atol 1e-4`), and
no file of the embedding or score tables is bit-identical across machines, as
expected from different CPUs and BLAS kernels. On Windows the artifact ids of
`fomc_statements` and `fomc_narrative_scores`, and the recorded `uv.lock`
digest, differ from the reference: Git for Windows checks out `MANIFEST.json`,
the taxonomy and `uv.lock` with CRLF line endings, which changes their sha256.
The data they describe, and every compared value, are the same.

## Reproducing the reference run (author's procedure)

```bash
python fetch_fomc.py                       # network: download the statements
git add data/raw && git commit -m "data: FOMC statements to <date>"
uv sync --extra model
uv run codecheck-fomc all                  # from a clean tree, so the commit is recorded
uv run codecheck-fomc freeze               # copy the run into expected/
git add expected && git commit -m "expected: reference run"
```

A dirty working tree is refused by the jobs unless `--allow-dirty`, and
`verify` flags artifacts whose recorded commit ends in `-dirty`.

## Contact

Gabin Taibi, University of Twente (BMS-IEBIS), gabin.taibi@gmail.com.
Issues and questions: https://github.com/GabinTB/ledgerworks/issues.
