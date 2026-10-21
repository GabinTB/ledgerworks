# The artifacts

The pipeline turns 46 HTML pages into three artifacts, each computed from the
previous one:

```
data/raw/html/*.htm + MANIFEST.json        (committed input, 46 pages)
        │  ingest: parse each page into paragraphs
        ▼
fomc_statements          46 rows        one row per statement
        │  embed: one vector per paragraph (FinModernBERT, 1024 dimensions)
        ▼
fomc_embeddings          190 rows       one row per paragraph
        │  score: mean paragraph cosine with each of the 10 narrative vectors
        ▼
fomc_narrative_scores    46 rows        one row per statement, one column per narrative
                         + narratives.parquet (the 10 narrative vectors)
```

Each artifact is a folder. In a run it lives in `ledger/{layer}/{kind}/{id}/`; the
author's copy of each is committed in `expected/{kind}/`. A folder holds:

- `2021.parquet` ... `2026.parquet`: the data, one file per year of statement dates;
- `meta.json`: the record ledgerworks writes (described at the end of this page);
- `README.md`: the same record rendered for reading (open any of them on GitHub).

Every parquet file also says what it holds, in its own metadata:

```python
import polars as pl

pl.read_parquet_metadata("expected/fomc_embeddings/2024.parquet")["description"]
```

`codecheck-fomc explain [KIND]` prints everything below for the artifacts in your
ledger, and `codecheck-fomc explain --reference` for the committed ones.

## Raw input: `data/raw/`

| File | Content |
|---|---|
| `html/monetaryYYYYMMDDa.htm` | One FOMC statement page, byte for byte as served by federalreserve.gov (public domain). 46 pages, 2021-01-27 to 2026-09-16, one per meeting. |
| `MANIFEST.json` | For each page: URL, date, SHA-256, size, retrieval time. Also the index page used and the date range. |

Produced once by the author with `python fetch_fomc.py` (network), then committed.
A page is kept only if its date is the decision day of a meeting on the Board's
calendar, so the August 2025 notation-vote release (a revised strategy statement,
not a meeting) is out. `python fetch_fomc.py --check` re-downloads everything and
confirms the committed files still match the live site (apart from Cloudflare's
per-request email-obfuscation tokens), without writing anything.

## `fomc_statements` (raw layer)

**Computed from:** `data/raw/`. The ingest step checks every page's SHA-256 against
`MANIFEST.json` first and refuses a mismatch.

**One row per statement.**

| Column | Type | Meaning |
|---|---|---|
| `doc_id` | String | Page name without extension, e.g. `monetary20260128a` |
| `date` | Date | Statement date (the meeting's decision day) |
| `url` | String | Where the page was downloaded from |
| `html_sha256` | String | SHA-256 of the committed HTML file |
| `n_paragraphs` | Int64 | Number of statement paragraphs kept by the parser |
| `paragraphs` | List(String) | The paragraphs, in page order |
| `text` | String | The paragraphs joined by blank lines |

**Hyperparameters (part of the id):** `manifest_sha256` (digest of `MANIFEST.json`:
any change to the input changes the id), `n_files`, `parser_version` (`"3"`; see
[PIPELINE.md](PIPELINE.md) for the parsing rules), and the year layout (`start`,
`end`, `partition_freq`).

**Content check (`verify`):** no statement without paragraphs.

## `fomc_embeddings`

**Computed from:** one `fomc_statements` artifact (its id is in `sources`).

**One row per statement paragraph.** The vector is spread over 1024 plain columns so
any tool shows it without unpacking.

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | Statement date |
| `doc_id` | String | The statement |
| `para_idx` | Int64 | Position of the paragraph in the statement, from 0 |
| `text` | String | The paragraph that was embedded |
| `text_sha256` | String | SHA-256 of `text` (UTF-8) |
| `e0000` ... `e1023` | Float32 | The embedding; the row vector has unit length |

It stays one row per paragraph (not per date) because the score is a mean of
paragraph-level cosines, which cannot be recovered from an averaged vector.

**Hyperparameters:** `model_id`, `revision` (the exact Hugging Face commit),
`max_seq_length` (512), `backend`, `unit` (`paragraph`), year layout.

**Model card** (also in `meta.json`): model id and revision, architecture, dimension
(1024), pooling, licence note, and how it ran (`serving`: CPU, float32, number of
threads, torch / transformers / sentence-transformers versions).

**Content check:** every row has as many `e` columns as the model's dimension, and
unit norm (within 1e-3).

## `fomc_narrative_scores`

**Computed from:** one `fomc_embeddings` artifact, plus the taxonomy file
`taxonomy/monetary_policy.yaml` (its SHA-256 is a hyperparameter).

**One row per statement, one column per narrative.**

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | Statement date |
| `doc_id` | String | The statement |
| `federal_reserve`, `inflation`, `rate_hike`, `rate_cut`, `balance_sheet`, `labor_market`, `economic_growth`, `recession`, `financial_conditions`, `uncertainty` | Float32 | Score of the statement on that narrative |

The score is the mean, over the statement's paragraphs, of the cosine between the
paragraph's embedding and the narrative's vector (formula in
[PIPELINE.md](PIPELINE.md)). It lies in [-1, 1]; in the committed run every value lies
between 0.48 and 0.70. These are raw similarities: comparing narratives
with each other needs a calibration that is out of scope here; comparing one
narrative across dates is what the column is for.

The same folder holds **`narratives.parquet`**, one row per narrative:

| Column | Type | Meaning |
|---|---|---|
| `narrative_id` | String | Column name used in the score tables |
| `label` | String | Human label from the taxonomy |
| `e0000` ... `e1023` | Float32 | The narrative vector: unit-length mean of the embeddings of the narrative's description and seed phrases |

**Hyperparameters:** `taxonomy`, `taxonomy_version`, `taxonomy_sha256`,
`narrative_vector` (how the vector is built), year layout.

**Content check:** `narratives.parquet` present; every year file has exactly the
columns `date`, `doc_id` and the narratives, with values in [-1, 1].

## `expected/RUN.json`

The ids of the three committed artifacts, written by `codecheck-fomc freeze`.

## The record: `meta.json`

Every artifact folder has one. Abridged from the committed embeddings:

```jsonc
{
  "artifact_id": "fomc_embeddings__BatuhanECB-FinModernBERT-embed-large-v1-4fd33d2335e7__v0.2.0__params030c4bc3__20261021T...",
  "kind": "fomc_embeddings",
  "pipeline": "ledgerworks-codecheck-fomc",
  "pipeline_version": "v0.2.0",
  "hyperparams": {                       // these define the id
    "backend": "finmodernbert",
    "model_id": "BatuhanECB/FinModernBERT-embed-large-v1",
    "revision": "4fd33d2335e75c79df57b345aee06d49c5f85bc8",
    "max_seq_length": 512,
    "unit": "paragraph",
    "partition_freq": "Y", "start": "2021-01-01", "end": "2026-12-31"
  },
  "sources": [                           // lineage: the input artifact(s)
    "fomc_statements__v0.2.0__end2026-12-31_manifest_sha25612103e8d_n_files46_parser_version3_partition_freqY_start2021-01-01__20261021T..."
  ],
  "model_card": {
    "model_id": "BatuhanECB/FinModernBERT-embed-large-v1",
    "commit": "4fd33d2335e75c79df57b345aee06d49c5f85bc8",
    "dim": 1024, "pooling": "mean", "backend": "sentence-transformers",
    "serving": {"device": "cpu", "dtype": "float32", "threads": 8, "torch": "2.14.1", ...}
  },
  "runs": [{                             // one entry per execution
    "run_start": "...", "run_end": "...", "partial": false,
    "pipeline_commit": "<git commit of the code that ran>",
    "produced": ["2021.parquet", ...],
    "env": {
      "python": "3.12.13", "implementation": "CPython",
      "platform": "macOS-27.0.1-arm64-arm-64bit", "machine": "arm64",
      "uv_lock_sha256": "<sha256 of examples/codecheck/uv.lock>",
      "dependencies": {"numpy": "2.5.4", "polars": "2.0.0", "torch": "2.14.1", ...}
    }
  }],
  "files": {                             // what verify re-hashes
    "2021.parquet": {"algorithm": "blake2b", "digest": "d13431f4...", "size_bytes": 396441},
    ...
  }
}
```

**How the id is built:** `kind`, then the model slug (`model id` + first 12
characters of the revision) when there is a model, then the pipeline version, then
the hyperparameters (inline when short, otherwise `params` + an 8-character digest of
all of them), then the creation minute (UTC). The same settings on another machine
give the same id up to the minute, which is how the cross-machine check knows it is
comparing like with like.

**Files not hashed:** `meta.json` and `README.md` themselves; `job.json` / `job.log`
(run state, not outputs) are left out of `expected/`.
