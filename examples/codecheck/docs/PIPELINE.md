# The method, step by step

What each of the three steps computes, with the code that does it. The artifacts they
write are described in [ARTIFACTS.md](ARTIFACTS.md).

## 1. ingest: from HTML pages to statements (`codecheck_fomc/corpus.py`)

For each page listed in `data/raw/MANIFEST.json`, the step checks the file's SHA-256
against the manifest, then keeps the text of the `<p>` elements inside the page's
`<div id="article">`. Everything that is not something the Committee says is dropped
(`parse_statement`, `PARSER_VERSION = "3"`):

| Dropped | Example | How it is recognised |
|---|---|---|
| Page furniture | "For release at 2:00 p.m. EDT", "Share", "Implementation Note issued …", "Last Update: …" | paragraph starts with one of a fixed list of phrases |
| Release-date line | "January 27, 2021" | the paragraph is only a date |
| Sidebar | "Plans for Reducing the Size of the Federal Reserve's Balance Sheet" (a link under "Related Information") | anything inside a `<div class="panel …">` |
| Vote paragraph, older format | "Voting for the monetary policy action were Jerome H. Powell, Chair; … Voting against this action was …, who preferred …" | starts with "Voting for/against this/the (monetary policy) action" |
| Approval line, format from June 2026 | "The Federal Open Market Committee approved the following statement for release by a 12 – 0 vote:" | that sentence, with any vote count |
| Standalone dissent | "<Name>, who preferred …", "<Name> dissented …" | a name followed by "who preferred/supported" or "dissented" |

What remains is the statement itself: 3 to 5 paragraphs per statement, 190 in total.
The rules match page structure and phrasing, never a specific date; the tests in
`tests/test_pipeline.py` cover each rule on synthetic pages of both formats.

Paragraphs are the unit of everything downstream because FOMC statements are written
as short, self-contained paragraphs (one on the economy, one on inflation, one on the
decision, …) and the model reads at most 512 tokens: the longest paragraph here is
355 tokens, the median 81, so nothing is truncated.

## 2. embed: one vector per paragraph (`codecheck_fomc/embed.py`, `backends.py`)

**Model:** [`BatuhanECB/FinModernBERT-embed-large-v1`](https://huggingface.co/BatuhanECB/FinModernBERT-embed-large-v1),
a ModernBERT-large encoder (395M parameters) adapted to financial text and
fine-tuned for sentence embeddings. Apache-2.0, 1024 dimensions, mean pooling.

**Pinned:** the code loads Hub revision `4fd33d2335e75c79df57b345aee06d49c5f85bc8`
(`FINMODERNBERT_REVISION` in `backends.py`), never "latest"; the revision is a
hyperparameter, so another revision would produce a different artifact id.
The library versions are pinned by `uv.lock` and recorded in each artifact.

**Run:** through `sentence-transformers`, on CPU, in float32, 512-token limit, batches
of 16, with `torch.use_deterministic_algorithms(True)` and a fixed seed. Each
vector is then scaled to unit length.

**Why not bit-identical across machines:** embedding 190 paragraphs with this model
takes about 10¹³ floating-point operations; different CPUs, instruction sets, BLAS libraries and
thread counts add them in different orders, which changes the last digits. On the
same machine and versions the output is bit-identical; across machines it agrees to
about 1e-5 (see the cross-machine numbers in [REVIEWING.md](REVIEWING.md)). That is
why the comparison uses a tolerance for this step and the next, and exact equality
for parsing.

A second backend, `hashing`, maps text to vectors with a fixed hash of its words and
word pairs (no model, exact everywhere). It exists for the offline tests and for
`--backend hashing`, a quick plumbing check; its outputs are not comparable with the
reference.

## 3. score: statements against narratives (`codecheck_fomc/score.py`)

**The taxonomy** (`taxonomy/monetary_policy.yaml`) lists 10 narratives. Each has an
id, a label, a one-paragraph description and a few seed phrases, for example:

```yaml
- id: inflation
  label: Inflation
  description: >
    The level and direction of inflation relative to the 2 percent objective: ...
  seeds:
    - Inflation remains elevated.
    - Longer-term inflation expectations appear to remain well anchored.
```

This file is the part a researcher replaces with their own taxonomy; the research
taxonomy of the PhD project is not published, and this one is a public stand-in with
the same structure.

**Narrative vectors** (`narrative_vectors`): the description and every seed phrase
are embedded with the same model and revision as the paragraphs, averaged, and scaled
to unit length. One vector per narrative, saved in `narratives.parquet`.

**Scores** (`score_frame`): for statement *s* with paragraph vectors
*p₁ … pₖ* and narrative vector *n*,

```
score(s, n) = (1/k) · Σᵢ cos(pᵢ, n) = (1/k) · Σᵢ pᵢ · n        (all vectors have unit length)
```

in code: `sims = pv @ nv.T` (every paragraph against every narrative), then the mean
of each column over the statement's paragraphs. The result has one row per statement
and one column per narrative.

**Reading the numbers:** a higher score means the statement's paragraphs are, on
average, closer in the model's embedding space to the narrative's description. The
values are raw: every narrative has its own baseline (around 0.66 for
`federal_reserve`, 0.56 for `financial_conditions` here), so they are meant for
following one narrative across dates, not for ranking narratives within a statement.
Calibration and the use of these series in forecasting are later stages of the
research and are not part of this package.

## What runs where

| Step | Command | Input | Needs the model | Time (Apple M4 Pro, model cached) |
|---|---|---|---|---|
| ingest | `codecheck-fomc ingest` | `data/raw/` | no | under 1 s |
| embed | `codecheck-fomc embed` | latest `fomc_statements` | yes | about 8 s, plus loading the model |
| score | `codecheck-fomc score` | latest `fomc_embeddings` + taxonomy | yes (10 narratives only) | about 1 s, plus loading the model |
| all three + verify + compare | `codecheck-fomc all` | `data/raw/` | yes | about 13 s |
