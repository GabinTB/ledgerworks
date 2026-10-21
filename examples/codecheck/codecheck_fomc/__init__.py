"""CodeCheck 2026 package: FOMC statements -> embeddings -> narrative scores, with ledgerworks.

Three jobs, each one artifact kind, each a one-file-per-year layout:

    fomc_statements          raw layer: parsed from the committed HTML (corpus.py)
    fomc_embeddings          paragraph embeddings of every statement (embed.py)
    fomc_narrative_scores    cosine scores against the taxonomy narratives (score.py)

``compare.py`` checks a fresh run against the committed ``expected/`` outputs with
explicit tolerances, and ``cli.py`` ties it together.
"""

from __future__ import annotations

from pathlib import Path

PIPELINE = "ledgerworks-codecheck-fomc"
PIPELINE_REPO = "https://github.com/GabinTB/ledgerworks"
PIPELINE_VERSION = "v0.2.0"

HERE = Path(__file__).resolve().parent.parent  # examples/codecheck
DATA_RAW = HERE / "data" / "raw"
TAXONOMY_DIR = HERE / "taxonomy"
EXPECTED_DIR = HERE / "expected"
DEFAULT_LEDGER = HERE / "ledger"
