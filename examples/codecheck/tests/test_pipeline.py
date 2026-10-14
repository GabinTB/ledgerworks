"""The whole pipeline, offline, on the fake corpus with the hashing backend."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from ledgerworks import Ledger, verify
from ledgerworks.jobs import JobRunner

from codecheck_fomc.backends import HashingBackend
from codecheck_fomc.compare import compare, freeze
from codecheck_fomc.corpus import IngestJob, Manifest, parse_statement
from codecheck_fomc.embed import EmbedJob
from codecheck_fomc.score import ScoreJob, Taxonomy
from tests.conftest import STATEMENTS

TAXONOMY = Path(__file__).resolve().parent.parent / "taxonomy" / "monetary_policy.yaml"


def _runner(ledger: Ledger) -> JobRunner:
    return JobRunner(ledger, allow_dirty=True, log_file=False, handle_signals=False)


def run_all(ledger: Ledger, raw_dir: Path, backend=None):
    backend = backend or HashingBackend(dim=64)
    runner = _runner(ledger)
    statements = runner.start(IngestJob(raw_dir))
    embeddings = runner.start(EmbedJob(statements, backend))
    scores = runner.start(ScoreJob(embeddings, Taxonomy.load(TAXONOMY), backend))
    return statements, embeddings, scores


class TestParsing:
    def test_statement_paragraphs_without_furniture(self, raw_dir):
        html = (raw_dir / "html" / "monetary20230322a.htm").read_text()
        paragraphs = parse_statement(html)
        assert paragraphs == STATEMENTS["20230322"]

    def test_page_without_article_div_falls_back_to_all_paragraphs(self):
        assert parse_statement("<html><body><p>one</p><p>Share</p><p>two</p></body></html>") == [
            "one",
            "two",
        ]

    def test_manifest_digest_mismatch_is_refused(self, raw_dir, tmp_path):
        manifest = Manifest.load(raw_dir)
        (raw_dir / "html" / "monetary20230322a.htm").write_text("<p>tampered</p>")
        with pytest.raises(ValueError, match="sha256"):
            from codecheck_fomc.corpus import statements_frame

            statements_frame(raw_dir, manifest, 2023)


class TestPipeline:
    def test_end_to_end(self, raw_dir, tmp_path):
        with Ledger(tmp_path / "ledger") as ledger:
            statements, embeddings, scores = run_all(ledger, raw_dir)

            assert statements.layer == "raw"
            assert sorted(p.name for p in statements.files()) == ["2023.parquet", "2024.parquet"]
            docs = pl.read_parquet(statements.path / "2023.parquet")
            assert docs["doc_id"].to_list() == ["monetary20230322a", "monetary20231213a"]
            assert docs["n_paragraphs"].to_list() == [4, 2]

            paras = pl.read_parquet(embeddings.path / "2023.parquet")
            assert paras.height == 6
            assert embeddings.meta.sources == [statements.artifact_id]
            assert embeddings.meta.model_card.backend == "hashing"
            assert embeddings.meta.hyperparams["backend"] == "hashing"

            table = pl.read_parquet(scores.path / "2024.parquet")
            assert table.height == 10  # one statement x ten narratives
            assert (scores.path / "narratives.parquet").exists()
            assert ledger.ancestors(scores.artifact_id) == sorted(
                [statements.artifact_id, embeddings.artifact_id]
            )
            # the easing statement scores the easing narrative above the tightening one
            by = {r["narrative_id"]: r["cos_para_max"] for r in table.iter_rows(named=True)}
            assert by["rate_cut"] > by["rate_hike"]

            report = verify(ledger)
            assert report.ok, report.summary()

    def test_freeze_then_compare_reproduces_and_detects_a_change(self, raw_dir, tmp_path):
        with Ledger(tmp_path / "ledger_a") as a:
            run_all(a, raw_dir)
            freeze(a, tmp_path / "expected")
        assert (tmp_path / "expected" / "fomc_narrative_scores" / "meta.json").exists()

        with Ledger(tmp_path / "ledger_b") as b:
            run_all(b, raw_dir)
            report = compare(b, tmp_path / "expected")
            assert report.passed, str(report)
            assert "3/3 files hash-identical" in str(report)  # hashing backend: bit for bit

            # a different embedding dimension changes every score: the comparison must fail
        with Ledger(tmp_path / "ledger_c") as c:
            run_all(c, raw_dir, backend=HashingBackend(dim=32))
            report = compare(c, tmp_path / "expected")
            assert not report.passed
            assert "FAIL" in str(report)

    def test_resume_after_a_crash_finishes_the_same_artifact(self, raw_dir, tmp_path):
        class Dies(HashingBackend):
            calls = 0

            def encode(self, texts):
                Dies.calls += 1
                if Dies.calls == 2:  # second unit (2024)
                    raise RuntimeError("killed")
                return super().encode(texts)

        with Ledger(tmp_path / "ledger") as ledger:
            runner = _runner(ledger)
            statements = runner.start(IngestJob(raw_dir))
            with pytest.raises(RuntimeError):
                runner.start(EmbedJob(statements, Dies(dim=64)))
            partial = ledger.list("fomc_embeddings", include_partial=True)
            assert len(partial) == 1 and partial[0].partial
            with pytest.raises(Exception):
                ledger.latest("fomc_embeddings")

            done = runner.resume(partial[0].artifact_id)
            assert not done.partial
            assert done.artifact_id == partial[0].artifact_id
            assert len(done.meta.runs) == 2
            assert sorted(p.name for p in done.files()) == ["2023.parquet", "2024.parquet"]
