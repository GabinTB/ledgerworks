"""The whole pipeline, offline, on the fake corpus with the hashing backend."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import polars as pl
import pytest
from ledgerworks import Ledger, verify
from ledgerworks.jobs import JobRunner

from codecheck_fomc.backends import HashingBackend
from codecheck_fomc.cli import main
from codecheck_fomc.compare import compare, freeze
from codecheck_fomc.corpus import IngestJob, Manifest, parse_statement
from codecheck_fomc.embed import EmbedJob
from codecheck_fomc.score import ScoreJob, Taxonomy
from tests.conftest import OPENERS, STATEMENTS, VOTES

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

    def test_release_date_line_is_dropped(self, raw_dir):
        html = (raw_dir / "html" / "monetary20230322a.htm").read_text()
        assert '<p class="article__time">March 22, 2023</p>' in html
        assert "March 22, 2023" not in parse_statement(html)

    def test_only_date_only_paragraphs_are_dropped(self):
        html = "<p>May 04, 2022</p><p>The Committee met on May 04, 2022.</p>"
        assert parse_statement(html) == ["The Committee met on May 04, 2022."]

    def test_related_information_panel_is_dropped(self, raw_dir):
        html = (raw_dir / "html" / "monetary20230322a.htm").read_text()
        assert "Related Information" in html
        paragraphs = parse_statement(html)
        assert not any("Plans for Reducing" in p for p in paragraphs)
        assert paragraphs[-1] == STATEMENTS["20230322"][-1]

    @pytest.mark.parametrize("stamp", ["20230322", "20231213", "20240918"])
    def test_vote_paragraphs_are_dropped_in_both_formats(self, raw_dir, stamp):
        html = (raw_dir / "html" / f"monetary{stamp}a.htm").read_text()
        for vote in OPENERS.get(stamp, []) + VOTES[stamp]:
            assert vote in html
        assert parse_statement(html) == STATEMENTS[stamp]

    @pytest.mark.parametrize(
        "vote",
        [
            "The Federal Open Market Committee approved the following statement for release "
            "by a 12 - 0 vote:",
            "Voting against this action were Stephen I. Miran, who preferred to lower the "
            "target range; and Jeffrey R. Schmid, who preferred no change.",
            "Jeffrey R. Schmid dissented, preferring no change to the target range.",
            "Christopher J. Waller, who supported no change for the federal funds target range "
            "but preferred to continue the current pace of decline in securities holdings.",
        ],
    )
    def test_vote_and_dissent_phrasings_are_dropped(self, vote):
        text = "The Committee decided to maintain the target range for the federal funds rate."
        assert parse_statement(f"<p>{text}</p><p>{vote}</p>") == [text]

    def test_statement_text_about_preferences_is_kept(self):
        text = "Households preferred to save, and the Committee approved the following plans."
        assert parse_statement(f"<p>{text}</p>") == [text]

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
            assert docs["n_paragraphs"].to_list() == [3, 2]

            paras = pl.read_parquet(embeddings.path / "2023.parquet")
            assert paras.height == 5
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

    def test_compare_without_reference_exits_zero(self, raw_dir, tmp_path, capsys):
        # regression: `all && freeze` on a first run never reached freeze
        expected = tmp_path / "expected"
        expected.mkdir()
        (expected / "README.md").write_text("reference run\n")
        with Ledger(tmp_path / "ledger") as ledger:
            run_all(ledger, raw_dir)
            report = compare(ledger, expected)
        assert report.passed and report.no_reference
        args = ["--ledger", str(tmp_path / "ledger"), "--expected", str(expected), "compare"]
        assert main(args) == 0
        assert "NO REFERENCE YET" in capsys.readouterr().out

    def test_compare_with_a_partial_reference_fails(self, raw_dir, tmp_path):
        with Ledger(tmp_path / "ledger") as ledger:
            run_all(ledger, raw_dir)
            freeze(ledger, tmp_path / "expected")
            shutil.rmtree(tmp_path / "expected" / "fomc_embeddings")
            report = compare(ledger, tmp_path / "expected")
        assert not report.passed and not report.no_reference

    def test_cli_records_this_package_lockfile(self, tmp_path):
        # regression: repo_dir was the git root, so env recorded the root uv.lock
        lock = Path(__file__).resolve().parent.parent / "uv.lock"
        args = ["--ledger", str(tmp_path / "ledger"), "--allow-dirty", "ingest"]
        assert main(args) == 0
        with Ledger(tmp_path / "ledger") as ledger:
            env = ledger.latest("fomc_statements").meta.runs[-1].env
        assert env["uv_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
        assert "polars" in env["dependencies"]

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
