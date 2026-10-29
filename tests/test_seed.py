"""Sandbox seeding: Ledger.seed_from and `ledgerworks seed`.

The source (production) ledger must come out byte-identical; the sandbox gets hard
links to the content files and its own copies of the sidecars.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path

import pytest

import ledgerworks.ledger as ledger_module
from ledgerworks.artifact import RunMeta, utc_now_iso
from ledgerworks.cli import main
from ledgerworks.jobs import JobRunner
from ledgerworks.layout import Layout
from ledgerworks.ledger import Ledger, LedgerError
from ledgerworks.meta import META_FILENAME, README_FILENAME, read_meta
from ledgerworks.verify import verify

PIPELINE = "test-pipeline"
VERSION = "v0.1.0"


def _produce(index: Ledger, kind: str, sources=(), files=None, **hyperparams) -> str:
    files = files or {"2010-01.parquet": b"jan", "2010-02.parquet": b"feb"}
    with index.run(
        kind=kind,
        pipeline=PIPELINE,
        pipeline_version=VERSION,
        hyperparams=hyperparams,
        sources=list(sources),
    ) as run:
        for name, data in files.items():
            (run.out_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (run.out_dir / name).write_bytes(data)
    return run.artifact_id


@pytest.fixture
def prod(tmp_path: Path) -> dict:
    """A closed production ledger: raw -> derived (with a sub-directory) -> output."""
    root = tmp_path / "Datalake"
    with Ledger(root, layers={"documents": "raw", "report": "output"}) as index:
        raw = _produce(index, "documents", src="vendor")
        mid = _produce(
            index,
            "embeddings",
            sources=[raw],
            files={"2010-01.parquet": b"e1", "parts/extra.bin": b"nested"},
        )
        out = _produce(index, "report", sources=[mid], files={"report.parquet": b"r"})
    return {"root": root, "raw": raw, "mid": mid, "out": out}


@pytest.fixture
def sandbox(tmp_path: Path):
    with Ledger(tmp_path / "Datalake_sandbox") as index:
        yield index


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _seed(sandbox: Ledger, root: Path, ids, **kwargs) -> list[str]:
    with Ledger(root, create=False) as source:
        return sandbox.seed_from(source, ids, **kwargs)


class TestSeed:
    def test_seeds_ancestors_parents_first(self, prod, sandbox):
        seeded = _seed(sandbox, prod["root"], [prod["out"]])
        assert seeded == [prod["raw"], prod["mid"], prod["out"]]
        assert sandbox.ancestors(prod["out"]) == sorted([prod["raw"], prod["mid"]])
        assert sandbox.latest("report").artifact_id == prod["out"]
        raw = sandbox.get(prod["raw"])
        assert raw.layer == "raw" and raw.path == sandbox.root / "raw" / "documents" / prod["raw"]

    def test_no_ancestors(self, prod, sandbox):
        assert _seed(sandbox, prod["root"], [prod["out"]], with_ancestors=False) == [prod["out"]]
        assert not sandbox.exists(prod["mid"])

    def test_files_linked_sidecars_copied(self, prod, sandbox):
        _seed(sandbox, prod["root"], [prod["mid"]])
        with Ledger(prod["root"], create=False) as source:
            src = source.get(prod["mid"]).path
        dst = sandbox.get(prod["mid"]).path
        for rel in ("2010-01.parquet", "parts/extra.bin"):
            assert os.stat(src / rel).st_ino == os.stat(dst / rel).st_ino
        for name in (META_FILENAME, README_FILENAME):
            assert os.stat(src / name).st_ino != os.stat(dst / name).st_ino
        meta, hashes = read_meta(dst)
        assert meta.notes.startswith(f"seeded from {prod['root']} on {utc_now_iso()[:10]}")
        assert hashes == read_meta(src)[1]
        assert sandbox.get(prod["mid"]).file_hashes  # index row has the hashes

    def test_job_files_copied_lock_not(self, prod, sandbox):
        with Ledger(prod["root"], create=False) as source:
            src = source.get(prod["raw"]).path
        (src / "job.json").write_text("{}")
        (src / "job.lock").write_text(json.dumps({"heartbeat_at": "2000-01-01T00:00:00+00:00"}))
        _seed(sandbox, prod["root"], [prod["raw"]])
        dst = sandbox.get(prod["raw"]).path
        assert (dst / "job.json").read_text() == "{}"
        assert os.stat(src / "job.json").st_ino != os.stat(dst / "job.json").st_ino
        assert not (dst / "job.lock").exists()

    def test_source_byte_identical(self, prod, sandbox):
        before = _snapshot(prod["root"])
        _seed(sandbox, prod["root"], [prod["out"]])
        assert _snapshot(prod["root"]) == before

    def test_new_run_in_sandbox_leaves_source_untouched(self, prod, sandbox):
        _seed(sandbox, prod["root"], [prod["out"]])
        before = _snapshot(prod["root"])
        new = _produce(sandbox, "report", sources=[prod["mid"]], variant="sandbox")
        assert sandbox.parents(new) == [prod["mid"]]
        assert _snapshot(prod["root"]) == before
        with Ledger(prod["root"], create=False) as source:
            assert not source.exists(new)
            assert source.children(prod["mid"]) == [prod["out"]]

    def test_atomic_rewrite_in_sandbox_breaks_the_link(self, prod, sandbox):
        _seed(sandbox, prod["root"], [prod["raw"]])
        with Ledger(prod["root"], create=False) as source:
            src = source.get(prod["raw"]).path / "2010-01.parquet"
        dst = sandbox.get(prod["raw"]).path / "2010-01.parquet"
        assert os.stat(src).st_ino == os.stat(dst).st_ino
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(b"rewritten")
        os.replace(tmp, dst)
        assert src.read_bytes() == b"jan"
        assert dst.read_bytes() == b"rewritten"
        assert os.stat(src).st_ino != os.stat(dst).st_ino

    def test_identical_target_skipped_and_reported(self, prod, sandbox, caplog):
        _seed(sandbox, prod["root"], [prod["out"]])
        with caplog.at_level(logging.WARNING, logger="ledgerworks.ledger"):
            assert _seed(sandbox, prod["root"], [prod["out"]]) == []
        assert caplog.text.count("identical files") == 3

    def test_dry_run_writes_nothing(self, prod, sandbox):
        sandbox.close()
        before_target, before_source = _snapshot(sandbox.root), _snapshot(prod["root"])
        with Ledger(sandbox.root, create=False) as target:
            ids = _seed(target, prod["root"], [prod["out"]], dry_run=True)
        assert ids == [prod["raw"], prod["mid"], prod["out"]]
        assert _snapshot(sandbox.root) == before_target
        assert _snapshot(prod["root"]) == before_source
        assert not any((sandbox.root / layer).exists() for layer in ("raw", "derived", "output"))


class TestSeedRefusals:
    def test_partial_refused_unless_allowed(self, prod, sandbox):
        with Ledger(prod["root"], create=False) as source:
            with pytest.raises(RuntimeError):
                with source.run(kind="draft", pipeline=PIPELINE, pipeline_version=VERSION) as run:
                    (run.out_dir / "a.parquet").write_bytes(b"x")
                    raise RuntimeError("crash")
            draft = run.artifact_id
        with pytest.raises(LedgerError, match="partial"):
            _seed(sandbox, prod["root"], [draft])
        assert not sandbox.exists(draft)
        assert _seed(sandbox, prod["root"], [draft], allow_partial=True) == [draft]
        assert sandbox.get(draft).partial

    def test_deprecated_ancestor_refused_unless_allowed(self, prod, sandbox):
        with Ledger(prod["root"], create=False) as source:
            source.deprecate(prod["raw"], "bad vendor file")
        with pytest.raises(LedgerError, match="deprecated"):
            _seed(sandbox, prod["root"], [prod["out"]])
        assert sandbox.list(include_partial=True, include_deprecated=True) == []
        seeded = _seed(sandbox, prod["root"], [prod["out"]], allow_deprecated=True)
        assert len(seeded) == 3 and sandbox.get(prod["raw"]).deprecated

    def test_live_lock_refused(self, prod, sandbox):
        with Ledger(prod["root"], create=False) as source:
            lock = source.get(prod["mid"]).path / "job.lock"
        lock.write_text(json.dumps({"heartbeat_at": utc_now_iso()}))
        with pytest.raises(LedgerError, match="live job"):
            _seed(sandbox, prod["root"], [prod["out"]])
        with pytest.raises(LedgerError, match="live job"):
            _seed(sandbox, prod["root"], [prod["mid"]], allow_partial=True)
        assert sandbox.list(include_partial=True) == []

    def test_existing_id_with_different_hashes_refused(self, prod, sandbox):
        _seed(sandbox, prod["root"], [prod["raw"]])
        dst = sandbox.get(prod["raw"]).path
        payload = json.loads((dst / META_FILENAME).read_text())
        payload["files"]["2010-01.parquet"]["digest"] = "0" * 64
        (dst / META_FILENAME).write_text(json.dumps(payload))
        with pytest.raises(LedgerError, match="different file hashes"):
            _seed(sandbox, prod["root"], [prod["raw"]])

    def test_other_filesystem_refused_never_copied(self, prod, sandbox, monkeypatch):
        def other_device(source: Path, target: Path) -> None:
            raise LedgerError(f"{source} and {target} are on different filesystems")

        monkeypatch.setattr(ledger_module, "_same_device", other_device)
        with pytest.raises(LedgerError, match="different filesystems"):
            _seed(sandbox, prod["root"], [prod["out"]])
        assert not any((sandbox.root / layer).exists() for layer in ("raw", "derived", "output"))

    def test_hash_mismatch_after_linking_rolls_back(self, prod, sandbox):
        with Ledger(prod["root"], create=False) as source:
            (source.get(prod["raw"]).path / "2010-01.parquet").write_bytes(b"corrupted")
        with pytest.raises(LedgerError, match="recorded hash"):
            _seed(sandbox, prod["root"], [prod["raw"]])
        assert not sandbox.exists(prod["raw"])
        assert not (sandbox.root / "raw" / "documents" / prod["raw"]).exists()

    def test_unknown_id_and_self_refused(self, prod, sandbox):
        with pytest.raises(LedgerError, match="no artifact"):
            _seed(sandbox, prod["root"], ["nope"])
        with Ledger(prod["root"], create=False) as source:
            with pytest.raises(LedgerError, match="itself"):
                source.seed_from(source, [prod["raw"]])


class TestSeedCli:
    def test_seed_and_dry_run(self, prod, sandbox, monkeypatch, capsys):
        sandbox.close()
        monkeypatch.setenv("LEDGER_ROOT", str(sandbox.root))
        args = ["seed", "--from", str(prod["root"]), prod["out"]]

        assert main([*args, "--dry-run"]) == 0
        assert "3 artifact(s) would seed" in capsys.readouterr().out
        assert not (sandbox.root / "output").exists()

        assert main([*args, "--no-ancestors"]) == 0
        assert f"seeded {prod['out']}" in capsys.readouterr().out
        assert main(args) == 0
        out = capsys.readouterr().out
        assert f"seeded {prod['raw']}" in out and f"seeded {prod['out']}" not in out

    def test_refusal_is_an_error(self, prod, sandbox, monkeypatch, capsys):
        sandbox.close()
        with Ledger(prod["root"], create=False) as source:
            source.deprecate(prod["out"], "superseded")
        args = ["--root", str(sandbox.root), "seed", "--from", str(prod["root"]), prod["out"]]
        assert main(args) == 1
        assert "deprecated" in capsys.readouterr().err
        assert main([*args, "--allow-deprecated"]) == 0


def _extend(index: Ledger, artifact_id: str) -> None:
    art = index.get(artifact_id)
    with index.run(
        kind=art.kind,
        pipeline=PIPELINE,
        pipeline_version=VERSION,
        hyperparams=art.meta.hyperparams,
        extend=artifact_id,
    ) as run:
        (run.out_dir / "2010-03.parquet").write_bytes(b"mar")


class TestReadOnlyLinks:
    def test_in_place_write_refused_source_unchanged(self, prod, sandbox):
        _seed(sandbox, prod["root"], [prod["mid"]])
        with Ledger(prod["root"], create=False) as source:
            src = source.get(prod["mid"]).path
        dst = sandbox.get(prod["mid"]).path
        for rel in ("2010-01.parquet", "parts/extra.bin"):
            with pytest.raises(PermissionError):
                open(dst / rel, "r+b")
            assert not os.stat(src / rel).st_mode & 0o222  # shared inode: read-only too
        assert (src / "2010-01.parquet").read_bytes() == b"e1"
        assert (src / "parts" / "extra.bin").read_bytes() == b"nested"
        assert os.access(dst, os.W_OK) and os.access(dst / "parts", os.W_OK)  # dirs untouched
        assert os.access(dst / META_FILENAME, os.W_OK)  # sidecars are copies

    def test_deleting_the_sandbox_leaves_the_source(self, prod, sandbox):
        import shutil

        _seed(sandbox, prod["root"], [prod["out"]])
        sandbox.close()
        shutil.rmtree(sandbox.root)
        with Ledger(prod["root"], create=False) as source:
            for artifact_id in (prod["raw"], prod["mid"], prod["out"]):
                path = source.get(artifact_id).path
                for f in (p for p in path.rglob("*") if p.is_file()):
                    if f.name in (META_FILENAME, README_FILENAME):
                        continue
                    assert os.stat(f).st_nlink == 1
                    assert not os.stat(f).st_mode & 0o222
            assert verify(source, check_content=False).ok

    def test_extend_and_resume_refused(self, prod, sandbox):
        with Ledger(prod["root"], create=False) as source:
            with pytest.raises(RuntimeError):
                with source.run(kind="draft", pipeline=PIPELINE, pipeline_version=VERSION) as run:
                    raise RuntimeError("crash")
            draft = run.artifact_id
        _seed(sandbox, prod["root"], [prod["raw"], draft], allow_partial=True)
        assert sandbox.get(prod["raw"]).meta.seeded_from["mode"] == "link"
        with pytest.raises(LedgerError, match="seeded artifact, read-only in this ledger"):
            _extend(sandbox, prod["raw"])
        with pytest.raises(LedgerError, match="seed with --copy"):
            with sandbox.run(
                kind="draft", pipeline=PIPELINE, pipeline_version=VERSION, resume=draft
            ):
                pass
        assert len(sandbox.get(prod["raw"]).meta.runs) == 1

    def test_job_runner_update_and_resume_refused(self, tmp_path, sandbox):
        from tests.test_jobs import ToyJob, _clean_repo

        layout = Layout("M", date(2008, 1, 1), date(2008, 3, 31))
        repo = _clean_repo(tmp_path)
        with Ledger(tmp_path / "jobs_prod") as prod_index:
            runner = JobRunner(prod_index, repo_dir=repo, handle_signals=False)
            done = runner.start(ToyJob(layout)).artifact_id
            ToyJob.die_at = "2008-02"
            try:
                with pytest.raises(RuntimeError):
                    runner.start(ToyJob(layout, seed=1))
            finally:
                ToyJob.die_at = None
            crashed = prod_index.list("toy_series", include_partial=True)[0].artifact_id
            assert prod_index.get(crashed).partial
            sandbox.seed_from(prod_index, [done, crashed], allow_partial=True)
        sandbox_runner = JobRunner(sandbox, repo_dir=repo, handle_signals=False)
        with pytest.raises(LedgerError, match="seeded artifact"):
            sandbox_runner.update_job(done, ToyJob(layout))
        with pytest.raises(LedgerError, match="seeded artifact"):
            sandbox_runner.resume_job(crashed, ToyJob(layout, seed=1))


class TestCopyMode:
    def test_copy_is_writable_and_extendable_source_untouched(self, prod, sandbox):
        before = _snapshot(prod["root"])
        with Ledger(prod["root"], create=False) as source:
            src = source.get(prod["mid"]).path
            modes = {p: os.stat(p).st_mode for p in src.rglob("*") if p.is_file()}
        assert _seed(sandbox, prod["root"], [prod["mid"]], copy=True) == [prod["raw"], prod["mid"]]
        dst = sandbox.get(prod["mid"]).path
        assert sandbox.get(prod["mid"]).meta.seeded_from["mode"] == "copy"
        for rel in ("2010-01.parquet", "parts/extra.bin"):
            assert os.stat(src / rel).st_ino != os.stat(dst / rel).st_ino
            with open(dst / rel, "r+b") as fh:
                fh.write(b"X")
        _extend(sandbox, prod["mid"])
        assert len(sandbox.get(prod["mid"]).meta.runs) == 2
        assert _snapshot(prod["root"]) == before
        assert {p: os.stat(p).st_mode for p in modes} == modes

    def test_cli_copy(self, prod, sandbox, capsys):
        sandbox.close()
        args = ["--root", str(sandbox.root), "seed", "--from", str(prod["root"]), prod["raw"]]
        assert main([*args, "--copy"]) == 0
        assert f"seeded (copy) {prod['raw']}" in capsys.readouterr().out
        with Ledger(sandbox.root, create=False) as target:
            assert target.get(prod["raw"]).meta.seeded_from["mode"] == "copy"


class TestSeededFromField:
    def test_sidecar_and_readme(self, prod, sandbox):
        _seed(sandbox, prod["root"], [prod["raw"]])
        dst = sandbox.get(prod["raw"]).path
        meta, _ = read_meta(dst)
        assert meta.seeded_from == {
            "root": str(prod["root"]),
            "date": utc_now_iso()[:10],
            "mode": "link",
        }
        assert sandbox.get(prod["raw"]).meta.seeded_from == meta.seeded_from  # index row
        readme = (dst / README_FILENAME).read_text()
        assert "## Seeded" in readme and str(prod["root"]) in readme and "read-only" in readme
        with Ledger(prod["root"], create=False) as source:
            assert source.get(prod["raw"]).meta.seeded_from is None
            assert "seeded_from" not in json.loads(
                (source.get(prod["raw"]).path / META_FILENAME).read_text()
            )

    @pytest.mark.parametrize(
        "seeded_from", [None, {"root": "/mnt/Datalake", "date": "2026-10-29", "mode": "copy"}]
    )
    def test_run_meta_round_trip(self, seeded_from):
        meta = RunMeta(
            kind="k", pipeline=PIPELINE, pipeline_version=VERSION, seeded_from=seeded_from
        )
        payload = meta.to_dict()
        assert ("seeded_from" in payload) is (seeded_from is not None)
        back = RunMeta.from_dict(json.loads(json.dumps(payload)))
        assert back.seeded_from == seeded_from
        assert back.artifact_id == meta.artifact_id
