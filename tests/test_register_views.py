"""register_views: one DuckDB view per (alias, artifact), and Ledger.register on top of it."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerworks import register_views
from ledgerworks.ledger import Ledger, LedgerError

duckdb = pytest.importorskip("duckdb")

PIPELINE = "test-pipeline"
VERSION = "v0.1.0"


@pytest.fixture
def index(tmp_path: Path) -> Ledger:
    with Ledger(tmp_path / "ledger") as idx:
        yield idx


@pytest.fixture
def conn():
    con = duckdb.connect()
    yield con
    con.close()


def _parquet_artifact(index: Ledger, kind: str, select: str, **hyperparams):
    """A complete artifact holding one parquet file written by DuckDB from ``select``."""
    with index.run(
        kind=kind, pipeline=PIPELINE, pipeline_version=VERSION, hyperparams=hyperparams
    ) as run:
        with duckdb.connect() as con:
            con.execute(f"COPY ({select}) TO '{run.out_dir / '2010-01.parquet'}' (FORMAT parquet)")
    return index.get(run.artifact_id)


def test_two_views_joined_in_one_statement(index, conn):
    docs = _parquet_artifact(
        index, "documents", "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) t(doc_id, text)"
    )
    scores = _parquet_artifact(
        index, "scores", "SELECT * FROM (VALUES (1, 0.5), (2, 0.25)) t(doc_id, score)"
    )
    names = register_views(conn, {"docs": docs, "scores": scores})
    assert names == ["docs", "scores"]
    rows = conn.execute(
        "SELECT d.text, s.score FROM docs d JOIN scores s USING (doc_id) ORDER BY doc_id"
    ).fetchall()
    assert rows == [("a", 0.5), ("b", 0.25)]


def test_returns_names_in_input_order(index, conn):
    docs = _parquet_artifact(index, "documents", "SELECT 1 AS x")
    assert register_views(conn, {"z_view": docs, "a_view": docs}) == ["z_view", "a_view"]


@pytest.mark.parametrize("alias", ["drop table x", "a;b", "x-y", ""])
def test_unsafe_alias_refused_and_nothing_created(index, conn, alias):
    docs = _parquet_artifact(index, "documents", "SELECT 1 AS x")
    with pytest.raises(LedgerError, match="unsafe view name"):
        register_views(conn, {"ok_first": docs, alias: docs})
    assert conn.execute("SELECT count(*) FROM duckdb_views() WHERE NOT internal").fetchone() == (0,)


def test_partial_refused(index, conn):
    with pytest.raises(RuntimeError):
        with index.run(kind="documents", pipeline=PIPELINE, pipeline_version=VERSION) as run:
            (run.out_dir / "2010-01.parquet").write_bytes(b"half")
            raise RuntimeError("crash")
    partial = index.list("documents", include_partial=True)[0]
    with pytest.raises(LedgerError, match=partial.artifact_id):
        register_views(conn, {"docs": partial})


def test_ledger_register_unchanged(index, conn):
    _parquet_artifact(index, "documents", "SELECT 7 AS x")
    assert index.register(conn, "documents") == "documents"
    assert conn.execute("SELECT x FROM documents").fetchall() == [(7,)]
    assert index.register(conn, "documents", alias="docs") == "docs"
    with pytest.raises(LedgerError, match="unsafe view name"):
        index.register(conn, "documents", alias="bad name")
