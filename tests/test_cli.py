"""The ledger CLI never creates a ledger by accident: no root, or a root without an
index, is an error; only `init` creates one."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerworks.cli import main


@pytest.fixture
def no_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env above tmp_path
    return tmp_path


def test_no_root_is_an_error_and_creates_nothing(no_env: Path, capsys):
    assert main(["ls"]) == 1
    assert "no ledger root" in capsys.readouterr().err
    assert list(no_env.iterdir()) == []


def test_a_root_without_index_is_an_error_and_creates_nothing(no_env: Path, capsys):
    root = no_env / "mount" / "Ledger"
    assert main(["--root", str(root), "ls"]) == 1
    assert "no ledger at" in capsys.readouterr().err
    assert not root.exists()


def test_init_creates_then_other_commands_open(no_env: Path, monkeypatch: pytest.MonkeyPatch):
    root = no_env / "lake"
    assert main(["--root", str(root), "init"]) == 0
    assert (root / "index.db").exists()
    monkeypatch.setenv("LEDGER_ROOT", str(root))
    assert main(["ls"]) == 0
