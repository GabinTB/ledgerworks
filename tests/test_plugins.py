"""Entry-point discovery: canonical group first, legacy group names still honoured."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from ledgerworks import plugins


def _fake_entry_points(table: dict[str, list[EntryPoint]]):
    def entry_points(*, group: str):
        return table.get(group, [])

    return entry_points


def test_group_names_canonical_first():
    assert plugins.group_names("jobs") == ["ledgerworks.jobs", "narrative_finance.jobs"]
    with pytest.raises(ValueError):
        plugins.group_names("nope")


def test_legacy_group_is_read_and_canonical_wins(monkeypatch):
    canonical = EntryPoint("a", "json:loads", "ledgerworks.verifiers")
    legacy_a = EntryPoint("a", "json:dumps", "narrative_finance.verifiers")
    legacy_b = EntryPoint("b", "json:dumps", "narrative_finance.verifiers")
    monkeypatch.setattr(
        plugins,
        "entry_points",
        _fake_entry_points(
            {
                "ledgerworks.verifiers": [canonical],
                "narrative_finance.verifiers": [legacy_a, legacy_b],
            }
        ),
    )
    found = {ep.name: ep.value for ep in plugins.iter_entry_points("verifiers")}
    assert found == {"a": "json:loads", "b": "json:dumps"}
