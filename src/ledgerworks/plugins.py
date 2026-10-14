"""Entry-point discovery for the three plugin groups.

A package contributes jobs, verifiers or equivalence checks by declaring them
in its ``pyproject.toml``::

    [project.entry-points."ledgerworks.jobs"]
    my_kind = "my_package.jobs:MyJob"

    [project.entry-points."ledgerworks.verifiers"]
    my_kind = "my_package.jobs:verify_artifact"

    [project.entry-points."ledgerworks.equivalence"]
    my_kind = "my_package.jobs:equivalence_checks"

``LEGACY_PREFIXES`` lists older group prefixes that are still honoured so that
projects written against the pre-release package keep working without edits.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import EntryPoint, entry_points

PREFIX = "ledgerworks"
LEGACY_PREFIXES: tuple[str, ...] = ("narrative_finance",)

GROUPS = ("jobs", "verifiers", "equivalence")


def group_names(group: str) -> list[str]:
    """Every entry-point group name consulted for ``group`` (canonical first)."""
    if group not in GROUPS:
        raise ValueError(f"unknown plugin group {group!r}, expected one of {GROUPS}")
    return [f"{PREFIX}.{group}", *(f"{p}.{group}" for p in LEGACY_PREFIXES)]


def iter_entry_points(group: str) -> Iterator[EntryPoint]:
    """Entry points of ``group`` across the canonical and legacy group names.

    When the same name is declared under several groups, the first one in
    ``group_names`` order wins (canonical before legacy).
    """
    seen: set[str] = set()
    for name in group_names(group):
        for ep in entry_points(group=name):
            if ep.name in seen:
                continue
            seen.add(ep.name)
            yield ep
