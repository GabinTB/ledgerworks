"""fomc_statements: the raw corpus, parsed from the committed HTML files.

The HTML under ``data/raw/html`` (downloaded once by ``fetch_fomc.py``, listed in
``data/raw/MANIFEST.json``) is the immutable input. This job turns it into one
parquet per year with one row per statement, so the parsing is itself a recorded,
re-runnable step: the raw artifact's hyperparams carry the manifest digest and
the parser version.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import polars as pl
from ledgerworks import Artifact, Ledger
from ledgerworks.jobs import Job, JobContext, Unit, register_job
from ledgerworks.layout import Layout
from ledgerworks.verify import Finding, Severity

from codecheck_fomc import DATA_RAW, PIPELINE, PIPELINE_REPO, PIPELINE_VERSION

PARSER_VERSION = "3"

# Paragraphs that are page furniture, not statement text.
_BOILERPLATE = re.compile(
    r"^(For release at|For immediate release|Last Update:|Share|Implementation Note"
    r"|For media inquiries|Board of Governors of the Federal Reserve System|Back to Top)",
    re.IGNORECASE,
)
# The release date printed above the title ("January 27, 2021"), not statement text.
_DATE_ONLY = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|November|December)"
    r" \d{1,2}, \d{4}$"
)
# Who voted how, not what the Committee said. Before 2026-06 a "Voting for ..." paragraph
# closes the statement, with dissents ("Voting against this action was X, who preferred
# ...") inside it; from 2026-06 an approval line opens it and a "Voting against ..."
# paragraph closes it when someone dissents.
_VOTE = re.compile(
    r"^Voting (for|against) (this|the) (monetary policy )?action"
    r"|^The Federal Open Market Committee approved the following statement for release"
    r" by a \d+ ?[\u2013\u2014-] ?\d+ vote:?$"
    r"|^[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+,? (who (preferred|supported)|dissent(ed|ing))\b"
)
_WS = re.compile(r"\s+")


class _Article(HTMLParser):
    """Collect every <p> outside sidebar panels as (inside the article div?, text)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[tuple[bool, str]] = []
        self._depth = 0  # div nesting inside <div id="article">
        self._panel = 0  # div nesting inside a <div class="panel ..."> (related links sidebar)
        self._in_p = False
        self._buf: list[str] = []

    @property
    def _in_article(self) -> bool:
        return self._depth > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "div":
            if self._in_article:
                self._depth += 1
            elif a.get("id") == "article":
                self._depth = 1
            if self._panel:
                self._panel += 1
            elif "panel" in (a.get("class") or "").split():
                self._panel = 1
        elif tag == "p":
            self._in_p, self._buf = True, []
        elif tag == "br" and self._in_p:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            if self._in_article:
                self._depth -= 1
            if self._panel:
                self._panel -= 1
        elif tag == "p" and self._in_p:
            self._in_p = False
            text = _WS.sub(" ", "".join(self._buf)).strip()
            if text and not self._panel:
                self.paragraphs.append((self._in_article, text))

    def handle_data(self, data: str) -> None:
        if self._in_p:
            self._buf.append(data)


def parse_statement(html: str) -> list[str]:
    """The statement's paragraphs, in order, without page furniture.

    Paragraphs inside ``<div id="article">`` when the page has one (every
    federalreserve.gov press release does), else every paragraph on the page.
    Dropped: page furniture, the release-date line, sidebar panels such as
    "Related Information", and vote paragraphs (who voted for or against, the
    2026-format approval line).
    """
    parser = _Article()
    parser.feed(html)
    inside = [t for in_article, t in parser.paragraphs if in_article]
    paragraphs = inside or [t for _, t in parser.paragraphs]
    return [
        p
        for p in paragraphs
        if not (_BOILERPLATE.match(p) or _DATE_ONLY.match(p) or _VOTE.match(p))
    ]


@dataclass(frozen=True)
class Manifest:
    path: Path
    files: dict[str, dict[str, Any]]
    sha256: str

    @classmethod
    def load(cls, raw_dir: Path = DATA_RAW) -> Manifest:
        path = raw_dir / "MANIFEST.json"
        body = path.read_bytes()
        return cls(path, json.loads(body)["files"], hashlib.sha256(body).hexdigest())

    def years(self) -> list[int]:
        return sorted({int(f["date"][:4]) for f in self.files.values()})


def statements_frame(raw_dir: Path, manifest: Manifest, year: int) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, entry in sorted(manifest.files.items()):
        if int(entry["date"][:4]) != year:
            continue
        body = (raw_dir / "html" / name).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{name}: sha256 {digest} differs from MANIFEST ({entry['sha256']})")
        paragraphs = parse_statement(body.decode("utf-8", errors="replace"))
        if not paragraphs:
            raise ValueError(f"{name}: no statement paragraphs found")
        rows.append(
            {
                "doc_id": name.removesuffix(".htm"),
                "date": entry["date"],
                "url": entry["url"],
                "html_sha256": digest,
                "n_paragraphs": len(paragraphs),
                "paragraphs": paragraphs,
                "text": "\n\n".join(paragraphs),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "doc_id": pl.String,
            "date": pl.String,
            "url": pl.String,
            "html_sha256": pl.String,
            "n_paragraphs": pl.Int64,
            "paragraphs": pl.List(pl.String),
            "text": pl.String,
        },
    ).with_columns(pl.col("date").str.to_date())


@register_job
class IngestJob(Job):
    """Parse the committed HTML into one parquet per year (raw layer)."""

    kind = "fomc_statements"
    pipeline = PIPELINE
    pipeline_repo = PIPELINE_REPO
    pipeline_version = PIPELINE_VERSION
    layer = "raw"

    def __init__(self, raw_dir: Path = DATA_RAW, manifest: Manifest | None = None) -> None:
        self.raw_dir = Path(raw_dir)
        self.manifest = manifest or Manifest.load(self.raw_dir)
        years = self.manifest.years()
        self.layout = Layout("Y", _jan1(years[0]), _dec31(years[-1]))

    def params(self) -> dict[str, Any]:
        return {
            **self.layout.hyperparams(),
            "manifest_sha256": self.manifest.sha256,
            "parser_version": PARSER_VERSION,
            "n_files": len(self.manifest.files),
        }

    def notes(self) -> str:
        return "FOMC statements, Board of Governors of the Federal Reserve System (public domain)."

    def units(self) -> list[Unit]:
        return [Unit(p.key) for p in self.layout.expected()]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / f"{unit.key}.parquet").exists()

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        frame = statements_frame(self.raw_dir, self.manifest, int(unit.key))
        _write_atomic(frame, ctx.out_dir / f"{unit.key}.parquet")
        ctx.log.info("%s: %d statements", unit.key, frame.height)

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: Ledger, **kwargs: Any) -> IngestJob:
        job = cls(**kwargs)
        if job.manifest.sha256 != artifact.meta.hyperparams.get("manifest_sha256"):
            raise ValueError("MANIFEST.json changed since the artifact was started")
        return job


def verify_artifact(artifact: Artifact) -> list[Finding]:
    findings: list[Finding] = []
    for path in artifact.files("*.parquet"):
        frame = pl.read_parquet(path)
        empty = frame.filter(pl.col("n_paragraphs") == 0).height
        if empty:
            findings.append(
                Finding(
                    Severity.ERROR,
                    artifact.artifact_id,
                    f"{path.name}: {empty} statements without paragraphs",
                )
            )
    return findings


def _jan1(year: int):
    from datetime import date

    return date(year, 1, 1)


def _dec31(year: int):
    from datetime import date

    return date(year, 12, 31)


def _write_atomic(frame: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)
