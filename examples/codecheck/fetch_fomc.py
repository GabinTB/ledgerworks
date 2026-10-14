#!/usr/bin/env python3
"""Download FOMC statements from federalreserve.gov into data/raw/html/.

This is the only step that touches the network, and the package author runs it
once: the downloaded HTML files are committed, so the reviewer reproduces
everything from the files in the repository. Content on the Board's website is
in the public domain (https://www.federalreserve.gov/disclaimer.htm); the Board
is credited as the source in data/raw/MANIFEST.json.

    python fetch_fomc.py                 # statements from 2021 onward
    python fetch_fomc.py --since 2023    # a later start year

Statement pages follow one pattern, linked from the calendars page:

    https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm

The parsing into a table is NOT done here but in ``codecheck_fomc.corpus`` (a
ledgerworks job), so it is part of the reproducible, offline pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

BASE = "https://www.federalreserve.gov"
CALENDARS = f"{BASE}/monetarypolicy/fomccalendars.htm"
STATEMENT_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
HTML_DIR = RAW / "html"
MANIFEST = RAW / "MANIFEST.json"
USER_AGENT = "ledgerworks-codecheck/0.1 (+https://github.com/GabinTB/ledgerworks)"


def get(url: str, *, retries: int = 3, pause: float = 1.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def statement_dates(index_html: str, since: int) -> list[str]:
    """Every statement date (YYYYMMDD) linked from the calendars page, sorted."""
    dates = sorted({m.group(1) for m in STATEMENT_RE.finditer(index_html)})
    return [d for d in dates if int(d[:4]) >= since]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", type=int, default=2021, help="first year to keep (default 2021)")
    ap.add_argument(
        "--until",
        type=str,
        default=None,
        help="last statement date to keep, YYYY-MM-DD (default: today)",
    )
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args(argv)
    until = args.until or date.today().isoformat()
    until_key = until.replace("-", "")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GET {CALENDARS}")
    index_html = get(CALENDARS).decode("utf-8", errors="replace")
    dates = [d for d in statement_dates(index_html, args.since) if d <= until_key]
    if not dates:
        print("no statement links found on the calendars page", file=sys.stderr)
        return 1
    print(f"{len(dates)} statements from {dates[0]} to {dates[-1]}")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    files = manifest.get("files", {})
    for d in dates:
        name = f"monetary{d}a.htm"
        url = f"{BASE}/newsevents/pressreleases/{name}"
        target = HTML_DIR / name
        if target.exists() and name in files:
            continue
        print(f"GET {url}")
        body = get(url)
        target.write_bytes(body)
        files[name] = {
            "url": url,
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
            "retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        time.sleep(args.pause)

    manifest = {
        "source": "Board of Governors of the Federal Reserve System, FOMC statements",
        "index_url": CALENDARS,
        "license": "Public domain (https://www.federalreserve.gov/disclaimer.htm); "
        "credit the Board as the source",
        "since": args.since,
        "until": until,
        "files": dict(sorted(files.items())),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(files)} files listed in {MANIFEST.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
