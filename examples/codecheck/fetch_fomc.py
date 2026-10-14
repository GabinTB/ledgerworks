#!/usr/bin/env python3
"""Download FOMC statements from federalreserve.gov into data/raw/html/.

This is the only step that touches the network, and the package author runs it
once: the downloaded HTML files are committed, so the reviewer reproduces
everything from the files in the repository. Content on the Board's website is
in the public domain (https://www.federalreserve.gov/disclaimer.htm); the Board
is credited as the source in data/raw/MANIFEST.json.

    python fetch_fomc.py                 # statements from 2021 onward
    python fetch_fomc.py --since 2023    # a later start year
    python fetch_fomc.py --check         # re-fetch and compare with MANIFEST.json, write nothing

Statement pages follow one pattern, linked from the calendars page:

    https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm

Only statements dated on a meeting's decision day are kept: the last day of a
meeting row on the calendars page ("16-17" -> the 17th; "Jan/Feb 31-1" -> 1 Feb).
Other releases linked with the same URL pattern, such as the notation-vote
revision of the Statement on Longer-Run Goals on 22 August 2025, are left out.

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


YEAR_RE = re.compile(r">(\d{4}) FOMC Meetings<")
MEETING_RE = re.compile(
    r"fomc-meeting__month[^>]*>\s*<strong>([^<]*)</strong>\s*</div>\s*"
    r'<div class="fomc-meeting__date[^"]*">([^<]*)</div>'
)
# A meeting's days: "28", "16-17", "16-17*" (* = projections). Anything else, such
# as "22 (notation vote)", is not a meeting.
MEETING_DAYS_RE = re.compile(r"(\d{1,2})(?:-(\d{1,2}))?\*?")
MONTHS = {
    m: i
    for i, m in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
    )
}


def decision_days(index_html: str) -> set[str]:
    """The last day (YYYYMMDD) of every meeting on the calendars page."""
    years = [(m.start(), int(m.group(1))) for m in YEAR_RE.finditer(index_html)]
    days: set[str] = set()
    for m in MEETING_RE.finditer(index_html):
        span = MEETING_DAYS_RE.fullmatch(m.group(2).strip())
        heading = [y for pos, y in years if pos < m.start()]
        if not span or not heading:
            continue
        month = MONTHS[m.group(1).strip().split("/")[-1][:3]]  # "Jan/Feb": decided in Feb
        day = int(span.group(2) or span.group(1))
        days.add(date(heading[-1], month, day).strftime("%Y%m%d"))
    return days


def statement_dates(index_html: str, since: int) -> list[str]:
    """Statement dates (YYYYMMDD) linked from the calendars page on a decision day, sorted."""
    linked = {m.group(1) for m in STATEMENT_RE.finditer(index_html)}
    meetings = decision_days(index_html)
    if not meetings:
        raise ValueError("no meeting dates found on the calendars page: layout changed?")
    return sorted(d for d in linked & meetings if int(d[:4]) >= since)


def entry(d: str, body: bytes) -> dict:
    name = f"monetary{d}a.htm"
    return {
        "url": f"{BASE}/newsevents/pressreleases/{name}",
        "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
        "retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


# Cloudflare rewrites every response: email links are XOR-obfuscated with a random key
# and a challenge script carries a ray id and timestamp. These change per request, so
# raw digests of a re-fetch never match; --check compares pages with them masked.
CLOUDFLARE_TOKENS = re.compile(
    rb"(?<=/cdn-cgi/l/email-protection#)[0-9a-f]+"
    rb"|(?<=data-cfemail=\")[0-9a-f]+"
    rb"|(?<=__CF\$cv\$params=\{r:')[^']*(?=',t:')"
    rb"|(?<=',t:')[^']*(?='\})"
)


def without_cloudflare_tokens(body: bytes) -> bytes:
    return CLOUDFLARE_TOKENS.sub(b"", body)


def check(args: argparse.Namespace) -> int:
    """Re-fetch the index and every statement; compare with MANIFEST.json. Writes nothing.

    Checks that the calendar lists the same statements as the MANIFEST (same url
    and date), that every committed file still has its MANIFEST sha256, and that
    every live page equals the committed file apart from Cloudflare's per-request
    tokens. Raw digests of the live pages are reported, never compared.
    """
    manifest = json.loads(MANIFEST.read_text())
    since, until = args.since or manifest["since"], args.until or manifest["until"]
    print(f"GET {CALENDARS}")
    index_html = get(CALENDARS).decode("utf-8", errors="replace")
    dates = [d for d in statement_dates(index_html, since) if d <= until.replace("-", "")]
    committed = manifest["files"]
    fetched = {f"monetary{d}a.htm": d for d in dates}
    problems = [f"missing from MANIFEST: {n}" for n in sorted(set(fetched) - set(committed))]
    problems += [f"not on the calendar: {n}" for n in sorted(set(committed) - set(fetched))]
    raw_equal = 0
    for name in sorted(set(fetched) & set(committed)):
        want = committed[name]
        expected = entry(fetched[name], b"")
        for key in ("url", "date"):
            if want[key] != expected[key]:
                problems.append(f"{name}: MANIFEST {key} {want[key]!r}, calendar {expected[key]!r}")
        local = (HTML_DIR / name).read_bytes()
        if hashlib.sha256(local).hexdigest() != want["sha256"]:
            problems.append(f"{name}: committed file does not match its MANIFEST sha256")
        live = get(want["url"])
        raw_equal += hashlib.sha256(live).hexdigest() == want["sha256"]
        if without_cloudflare_tokens(live) != without_cloudflare_tokens(local):
            problems.append(f"{name}: live page differs beyond Cloudflare tokens")
        time.sleep(args.pause)
    for line in problems:
        print(f"  MISMATCH  {line}")
    print(
        f"{raw_equal}/{len(fetched)} live pages byte-identical"
        " (Cloudflare tokens differ per request)"
    )
    print(
        f"{len(fetched)} statements from {since} to {until}: "
        + ("MANIFEST matches" if not problems else f"{len(problems)} mismatch(es)")
    )
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", type=int, default=None, help="first year to keep (default 2021)")
    ap.add_argument(
        "--until",
        type=str,
        default=None,
        help="last statement date to keep, YYYY-MM-DD (default: today)",
    )
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument(
        "--check",
        action="store_true",
        help="re-fetch and compare with MANIFEST.json (its since/until by default); write nothing",
    )
    args = ap.parse_args(argv)
    if args.check:
        return check(args)
    since = args.since or 2021
    until = args.until or date.today().isoformat()
    until_key = until.replace("-", "")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GET {CALENDARS}")
    index_html = get(CALENDARS).decode("utf-8", errors="replace")
    dates = [d for d in statement_dates(index_html, since) if d <= until_key]
    if not dates:
        print("no statement links found on the calendars page", file=sys.stderr)
        return 1
    print(f"{len(dates)} statements from {dates[0]} to {dates[-1]}")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    wanted = {f"monetary{d}a.htm" for d in dates}
    files = {n: e for n, e in manifest.get("files", {}).items() if n in wanted}
    for d in dates:
        name = f"monetary{d}a.htm"
        url = f"{BASE}/newsevents/pressreleases/{name}"
        target = HTML_DIR / name
        if target.exists() and name in files:
            continue
        print(f"GET {url}")
        body = get(url)
        target.write_bytes(body)
        files[name] = entry(d, body)
        time.sleep(args.pause)

    manifest = {
        "source": "Board of Governors of the Federal Reserve System, FOMC statements",
        "index_url": CALENDARS,
        "license": "Public domain (https://www.federalreserve.gov/disclaimer.htm); "
        "credit the Board as the source",
        "since": since,
        "until": until,
        "files": dict(sorted(files.items())),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(files)} files listed in {MANIFEST.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
