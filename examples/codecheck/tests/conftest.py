"""A small fake raw corpus in the Fed page layout, so the pipeline runs offline."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

PAGE = """<!DOCTYPE html><html><head><title>Federal Reserve Board - FOMC statement</title></head>
<body>
<div id="content">
<div class="row"><p>Skip to main content</p></div>
<div id="article">
  <div class="heading col-xs-12 col-sm-8 col-md-8">
    <p class="article__time">{long_date}</p>
  </div>
  <div class="row">
    <div class="col-xs-12 col-sm-8 col-md-8">
      <h3 class="title">Federal Reserve issues FOMC statement</h3>
      <p class="article__time">For release at 2:00 p.m. EDT</p>
      <p>Share</p>
{paragraphs}
      <p><a href="/newsevents/pressreleases/monetary{stamp}a1.htm">Implementation Note
      issued {pretty}</a></p>
    </div>
    <div class="col-xs-12 col-sm-4 col-md-4">
      <div class="panel panel-related">
        <div class="panel-heading"><h5 class="panel-title">Related Information</h5></div>
        <div class="panel-body">
          <p><a href="/newsevents/pressreleases/monetary{stamp}b.htm">Plans for Reducing
          the Size of the Federal Reserve's Balance Sheet</a></p>
        </div>
      </div>
    </div>
  </div>
</div>
<div id="footer"><p>Last Update: {pretty}</p>
<p>Board of Governors of the Federal Reserve System</p></div>
</div>
</body></html>
"""

# The statement text, exactly as parse_statement must return it.
STATEMENTS: dict[str, list[str]] = {
    "20230322": [
        "Recent indicators point to modest growth in spending and production. Job gains have "
        "picked up in recent months and are running at a robust pace; the unemployment rate has "
        "remained low. Inflation remains elevated.",
        "The U.S. banking system is sound and resilient. Recent developments are likely to result "
        "in tighter credit conditions for households and businesses and to weigh on economic "
        "activity, hiring, and inflation. The extent of these effects is uncertain. The Committee "
        "remains highly attentive to inflation risks.",
        "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent "
        "over the longer run. In support of these goals, the Committee decided to raise the target "
        "range for the federal funds rate to 4-3/4 to 5 percent. The Committee will closely "
        "monitor incoming information and assess the implications for monetary policy. The "
        "Committee anticipates that some additional policy firming may be appropriate in order to "
        "attain a stance of monetary policy that is sufficiently restrictive to return inflation "
        "to 2 percent over time. In addition, the Committee will continue reducing its holdings "
        "of Treasury securities and agency debt and agency mortgage-backed securities, as "
        "described in its previously announced plans.",
    ],
    "20231213": [
        "Recent indicators suggest that growth of economic activity has slowed from its strong "
        "pace in the third quarter. Job gains have moderated since earlier in the year but remain "
        "strong, and the unemployment rate has remained low. Inflation has eased over the past "
        "year but remains elevated.",
        "The Committee decided to maintain the target range for the federal funds rate at 5-1/4 "
        "to 5-1/2 percent. In determining the extent of any additional policy firming that may be "
        "appropriate to return inflation to 2 percent over time, the Committee will take into "
        "account the cumulative tightening of monetary policy, the lags with which monetary policy "
        "affects economic activity and inflation, and economic and financial developments.",
    ],
    "20240918": [
        "Recent indicators suggest that economic activity has continued to expand at a solid "
        "pace. Job gains have slowed, and the unemployment rate has moved up but remains low. "
        "Inflation has made further progress toward the Committee's 2 percent objective but "
        "remains somewhat elevated.",
        "The Committee has gained greater confidence that inflation is moving sustainably toward "
        "2 percent, and judges that the risks to achieving its employment and inflation goals are "
        "roughly in balance. The economic outlook is uncertain, and the Committee is attentive to "
        "the risks to both sides of its dual mandate.",
        "In light of the progress on inflation and the balance of risks, the Committee decided to "
        "lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 "
        "percent. In considering additional adjustments to the target range for the federal funds "
        "rate, the Committee will carefully assess incoming data, the evolving outlook, and the "
        "balance of risks.",
    ],
}


# Vote material the parser drops, in both page formats: before 2026-06 a "Voting for"
# paragraph closes the statement (dissents inside it); from 2026-06 an approval line
# opens it and a "Voting against" paragraph closes it only when someone dissents.
OPENERS: dict[str, list[str]] = {
    "20240918": [
        "The Federal Open Market Committee approved the following statement for release by "
        "a 11 \u2013 1 vote:"
    ],
}
VOTES: dict[str, list[str]] = {
    "20230322": [
        "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, "
        "Vice Chair; and the other members of the Committee."
    ],
    "20231213": [
        "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, "
        "Vice Chair; and the other members of the Committee. Voting against this action was "
        "Esther L. George, who preferred at this meeting to raise the target range."
    ],
    "20240918": [
        "Voting against the monetary policy action was Michelle W. Bowman, who preferred to "
        "lower the target range for the federal funds rate by 1/4 percentage point at this "
        "meeting."
    ],
}


def _long_date(stamp: str) -> str:
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:])).strftime("%B %d, %Y")


def _pretty(stamp: str) -> str:
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"


def write_raw_corpus(raw_dir: Path, statements: dict[str, list[str]] = STATEMENTS) -> Path:
    """Write ``html/`` pages and ``MANIFEST.json`` the way ``fetch_fomc.py`` does."""
    html_dir = raw_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for stamp, paragraphs in sorted(statements.items()):
        name = f"monetary{stamp}a.htm"
        page = OPENERS.get(stamp, []) + paragraphs + VOTES.get(stamp, [])
        body = PAGE.format(
            paragraphs="\n".join(f"      <p>{p}</p>" for p in page),
            stamp=stamp,
            pretty=_pretty(stamp),
            long_date=_long_date(stamp),
        ).encode()
        (html_dir / name).write_bytes(body)
        files[name] = {
            "url": f"https://www.federalreserve.gov/newsevents/pressreleases/{name}",
            "date": _pretty(stamp),
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
            "retrieved_at": "2026-10-14T00:00:00+00:00",
        }
    manifest = {
        "source": "test fixture in the layout of federalreserve.gov statement pages",
        "index_url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "license": "test fixture",
        "since": 2023,
        "until": "2024-12-31",
        "files": files,
    }
    (raw_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return raw_dir


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return write_raw_corpus(tmp_path / "raw")
