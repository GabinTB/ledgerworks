"""fetch_fomc: which linked statements count as meeting statements (offline)."""

from __future__ import annotations

import pytest

from fetch_fomc import decision_days, statement_dates, without_cloudflare_tokens

ROW = """
<div class="row fomc-meeting">
    <div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>{month}</strong></div>
    <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">{days}</div>
    <div class="col-xs-12 col-md-4 col-lg-2">
        <a href="/newsevents/pressreleases/monetary{stamp}a.htm">HTML</a>
    </div>
</div>
"""


def calendar(years: dict[int, list[tuple[str, str, str]]]) -> str:
    """Calendars-page markup: a heading per year (newest first) and its meeting rows."""
    parts = []
    for year, rows in years.items():
        parts.append(f'<h4><a id="{year}">{year} FOMC Meetings</a></h4>')
        parts += [ROW.format(month=m, days=d, stamp=s) for m, d, s in rows]
    return "\n".join(parts)


PAGE = calendar(
    {
        2025: [
            ("July", "29-30", "20250730"),
            ("August", "22 (notation vote)", "20250822"),
            ("September", "16-17*", "20250917"),
        ],
        2022: [("Jan/Feb", "31-1", "20220201"), ("Apr/May", "3-4", "20220504")],
        2020: [("March", "15", "20200315")],
    }
)


def test_decision_day_is_the_last_day_of_the_meeting():
    assert decision_days(PAGE) == {
        "20250730",
        "20250917",
        "20220201",
        "20220504",
        "20200315",
    }


def test_notation_vote_release_is_not_a_meeting_statement():
    dates = statement_dates(PAGE, since=2021)
    assert "20250822" not in dates
    assert dates == ["20220201", "20220504", "20250730", "20250917"]


def test_link_without_a_meeting_row_is_dropped():
    page = PAGE + '<a href="/newsevents/pressreleases/monetary20250101a.htm">x</a>'
    assert "20250101" not in statement_dates(page, since=2021)


def test_unparseable_calendar_is_an_error_not_an_empty_corpus():
    with pytest.raises(ValueError, match="no meeting dates"):
        statement_dates('<a href="/newsevents/pressreleases/monetary20250917a.htm">x</a>', 2021)


def test_cloudflare_tokens_are_masked_and_nothing_else():
    def page(key: str, ray: str, text: str) -> bytes:
        return (
            f'<a href="/cdn-cgi/l/email-protection#{key}">'
            f'<span class="__cf_email__" data-cfemail="{key}">[email protected]</span></a>'
            f"<p>{text}</p><script>window.__CF$cv$params={{r:'{ray}',t:'MTc5MQ=='}}</script>"
        ).encode()

    a = page("83bce1ece7", "a4a6a04d1ab5e5d9", "The Committee decided to raise the target range.")
    b = page("4f702d202b", "a4a6a0549e297a85", "The Committee decided to raise the target range.")
    c = page("4f702d202b", "a4a6a0549e297a85", "The Committee decided to lower the target range.")
    assert a != b
    assert without_cloudflare_tokens(a) == without_cloudflare_tokens(b)
    assert without_cloudflare_tokens(b) != without_cloudflare_tokens(c)
