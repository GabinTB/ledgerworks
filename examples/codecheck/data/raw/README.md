# data/raw

Produced once by the package author with `python fetch_fomc.py` (see the
example README), then committed:

- `html/monetaryYYYYMMDDa.htm`: each FOMC statement page, byte for byte as
  served by federalreserve.gov (public domain; source: Board of Governors of the
  Federal Reserve System).
- `MANIFEST.json`: URL, date, SHA-256, size and retrieval time of every page.

The pipeline's first job (`codecheck-fomc ingest`) parses these files; it
refuses a file whose digest no longer matches the manifest.
