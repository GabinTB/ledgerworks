"""codecheck-fomc: run the pipeline, verify it, compare it to the reference.

    codecheck-fomc all        --backend finmodernbert     ingest + embed + score, then compare
    codecheck-fomc ingest
    codecheck-fomc embed      --backend hashing
    codecheck-fomc score      --taxonomy monetary_policy
    codecheck-fomc verify                                 ledgerworks verify on the ledger
    codecheck-fomc compare    [--atol 1e-4] [--min-cos 0.9999]
    codecheck-fomc freeze                                 copy the latest run into expected/

The ledger (artifact tree + index) lives in ./ledger by default (``--ledger``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ledgerworks import Ledger, verify
from ledgerworks.jobs import JobRunner

from codecheck_fomc import DEFAULT_LEDGER, EXPECTED_DIR, HERE, TAXONOMY_DIR
from codecheck_fomc.backends import make_backend
from codecheck_fomc.compare import compare, freeze
from codecheck_fomc.corpus import IngestJob
from codecheck_fomc.embed import EmbedJob
from codecheck_fomc.score import ScoreJob, Taxonomy


def _runner(ledger: Ledger, args: argparse.Namespace) -> JobRunner:
    return JobRunner(
        ledger,
        allow_dirty=args.allow_dirty,
        repo_dir=HERE.parent.parent,
        log_level="INFO" if args.verbose else "WARNING",
        handle_signals=False,
    )


def _backend(args: argparse.Namespace):
    if args.backend == "hashing":
        return make_backend("hashing", dim=args.dim)
    return make_backend("finmodernbert", threads=args.threads, batch_size=args.batch_size)


def cmd_ingest(ledger: Ledger, args: argparse.Namespace) -> None:
    art = _runner(ledger, args).start(IngestJob())
    print(f"fomc_statements          {art.artifact_id}")


def cmd_embed(ledger: Ledger, args: argparse.Namespace) -> None:
    source = ledger.latest("fomc_statements")
    art = _runner(ledger, args).start(EmbedJob(source, _backend(args)))
    print(f"fomc_embeddings          {art.artifact_id}")


def cmd_score(ledger: Ledger, args: argparse.Namespace) -> None:
    source = ledger.latest("fomc_embeddings")
    taxonomy = Taxonomy.load(TAXONOMY_DIR / f"{args.taxonomy}.yaml")
    backend = EmbedJob.from_artifact(source, ledger).backend  # the model the vectors came from
    art = _runner(ledger, args).start(ScoreJob(source, taxonomy, backend))
    print(f"fomc_narrative_scores    {art.artifact_id}")


def cmd_verify(ledger: Ledger, args: argparse.Namespace) -> int:
    report = verify(ledger)
    print(report.summary())
    return 0 if report.ok else 1


def cmd_compare(ledger: Ledger, args: argparse.Namespace) -> int:
    report = compare(ledger, args.expected, atol=args.atol, min_cos=args.min_cos)
    print(report)
    if report.no_reference:
        print("\nNO REFERENCE YET: run `codecheck-fomc freeze`")
    else:
        print("\nREPRODUCED" if report.passed else "\nNOT REPRODUCED")
    return 0 if report.passed else 1


def cmd_freeze(ledger: Ledger, args: argparse.Namespace) -> None:
    ids = freeze(ledger, args.expected)
    for kind, aid in ids.items():
        print(f"{kind:24s} -> expected/{kind}  ({aid})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="codecheck-fomc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="ledger root")
    ap.add_argument("--expected", type=Path, default=EXPECTED_DIR, help="reference outputs")
    ap.add_argument("--backend", choices=["finmodernbert", "hashing"], default="finmodernbert")
    ap.add_argument("--taxonomy", default="monetary_policy", help="file stem under taxonomy/")
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256, help="hashing backend dimension")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--allow-dirty", action="store_true", help="run from a dirty git tree")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("all", "ingest", "embed", "score", "verify", "compare", "freeze"):
        sub.add_parser(name)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    with Ledger(args.ledger) as ledger:
        if args.command == "all":
            cmd_ingest(ledger, args)
            cmd_embed(ledger, args)
            cmd_score(ledger, args)
            rc = cmd_verify(ledger, args)
            return max(rc, cmd_compare(ledger, args))
        if args.command == "ingest":
            cmd_ingest(ledger, args)
        elif args.command == "embed":
            cmd_embed(ledger, args)
        elif args.command == "score":
            cmd_score(ledger, args)
        elif args.command == "verify":
            return cmd_verify(ledger, args)
        elif args.command == "compare":
            return cmd_compare(ledger, args)
        elif args.command == "freeze":
            cmd_freeze(ledger, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
