"""codecheck-fomc: run the pipeline, verify it, compare it to the reference.

    codecheck-fomc all        --backend finmodernbert     ingest + embed + score, then compare
    codecheck-fomc ingest
    codecheck-fomc embed      --backend hashing
    codecheck-fomc score      --taxonomy monetary_policy
    codecheck-fomc verify                                 ledgerworks verify on the ledger
    codecheck-fomc compare    [--atol 1e-4] [--min-cos 0.9999]
    codecheck-fomc freeze                                 copy the latest run into expected/
    codecheck-fomc rescore    --no-model                  scores from the committed embeddings
                                                          and narrative vectors (numpy only)
    codecheck-fomc rescore                                score the committed embeddings with
                                                          the model (re-embeds the narratives)
    codecheck-fomc explain    [KIND] [--reference]        everything recorded about the latest
                                                          artifacts (or the committed ones)

The ledger (artifact tree + index) lives in ./ledger by default (``--ledger``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ledgerworks import Ledger, LedgerError, read_meta, verify
from ledgerworks.jobs import JobRunner

from codecheck_fomc import DEFAULT_LEDGER, EXPECTED_DIR, HERE, TAXONOMY_DIR
from codecheck_fomc.backends import make_backend
from codecheck_fomc.compare import KINDS as KIND_NAMES
from codecheck_fomc.compare import compare, freeze
from codecheck_fomc.corpus import IngestJob
from codecheck_fomc.embed import EmbedJob
from codecheck_fomc.explain import explain, summary
from codecheck_fomc.score import ScoreJob, Taxonomy


def _runner(ledger: Ledger, args: argparse.Namespace) -> JobRunner:
    return JobRunner(
        ledger,
        allow_dirty=args.allow_dirty,
        repo_dir=HERE,  # git finds the work tree; uv.lock is the one beside pyproject.toml
        log_level="INFO" if args.verbose else "WARNING",
        handle_signals=False,
    )


def _backend(args: argparse.Namespace):
    if args.backend == "hashing":
        return make_backend("hashing", dim=args.dim)
    return make_backend("finmodernbert", threads=args.threads, batch_size=args.batch_size)


def cmd_ingest(ledger: Ledger, args: argparse.Namespace) -> None:
    art = _runner(ledger, args).start(IngestJob())
    print(summary(art))


def cmd_embed(ledger: Ledger, args: argparse.Namespace) -> None:
    source = ledger.latest("fomc_statements")
    art = _runner(ledger, args).start(EmbedJob(source, _backend(args)))
    print(summary(art))


def cmd_score(ledger: Ledger, args: argparse.Namespace) -> None:
    source = ledger.latest("fomc_embeddings")
    taxonomy = Taxonomy.load(TAXONOMY_DIR / f"{args.taxonomy}.yaml")
    backend = EmbedJob.from_artifact(source, ledger).backend  # the model the vectors came from
    art = _runner(ledger, args).start(ScoreJob(source, taxonomy, backend))
    print(summary(art))


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


def _shown(path: Path) -> str:
    """A path relative to the working directory when it is inside it."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def cmd_rescore_without_model(args: argparse.Namespace) -> int:
    from codecheck_fomc.reference import rescore_without_model

    print(
        "Recomputing the narrative scores from the committed embeddings and narrative vectors\n"
        f"in {_shown(args.expected)} (numpy only, no model), then comparing them with the\n"
        "committed scores.\n"
    )
    report = rescore_without_model(args.expected, show_year=f"{args.year}.parquet")
    print(report)
    print(
        "\nREPRODUCED: the committed scores follow from the committed embeddings"
        if report.passed
        else "\nNOT REPRODUCED"
    )
    return 0 if report.passed else 1


def cmd_rescore(ledger: Ledger, args: argparse.Namespace) -> int:
    from codecheck_fomc.reference import describe, register_reference

    print(
        "Scoring the committed embeddings with the model: the 10 narratives are embedded\n"
        "again, the paragraph vectors are the author's. Inputs from "
        f"{_shown(args.expected)}, checked against their meta.json:\n"
    )
    refs = register_reference(ledger, args.expected, ["fomc_statements", "fomc_embeddings"])
    for ref in refs:
        print(describe(ref))
    source = ledger.get(refs[-1].artifact.artifact_id)
    taxonomy = Taxonomy.load(TAXONOMY_DIR / f"{args.taxonomy}.yaml")
    committed = args.expected / "fomc_narrative_scores" / "meta.json"
    if committed.exists():
        recorded = read_meta(committed.parent)[0].hyperparams.get("taxonomy_sha256")
        if recorded != taxonomy.sha256:
            print(f"\nWARNING: {taxonomy.path.name} differs from the one the reference used")
    backend = EmbedJob.from_artifact(source, ledger).backend
    art = _runner(ledger, args).start(ScoreJob(source, taxonomy, backend))
    print("\n" + summary(art) + "\n        (its input is the committed embeddings)\n")
    report = compare(
        ledger, args.expected, atol=args.atol, min_cos=args.min_cos, kinds=(ScoreJob.kind,)
    )
    print(report)
    print("\nREPRODUCED" if report.passed else "\nNOT REPRODUCED")
    return 0 if report.passed else 1


def cmd_explain(args: argparse.Namespace) -> int:
    from codecheck_fomc.compare import KINDS
    from codecheck_fomc.reference import load_reference

    kinds = [args.kind] if args.kind else list(KINDS)
    blocks = []
    if args.reference:
        for kind in kinds:
            ref = load_reference(args.expected, kind)
            blocks.append(explain(ref.artifact, " (committed reference, expected/)"))
    else:
        with Ledger(args.ledger) as ledger:
            for kind in kinds:
                try:
                    blocks.append(explain(ledger.latest(kind), " (latest in the ledger)"))
                except LedgerError:
                    blocks.append(f"== {kind}: nothing in {args.ledger} yet")
    print("\n\n".join(blocks))
    return 0


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
    rescore = sub.add_parser("rescore", help="score the committed embeddings (see docs)")
    rescore.add_argument(
        "--no-model", action="store_true", help="reuse the committed narrative vectors"
    )
    rescore.add_argument("--year", default="2026", help="year whose table is printed")
    show = sub.add_parser("explain", help="everything recorded about the artifacts")
    show.add_argument("kind", nargs="?", choices=[*KIND_NAMES], help="one kind (default: all)")
    show.add_argument(
        "--reference", action="store_true", help="the committed artifacts in expected/"
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    try:
        return _run(args)
    except (LedgerError, ValueError) as exc:  # refusals the reviewer should read, not a trace
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.command == "explain":
        return cmd_explain(args)  # reads only: opens the ledger itself
    if args.command == "rescore" and args.no_model:
        return cmd_rescore_without_model(args)  # reads expected/ only: no ledger
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
        elif args.command == "rescore":
            return cmd_rescore(ledger, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
