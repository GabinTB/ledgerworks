"""ledgerworks: run, record, verify and trace pipeline artifacts.

Every output of a pipeline is registered as an immutable artifact directory
with a ``meta.json`` sidecar (identity, hyperparameters, inputs, model, one
record per execution with its git commit, file hashes) and a SQLite index
that can always be rebuilt from those sidecars.

    from ledgerworks import Ledger

    ledger = Ledger("/data/ledger")
    source = ledger.latest("documents")

    with ledger.run(kind="document_embeddings", pipeline="my-project",
                    pipeline_version="v0.1.0", sources=[source],
                    verifier="document_embeddings") as run:
        ...  # write outputs into run.out_dir

Optional pieces: ``ledgerworks.jobs`` (resumable long runs), ``ledgerworks.periods``
and ``ledgerworks.layout`` (calendar partitions), ``ledgerworks.relink`` and
``ledgerworks.equivalence`` (re-register outputs under new inputs, checked;
needs the ``relink`` extra).
"""

from ledgerworks.artifact import (
    Artifact,
    ModelCard,
    RunMeta,
    RunRecord,
    utc_now_iso,
)
from ledgerworks.ledger import DEFAULT_LAYER, VALID_LAYERS, Ledger, LedgerError, RunHandle
from ledgerworks.meta import git_commit, hash_file, read_meta, write_sidecars
from ledgerworks.verify import (
    Finding,
    Severity,
    VerifyReport,
    discover_verifiers,
    verify,
)

__version__ = "0.1.0"

# Names from the pre-release package, kept so older code keeps importing.
DatalakeIndex = Ledger
DatalakeError = LedgerError

__all__ = [
    "Artifact",
    "DEFAULT_LAYER",
    "DatalakeError",
    "DatalakeIndex",
    "Finding",
    "Ledger",
    "LedgerError",
    "ModelCard",
    "RunHandle",
    "RunMeta",
    "RunRecord",
    "Severity",
    "VALID_LAYERS",
    "VerifyReport",
    "__version__",
    "discover_verifiers",
    "git_commit",
    "hash_file",
    "read_meta",
    "utc_now_iso",
    "verify",
    "write_sidecars",
]
