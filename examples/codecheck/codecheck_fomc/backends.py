"""Embedding backends: the real model, and a network-free one for tests and CI.

A backend exposes what ``ledgerworks.jobs`` expects of a model (``identity()`` and
``check_unchanged()``), a ``model_card()`` for the artifact record, and
``encode(texts)`` returning L2-normalised float32 vectors.
"""

from __future__ import annotations

import hashlib
import math
import platform
import re
from typing import Any, Protocol

import numpy as np
from ledgerworks import ModelCard

FINMODERNBERT_ID = "BatuhanECB/FinModernBERT-embed-large-v1"
# Pinned commit of the model repo on the Hugging Face Hub (main as of 2026-10-14).
FINMODERNBERT_REVISION = "4fd33d2335e75c79df57b345aee06d49c5f85bc8"
FINMODERNBERT_DIM = 1024
FINMODERNBERT_MAX_SEQ = 512  # the training length; the architecture allows 8192


class Backend(Protocol):
    name: str

    def identity(self) -> dict[str, Any]: ...
    def check_unchanged(self) -> None: ...
    def model_card(self) -> ModelCard: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Hashing backend: deterministic, stdlib + numpy, no weights
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class HashingBackend:
    """Signed feature hashing of lowercase word unigrams and bigrams.

    Not a language model: it exists so the whole pipeline runs offline in tests and
    CI, with outputs that are bit-for-bit identical on every machine.
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def identity(self) -> dict[str, Any]:
        return {"backend": self.name, "dim": self.dim, "version": "1"}

    def check_unchanged(self) -> None:
        return None

    def model_card(self) -> ModelCard:
        return ModelCard(
            model_id="hashing-unigram-bigram",
            version="1",
            repo="https://github.com/GabinTB/ledgerworks",
            weights_public=True,
            architecture="signed feature hashing (blake2b)",
            dim=self.dim,
            pooling="sum, then L2 normalised",
            notes="test backend: no learned weights",
            backend=self.name,
            serving={"numpy": np.__version__},
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            feats = tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]
            for feat in feats:
                h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
            if tokens:
                out[i] /= math.sqrt(len(feats))
        return normalise(out)


# ---------------------------------------------------------------------------
# Sentence-transformers backend: FinModernBERT-embed-large-v1
# ---------------------------------------------------------------------------


class SentenceTransformerBackend:
    """A sentence-transformers model, pinned to a Hub revision, run on CPU in fp32.

    ``threads`` caps intra-op parallelism; ``deterministic`` turns on
    ``torch.use_deterministic_algorithms``. Results are reproducible to floating
    point tolerance across machines, not bit for bit (BLAS kernels differ).
    """

    name = "sentence-transformers"

    def __init__(
        self,
        model_id: str = FINMODERNBERT_ID,
        revision: str | None = FINMODERNBERT_REVISION,
        *,
        max_seq_length: int = FINMODERNBERT_MAX_SEQ,
        batch_size: int = 16,
        threads: int | None = None,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if threads:
            torch.set_num_threads(threads)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)
        self.model_id, self.revision = model_id, revision
        self.max_seq_length, self.batch_size, self.device = max_seq_length, batch_size, device
        self.model = SentenceTransformer(model_id, revision=revision, device=device)
        self.model.max_seq_length = max_seq_length
        self.model.eval()
        self._torch = torch
        self._recorded = self.identity()

    def _dim(self) -> int:
        get = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        return int(get() or 0)

    def identity(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "max_seq_length": self.max_seq_length,
            "dim": self._dim(),
        }

    def check_unchanged(self) -> None:
        now = self.identity()
        if now != self._recorded:
            raise RuntimeError(f"model identity changed: {self._recorded} -> {now}")

    def model_card(self) -> ModelCard:
        import sentence_transformers
        import transformers

        pooling = None
        for module in self.model:
            mode = getattr(module, "pooling_mode", None)  # the Pooling module
            if isinstance(mode, str):
                pooling = mode
                break
        return ModelCard(
            model_id=self.model_id,
            version=(self.revision or "local")[:12],
            repo=f"https://huggingface.co/{self.model_id}",
            weights_public=True,
            commit=self.revision,
            architecture="ModernBERT-large, contrastively fine-tuned (sentence-transformers)",
            dim=self._dim(),
            pooling=pooling,
            trained_on="FinModernBERT-large-DAPT + SEC-synthetic pairs (see model card)",
            notes="Apache-2.0 weights; author-reported FinMTEB scores on the Hub card",
            backend=self.name,
            serving={
                "device": self.device,
                "dtype": "float32",
                "max_seq_length": self.max_seq_length,
                "batch_size": self.batch_size,
                "torch": self._torch.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "threads": self._torch.get_num_threads(),
            },
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._torch.inference_mode():
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        return normalise(vectors)


BACKENDS = {"hashing": HashingBackend, "finmodernbert": SentenceTransformerBackend}


def make_backend(name: str, **kwargs: Any) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}, expected one of {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
