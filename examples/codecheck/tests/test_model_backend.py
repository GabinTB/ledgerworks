"""The sentence-transformers backend, on a tiny randomly initialised ModernBERT.

No network: the model is built here from a config and a tokenizer trained on the
fixture text. Skipped when the ``model`` extra is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ledgerworks import Ledger
from ledgerworks.jobs import JobRunner

from codecheck_fomc.corpus import IngestJob
from codecheck_fomc.embed import EmbedJob
from codecheck_fomc.score import ScoreJob, Taxonomy
from tests.conftest import STATEMENTS

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

TAXONOMY = Path(__file__).resolve().parent.parent / "taxonomy" / "monetary_policy.yaml"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("model") / "tiny-modernbert"
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    corpus = [p for ps in STATEMENTS.values() for p in ps]
    tk.train_from_iterator(
        corpus, trainers.WordPieceTrainer(vocab_size=600, special_tokens=specials)
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        model_max_length=512,
    )
    cfg = ModernBertConfig(
        vocab_size=tok.vocab_size + len(specials),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=8192,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.cls_token_id,
        eos_token_id=tok.sep_token_id,
        cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id,
    )
    torch.manual_seed(0)
    hf = root / "hf"
    ModernBertModel(cfg).save_pretrained(hf)
    tok.save_pretrained(hf)
    st = SentenceTransformer(
        modules=[Transformer(str(hf)), Pooling(64, pooling_mode="mean")], device="cpu"
    )
    st.save(str(root))
    return root


def test_backend_encodes_unit_vectors_deterministically(tiny_model):
    from codecheck_fomc.backends import SentenceTransformerBackend

    b = SentenceTransformerBackend(model_id=str(tiny_model), revision=None, max_seq_length=128)
    texts = ["Inflation remains elevated.", "The Committee decided to lower the target range."]
    v1, v2 = b.encode(texts), b.encode(texts)
    assert v1.shape == (2, 64) and v1.dtype == np.float32
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v1, v2)
    card = b.model_card()
    assert card.dim == 64 and card.pooling == "mean" and card.backend == "sentence-transformers"
    assert card.serving["dtype"] == "float32" and "torch" in card.serving
    b.check_unchanged()


def test_pipeline_with_the_model_backend_and_resume(raw_dir, tiny_model, tmp_path):
    from codecheck_fomc.backends import SentenceTransformerBackend

    backend = SentenceTransformerBackend(
        model_id=str(tiny_model), revision=None, max_seq_length=128
    )
    with Ledger(tmp_path / "ledger") as ledger:
        runner = JobRunner(ledger, allow_dirty=True, log_file=False, handle_signals=False)
        statements = runner.start(IngestJob(raw_dir))
        embeddings = runner.start(EmbedJob(statements, backend))
        assert embeddings.meta.hyperparams["backend"] == "finmodernbert"
        assert embeddings.meta.hyperparams["model_id"] == str(tiny_model)
        assert embeddings.meta.model_card.dim == 64

        # the scorer rebuilds the backend from the embeddings artifact alone
        rebuilt = EmbedJob.from_artifact(embeddings, ledger).backend
        assert rebuilt.identity() == backend.identity()
        scores = runner.start(ScoreJob(embeddings, Taxonomy.load(TAXONOMY), rebuilt))
        assert scores.meta.model_card.model_id == str(tiny_model)
        assert len(scores.files()) == 3  # 2023, 2024, narratives
