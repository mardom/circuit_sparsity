from pathlib import Path

import pytest
import torch

MODEL_SUBDIR = (
    "achyuta-csp-achyuta_multik_yolo_exp-_newtok_12L_btableb_"
    "spembed_spunembed_dhead8c_xent_multiK_16B_nwise_annealing__"
    "lr1.28e-2_4x_pfrac6.25e-2"
)


def test_gpt_forward_pass(monkeypatch):
    pytest.importorskip("mpi4py")
    monkeypatch.setenv("NO_COMMS", "1")

    from circuit_sparsity.inference.gpt import GPT, GPTConfig

    config = GPTConfig(
        block_size=8,
        vocab_size=16,
        n_layer=1,
        n_head=1,
        d_model=8,
        dropout=0.0,
        flash=False,
        grad_checkpointing=False,
    )
    model = GPT(config)

    batch_size = 2
    seq_len = 4
    idx = torch.randint(config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    logits, loss, hidden_states = model(idx, targets=targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss.shape == torch.Size([])
    assert len(hidden_states) == config.n_layer


def test_pretrained_gpt_forward_pass(monkeypatch):
    pytest.importorskip("mpi4py")
    monkeypatch.setenv("NO_COMMS", "1")

    from circuit_sparsity.inference.gpt import load_model

    model_dir = (
        Path(__file__).resolve().parent.parent.parent  # repo root
        / "data"
        / "models"
        / MODEL_SUBDIR
    )

    assert (model_dir / "final_model.pt").is_file()

    model = load_model(str(model_dir), cuda=False)

    batch_size = 1
    seq_len = 8
    vocab_size = model.config.vocab_size

    idx = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        logits, loss, hidden_states = model(idx, targets=targets)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert len(hidden_states) == model.config.n_layer
