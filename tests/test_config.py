import dataclasses
import importlib

import pytest

from medqa import config


def test_both_specs_registered():
    assert set(config.MODEL_SPECS) == {"gpt2", "tinyllama"}


def test_get_spec_rejects_unknown():
    with pytest.raises(KeyError, match="unknown model"):
        config.get_spec("llama-70b")


def test_effective_batch_size_is_matched_across_arms():
    """The comparison is only fair if both arms see the same effective batch.

    GPT-2 gets 8x2 and TinyLlama 4x4 — different physical shapes to fit VRAM,
    identical optimisation steps. If someone tunes one side, this fails loudly.
    """
    assert config.GPT2.effective_batch_size == config.TINYLLAMA.effective_batch_size == 16


def test_qlora_arm_disables_mixed_precision():
    """fp32 adapters + paged 8-bit optimiser + no grad scaler -> no fp16/bf16."""
    assert config.TINYLLAMA.load_in_4bit is True
    assert config.TINYLLAMA.mixed_precision is False
    assert config.GPT2.load_in_4bit is False


def test_max_len_within_gpt2_context_window():
    assert config.MAX_LEN <= 1024


def test_hub_repo_follows_hf_user_env(monkeypatch):
    """A clone must not silently load someone else's adapters."""
    monkeypatch.setenv("HF_USER", "someone-else")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.GPT2.hub_repo == "someone-else/gpt2-medqa-lora"
    finally:
        monkeypatch.delenv("HF_USER", raising=False)
        importlib.reload(config)


def test_specs_are_frozen():
    """A spec mutated at runtime would not match the run_config.json saved beside
    the adapter, which is the whole point of saving it."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.GPT2.per_device_batch_size = 99


def test_deliberately_broken_to_prove_ci_blocks():
    assert 1 == 2
