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


# ── seeds: three jobs, three names (Phase 6.4) ─────────────────────────
def test_split_seed_and_train_seed_are_separate_constants():
    """They were one. Varying the training seed would have re-split the data,
    so the measured "run-to-run spread" would have included the split."""
    assert config.SPLIT_SEED is not None and config.TRAIN_SEED is not None
    assert "SEED" not in {n for n in dir(config) if n == "SEED"}


def test_train_seeds_include_the_default():
    """The sweep has to bracket the published run, not sit beside it."""
    assert config.TRAIN_SEED in config.TRAIN_SEEDS
    assert len(set(config.TRAIN_SEEDS)) >= 3


def test_default_seed_keeps_the_plain_output_dir():
    assert config.GPT2.seeded_output_dir(config.TRAIN_SEED) == config.GPT2.output_dir


def test_variance_run_gets_its_own_output_dir():
    """Otherwise the second training run overwrites the first."""
    seeded = config.GPT2.seeded_output_dir(43)
    assert seeded != config.GPT2.output_dir
    assert seeded.parent == config.OUTPUT_DIR and "43" in seeded.name


def test_variance_run_gets_its_own_metrics_key():
    assert config.GPT2.seeded_run_name(43) == "gpt2-lora-seed43"
    assert config.TINYLLAMA.seeded_run_name(43) == "tinyllama-qlora-seed43"
    assert config.GPT2.seeded_run_name(43, base=True) == "gpt2-base-seed43"


def test_arm_names_have_exactly_one_definition():
    """`load_arm` used to build this string itself. Two copies drifting apart
    would file one run in metrics.json under two keys, looking like two arms."""
    import inspect

    from medqa import models

    source = inspect.getsource(models.load_arm)
    assert "seeded_run_name" in source
    assert "qlora" not in source  # the naming rule lives on the spec, not here
