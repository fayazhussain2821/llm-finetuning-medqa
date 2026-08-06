import pytest

from medqa import config, models


def test_greedy_preset_omits_sampling_params():
    """transformers warns when temperature/top_p are set but do_sample is False."""
    kwargs = models.GREEDY.to_kwargs()
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_sampled_preset_keeps_sampling_params():
    kwargs = models.SAMPLED.to_kwargs()
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9


def test_metric_bearing_presets_are_deterministic():
    """Anything compared across models must not sample. The demo included —
    that drift between the two notebook copies is exactly what this pins."""
    assert models.GREEDY.do_sample is False
    assert models.DEMO.do_sample is False


def test_all_presets_penalise_repetition():
    """Both small models loop without it; the notebooks agreed on 1.15."""
    assert all(p.repetition_penalty == 1.15 for p in models.PRESETS.values())


def test_lora_config_targets_the_right_modules():
    assert models.lora_config(config.GPT2).target_modules == {"c_attn", "c_proj"}
    assert models.lora_config(config.TINYLLAMA).target_modules == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }


def test_lora_hyperparams_shared_across_arms():
    """r and alpha must match, or the comparison confounds rank with method."""
    gpt2_cfg, tiny_cfg = models.lora_config(config.GPT2), models.lora_config(config.TINYLLAMA)
    assert (gpt2_cfg.r, gpt2_cfg.lora_alpha) == (tiny_cfg.r, tiny_cfg.lora_alpha)


def test_best_device_is_a_known_backend():
    assert models.best_device() in {"cuda", "mps", "cpu"}


def test_cpu_stays_fp32():
    """fp16 on CPU is slower than fp32 and numerically worse."""
    import torch

    assert models.default_dtype("cpu") is torch.float32


@pytest.mark.network
def test_domain_chat_model_builds_instruct_prompt():
    """Prompt construction against a real tokenizer. Needs the Hub."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    obj = models.DomainChatModel.__new__(models.DomainChatModel)
    obj.spec, obj.tokenizer = config.GPT2, tok
    assert obj.build_prompt("  Why?  ") == config.PROMPT_TEMPLATE.format(q="Why?")
