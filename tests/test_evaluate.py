import json

from medqa import config, evaluate


def test_prompt_token_count_exact_prefix():
    assert evaluate._prompt_token_count([1, 2, 3], [1, 2, 3, 4, 5]) == 3


def test_prompt_token_count_handles_retokenised_seam():
    """Chat templates sometimes merge the last prompt token with the first answer
    token, so the prompt is not a literal prefix. Fall back to the shared prefix."""
    assert evaluate._prompt_token_count([1, 2, 9], [1, 2, 7, 8]) == 2


def test_prompt_token_count_no_overlap():
    assert evaluate._prompt_token_count([5], [6, 7]) == 0


def test_prompt_and_full_text_instruct(tokenizer):
    ex = {"question": "What is anemia?", "answer": "Low iron."}
    prompt, full = evaluate._prompt_and_full_text(ex, config.GPT2, tokenizer)
    assert prompt == config.PROMPT_TEMPLATE.format(q="What is anemia?")
    assert full == prompt + "Low iron."
    assert full.startswith(prompt)


def test_prompt_and_full_text_chat_is_a_prefix(tokenizer):
    """The scoring split depends on the prompt being a prefix of the full text."""
    ex = {"question": "What is anemia?", "answer": "Low iron."}
    prompt, full = evaluate._prompt_and_full_text(ex, config.TINYLLAMA, tokenizer)
    assert "<|assistant|>" in prompt
    assert "Low iron." in full and "Low iron." not in prompt


class _FakeConfig:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeModel:
    def __init__(self, cfg=None):
        self.config = cfg


def test_context_window_clamps_to_a_short_model():
    """MAX_LEN is tuned to GPT-2's 1024; a shorter model must not be overrun."""
    model = _FakeModel(_FakeConfig(max_position_embeddings=512))
    assert evaluate.context_window(model) == 512


def test_context_window_never_exceeds_max_len():
    """TinyLlama's window is 2048, but MAX_LEN is the experiment's decision."""
    model = _FakeModel(_FakeConfig(max_position_embeddings=2048))
    assert evaluate.context_window(model) == config.MAX_LEN


def test_context_window_falls_back_to_n_positions():
    assert evaluate.context_window(_FakeModel(_FakeConfig(n_positions=256))) == 256


def test_context_window_defaults_when_model_declares_nothing():
    assert evaluate.context_window(_FakeModel(_FakeConfig())) == config.MAX_LEN


def test_write_metrics_merges_across_runs(tmp_path):
    """Evaluating TinyLlama must not erase the GPT-2 numbers."""
    path = tmp_path / "metrics.json"
    evaluate.write_metrics("gpt2-lora", {"bits_per_byte": 1.5}, path)
    evaluate.write_metrics("tinyllama-qlora", {"bits_per_byte": 1.1}, path)

    written = json.loads(path.read_text())
    assert set(written) == {"gpt2-lora", "tinyllama-qlora"}


def test_comparison_table_reads_the_file_not_hardcoded_numbers(tmp_path):
    """The bug this module exists to kill: cell 30's `gpt2_ppl, tiny_ppl = 5.99, 2.80`."""
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "gpt2-lora": {
                    "bits_per_byte": 1.2345,
                    "perplexity": 7.77,
                    "n_examples": 10,
                    "model": "gpt2",
                },
                "gpt2-base": {
                    "bits_per_byte": 2.4690,
                    "perplexity": 9.99,
                    "n_examples": 10,
                    "model": "gpt2",
                },
            }
        )
    )
    table = evaluate.comparison_table(path)

    assert "1.2345" in table and "7.77" in table
    assert "5.99" not in table and "2.80" not in table
    assert "50.0%" in table  # 2.4690 -> 1.2345 is exactly half


def test_comparison_table_flags_a_missing_control(tmp_path):
    """An improvement with no control condition is unattributable — say so."""
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "tinyllama-qlora": {
                    "bits_per_byte": 1.0,
                    "perplexity": 2.8,
                    "n_examples": 10,
                    "model": "tinyllama",
                }
            }
        )
    )
    assert "unattributable" in evaluate.comparison_table(path)


def test_comparison_table_without_metrics_file(tmp_path):
    assert "no metrics yet" in evaluate.comparison_table(tmp_path / "absent.json")


def test_metrics_path_lives_under_outputs():
    """outputs/ is gitignored — metrics must never land in a commit by accident."""
    assert config.METRICS_PATH.parent == config.OUTPUT_DIR
