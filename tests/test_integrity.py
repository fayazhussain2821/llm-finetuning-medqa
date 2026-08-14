"""The things that break silently and invalidate every number downstream.

Not coverage-chasing. Each test here corresponds to a failure mode that produces
*plausible-looking* results rather than an error — which is the only kind of bug
that survives to the README.
"""

from datasets import Dataset

from medqa import config, data, evaluate, models


def _wide_dataset(n: int = 300) -> Dataset:
    return Dataset.from_dict(
        {
            config.QUESTION_COL: [f"question number {i}" for i in range(n)],
            config.ANSWER_COL: [f"answer number {i}" for i in range(n)],
        }
    )


def test_collator_masks_every_pad_position(real_tokenizer):
    """`labels == -100` must equal the pad count exactly.

    If padding were ever counted as a real token, the model would be trained and
    scored on `<pad>`, quietly deflating loss on short answers. Cell 8 of the
    notebook checked this by eye, once. This checks it on every run.
    """
    collator = data.get_collator(real_tokenizer)
    batch = collator(
        [
            {"input_ids": [3, 4, 5, 6], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [3, 4], "attention_mask": [1, 1]},
            {"input_ids": [7], "attention_mask": [1]},
        ]
    )

    n_pad = (batch["attention_mask"] == 0).sum().item()
    n_masked = (batch["labels"] == -100).sum().item()
    assert n_pad == 5  # (4-4) + (4-2) + (4-1)
    assert n_masked == n_pad


def test_collator_builds_labels_from_input_ids(real_tokenizer):
    """Causal LM, not masked LM: unpadded labels are a copy of the inputs."""
    batch = data.get_collator(real_tokenizer)([{"input_ids": [3, 4, 5], "attention_mask": [1] * 3}])
    assert batch["labels"].tolist() == [[3, 4, 5]]


def test_no_train_eval_overlap():
    """Leakage is the worst failure mode here: perplexity on a leaked eval set
    looks *great*, and nothing in the output would tell you."""
    splits = data.split_dataset(data.format_examples(_wide_dataset()))
    train_qs = set(splits["train"]["question"])
    eval_qs = set(splits["test"]["question"])

    assert train_qs.isdisjoint(eval_qs)
    assert len(train_qs) + len(eval_qs) == 300


def test_eval_set_is_identical_across_model_arms():
    """GPT-2 and TinyLlama must be scored on the same rows, or the comparison
    measures the split as much as the model."""
    formatted = data.format_examples(_wide_dataset())
    first = data.split_dataset(formatted)["test"]["question"]
    second = data.split_dataset(formatted)["test"]["question"]
    assert first == second


def test_generation_and_scoring_ask_the_same_question(tokenizer):
    """The likelihood metric scores the span after a prompt; the quality metric
    generates from one. If those two prompts ever drift apart, the two numbers
    describe different experiments and neither table would say so."""
    ex = {"question": "  What is anemia?  ", "answer": "Low iron."}
    for spec in (config.GPT2, config.TINYLLAMA):
        scored_prompt, _ = evaluate._prompt_and_full_text(ex, spec, tokenizer)
        assert models.build_prompt(ex["question"], spec, tokenizer) == scored_prompt


def test_config_paths_resolve_locally():
    """The Colab/local shim has two branches and CI only ever exercises one."""
    assert config.ROOT.exists()
    assert (config.ROOT / "pyproject.toml").exists()
    assert config.OUTPUT_DIR.parent == config.ROOT
    assert config.METRICS_PATH.parent == config.OUTPUT_DIR


def test_colab_branch_points_at_the_clone_target():
    """The bootstrap cell clones to /content/llm-finetuning-medqa — if these two
    ever disagree, every path in Colab silently resolves to the wrong place."""
    import inspect

    source = inspect.getsource(config)
    assert '"/content/llm-finetuning-medqa"' in source


def test_output_dirs_are_gitignored():
    """Adapters and metrics must never reach a commit."""
    ignored = (config.ROOT / ".gitignore").read_text()
    assert "outputs/" in ignored
    for spec in config.MODEL_SPECS.values():
        assert spec.output_dir.parent == config.OUTPUT_DIR
