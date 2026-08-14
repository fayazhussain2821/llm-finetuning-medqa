from datasets import Dataset

from medqa import config, data


def test_format_examples_strips_and_templates(raw_dataset):
    formatted = data.format_examples(raw_dataset)

    assert set(formatted.column_names) == {"question", "answer", "prompt", "text"}
    first = formatted[0]
    assert first["question"] == "What is Tourette syndrome?"  # whitespace stripped
    assert first["prompt"] == config.PROMPT_TEMPLATE.format(q=first["question"])
    assert first["text"] == first["prompt"] + first["answer"]
    assert formatted[3]["answer"].endswith("fatigue.")  # trailing spaces gone


def test_chat_formatting_does_not_string_surgery_the_prompt(raw_dataset, tokenizer):
    """The notebook recovered the answer with `text.split("### Response:\\n")[-1]`.

    Row 1's answer *contains* that marker, so the notebook would have dropped
    everything before it. Formatting from the raw column keeps it whole.
    """
    formatted = data.format_examples(raw_dataset)
    chat = data.to_chat_text(formatted, tokenizer)

    assert chat.column_names == ["text"]
    assert "Low iron." in chat[1]["text"]
    assert "More detail here." in chat[1]["text"]
    assert chat[0]["text"].startswith("<|user|>")


def test_split_is_reproducible(raw_dataset):
    formatted = data.format_examples(raw_dataset)
    first = data.split_dataset(formatted)["test"]["question"]
    second = data.split_dataset(formatted)["test"]["question"]
    assert first == second


def test_split_actually_uses_the_seed():
    """Guards against `seed` being accepted and then ignored."""
    wide = Dataset.from_dict(
        {
            config.QUESTION_COL: [f"q{i}" for i in range(200)],
            config.ANSWER_COL: [f"a{i}" for i in range(200)],
        }
    )
    formatted = data.format_examples(wide)
    a = data.split_dataset(formatted, seed=1)["test"]["question"]
    b = data.split_dataset(formatted, seed=999)["test"]["question"]
    assert a != b


def test_split_holds_out_the_configured_fraction():
    wide = Dataset.from_dict(
        {
            config.QUESTION_COL: [f"q{i}" for i in range(200)],
            config.ANSWER_COL: [f"a{i}" for i in range(200)],
        }
    )
    splits = data.split_dataset(data.format_examples(wide))
    assert len(splits["test"]) == int(200 * config.TEST_SIZE)
    assert len(splits["train"]) + len(splits["test"]) == 200


def test_tokenize_appends_eos_for_instruct_style(raw_dataset, tokenizer):
    formatted = data.format_examples(raw_dataset)

    with_eos = data.tokenize(formatted, tokenizer, append_eos=True)
    without = data.tokenize(formatted, tokenizer, append_eos=False)

    # the stub tokenizer emits one id per whitespace token
    assert len(with_eos[0]["input_ids"]) == len(without[0]["input_ids"]) + 1


def test_tokenize_drops_text_columns_and_builds_no_labels(raw_dataset, tokenizer):
    """Labels are the collator's job — pre-building them breaks pad masking."""
    tokenized = data.tokenize(data.format_examples(raw_dataset), tokenizer)
    assert set(tokenized.column_names) == {"input_ids", "attention_mask"}


def test_tokenize_truncates_at_max_len(raw_dataset, tokenizer):
    tokenized = data.tokenize(data.format_examples(raw_dataset), tokenizer, max_len=3)
    assert all(len(row["input_ids"]) <= 3 for row in tokenized)


def test_probe_token_lengths_reports_percentiles(raw_dataset, tokenizer):
    stats = data.probe_token_lengths(raw_dataset, tokenizer, n=4)
    assert {"p50", "p90", "p95", "p99", "max", "pct_truncated"} <= set(stats)
    assert stats["max"] >= stats["p50"]
    assert 0.0 <= stats["pct_truncated"] <= 100.0
