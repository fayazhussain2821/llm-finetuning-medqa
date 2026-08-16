"""Load → format → split → tokenize.

Ported from `notebooks/original/Training.ipynb` cells 3–8 and 12.

One deliberate change from the notebook: the chat-template formatter no longer
recovers the question and answer by string-splitting a prompt it just built
(``ex["text"].split("### Response:\\n", 1)[-1]``). Any answer containing that
marker would have been silently mangled. The raw fields are carried through as
``question``/``answer`` columns instead, so both prompt styles format from the
same source of truth.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from medqa import config


def load_medquad(split: str = "train") -> Dataset:
    """Download MedQuAD from the Hub. No Drive mount, no local copy."""
    return load_dataset(config.DATASET_ID, split=split)


def format_examples(raw: Dataset) -> Dataset:
    """Attach the instruction-template `prompt`/`text` plus the raw q/a fields."""

    def _format(ex: dict) -> dict:
        question = ex[config.QUESTION_COL].strip()
        answer = ex[config.ANSWER_COL].strip()
        prompt = config.PROMPT_TEMPLATE.format(q=question)
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "text": prompt + answer,
        }

    return raw.map(_format, remove_columns=raw.column_names)


def split_dataset(dataset: Dataset, seed: int = config.SPLIT_SEED) -> DatasetDict:
    """Seeded split — the eval set must be identical across every run and model."""
    return dataset.train_test_split(test_size=config.TEST_SIZE, seed=seed)


def held_out(limit: int | None = None) -> Dataset:
    """The evaluation rows, in a fixed order, shared by every measurement.

    `limit` takes the first N — not a random sample — so a 200-row generation run
    and a 200-row rerun score the *same* questions and stay comparable.
    """
    evaluation = split_dataset(format_examples(load_medquad()))["test"]
    return evaluation.select(range(min(limit, len(evaluation)))) if limit else evaluation


def to_chat_text(dataset: Dataset, tokenizer) -> Dataset:
    """Re-render the same examples with the model's native chat template."""
    if tokenizer.chat_template is None:
        raise ValueError(f"{tokenizer.name_or_path} ships no chat template")

    def _to_chat(ex: dict) -> dict:
        msgs = [
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    return dataset.map(_to_chat, remove_columns=[c for c in dataset.column_names if c != "text"])


def get_tokenizer(model_id: str):
    """Load a tokenizer configured for *training* (right padding)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize(
    dataset: Dataset,
    tokenizer,
    max_len: int = config.MAX_LEN,
    append_eos: bool = True,
) -> Dataset:
    """Tokenize without building labels — the collator does that (see `get_collator`).

    `append_eos` is False for chat-formatted text, where `apply_chat_template`
    has already terminated each turn; appending again would train a double EOS.
    """

    def _tokenize(ex: dict) -> dict:
        text = ex["text"] + tokenizer.eos_token if append_eos else ex["text"]
        return tokenizer(text, truncation=True, max_length=max_len, padding=False)

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def get_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """Dynamic per-batch padding; `mlm=False` -> causal LM, labels masked at pads."""
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def probe_token_lengths(
    raw: Dataset,
    tokenizer=None,
    n: int = 2000,
    percentiles: tuple[int, ...] = (50, 90, 95, 99),
) -> dict[str, float]:
    """Measure prompt+answer token length so MAX_LEN stays a decision, not a guess.

    Returns the percentiles plus the fraction truncated at `config.MAX_LEN`.
    """
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.GPT2.base_id)
    sample = raw.select(range(min(n, len(raw))))
    lengths = np.array(
        [
            len(
                tokenizer(
                    config.PROMPT_TEMPLATE.format(q=r[config.QUESTION_COL]) + r[config.ANSWER_COL]
                )["input_ids"]
            )
            for r in sample
        ]
    )
    stats = {f"p{p}": float(np.percentile(lengths, p)) for p in percentiles}
    stats["max"] = float(lengths.max())
    stats["pct_truncated"] = float((lengths > config.MAX_LEN).mean() * 100)
    return stats


def build_splits(spec: config.ModelSpec, tokenizer) -> tuple[Dataset, Dataset]:
    """The whole pipeline for one model arm: raw → tokenized train/eval.

    Both arms split with the same seed *before* prompt formatting diverges, so
    GPT-2 and TinyLlama are evaluated on exactly the same held-out rows.
    """
    formatted = format_examples(load_medquad())
    splits = split_dataset(formatted)
    train, evaluation = splits["train"], splits["test"]

    if spec.prompt_style == "chat":
        train, evaluation = to_chat_text(train, tokenizer), to_chat_text(evaluation, tokenizer)
        append_eos = False
    else:
        append_eos = True

    return (
        tokenize(train, tokenizer, append_eos=append_eos),
        tokenize(evaluation, tokenizer, append_eos=append_eos),
    )
