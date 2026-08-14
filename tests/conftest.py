"""Offline fixtures.

The suite must run without touching the network — CI has no HF token and a test
that downloads 500 MB is a test nobody runs. Anything that genuinely needs the
Hub is marked `@pytest.mark.network` and deselected by default.
"""

from __future__ import annotations

import datasets
import pytest
from datasets import Dataset

from medqa import config

datasets.disable_progress_bars()  # keeps `pytest -q` actually quiet


class StubTokenizer:
    """A whitespace tokenizer with a chat template — enough to exercise our code.

    Deliberately *not* a real tokenizer: these tests check our formatting and
    splitting logic, not HuggingFace's.
    """

    eos_token = "</s>"
    pad_token = "</s>"
    padding_side = "right"
    name_or_path = "stub"
    chat_template = "stub-template"

    def __call__(self, text, truncation=False, max_length=None, padding=None, **kwargs):
        # real tokenizers give EOS its own id, so don't let it glue to the last word
        ids = [len(w) for w in text.replace(self.eos_token, f" {self.eos_token} ").split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>\n{m['content']}</s>" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture
def real_tokenizer():
    """A genuine `PreTrainedTokenizerFast` built offline from a toy vocabulary.

    The collator test must exercise HuggingFace's real padding and label-masking
    code — a stub with a hand-written `pad()` would only test the stub. A
    WordLevel tokenizer gives us the real class with no network call.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<pad>", "<unk>", "</s>", "what", "is", "anemia", "low", "iron", "the", "flu"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )


@pytest.fixture
def raw_dataset() -> Dataset:
    """Shaped like MedQuAD, including an answer that would break naive splitting."""
    return Dataset.from_dict(
        {
            config.QUESTION_COL: [
                "  What is Tourette syndrome?  ",
                "What causes anemia?",
                "How is asthma treated?",
                "What are the symptoms of flu?",
            ],
            config.ANSWER_COL: [
                "A neurological disorder.",
                # contains the response marker — the notebook's split() would mangle it
                f"Low iron.\n{config.RESPONSE_MARKER}More detail here.",
                "Inhalers and avoidance of triggers.",
                "Fever, cough, and fatigue.  ",
            ],
            "qtype": ["information", "causes", "treatment", "symptoms"],
        }
    )
