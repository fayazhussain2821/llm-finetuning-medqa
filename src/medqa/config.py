"""Every constant, one place.

The notebooks scattered these across cells, which is why `MAX_LEN` had to be
re-declared in cell 17 and `HF_USER` drifted between notebooks. Import from here
instead of retyping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── paths: one codebase, two environments ──────────────────────────────
IN_COLAB = "COLAB_GPU" in os.environ or Path("/content").exists()

if IN_COLAB:
    ROOT = Path("/content/llm-finetuning-medqa")
else:
    ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
GENERATIONS_DIR = OUTPUT_DIR / "generations"

# Generation is ~1000x slower per row than a scoring forward pass, so answer
# quality is measured on a prefix of the held-out set rather than all of it.
# Fixed order, so every arm is asked the same questions.
QUALITY_N = 200
QUALITY_MAX_NEW_TOKENS = 200

# HF_USER is read from the environment, never hardcoded — otherwise anyone who
# clones this repo silently loads *Fayaz's* adapters from the Hub.
HF_USER = os.environ.get("HF_USER", "Babblu2821")

# ── data ───────────────────────────────────────────────────────────────
DATASET_ID = "keivalya/MedQuad-MedicalQnADataset"
QUESTION_COL = "Question"
ANSWER_COL = "Answer"

SEED = 42
TEST_SIZE = 0.1

# Chosen by measuring, not guessing: the 95th percentile of prompt+answer token
# length sits just under GPT-2's 1024 ceiling. See `data.probe_token_lengths`.
MAX_LEN = 1024

PROMPT_TEMPLATE = "### Instruction:\n{q}\n\n### Response:\n"
RESPONSE_MARKER = "### Response:\n"
INSTRUCTION_MARKER = "### Instruction:\n"

# ── LoRA ───────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ── training ───────────────────────────────────────────────────────────
LEARNING_RATE = 2e-4  # LoRA tolerates a higher LR than full fine-tuning
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 25
EVAL_STEPS = 100  # evaluate *during* training so overfitting is visible in time


@dataclass(frozen=True)
class ModelSpec:
    """Everything that differs between the two arms of the comparison."""

    key: str
    base_id: str
    adapter_name: str
    #  "instruct" -> ### Instruction/### Response;  "chat" -> tokenizer chat template
    prompt_style: str
    target_modules: list[str]
    per_device_batch_size: int
    gradient_accumulation_steps: int
    load_in_4bit: bool = False
    gradient_checkpointing: bool = False
    # QLoRA trains fp32 adapters with no grad scaler, so mixed precision is off
    # for TinyLlama — enabling it collides with the paged 8-bit optimiser.
    mixed_precision: bool = True
    optim: str = "adamw_torch"
    warmup_ratio: float = 0.03
    extra: dict = field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def hub_repo(self) -> str:
        return f"{HF_USER}/{self.adapter_name}"

    @property
    def output_dir(self) -> Path:
        return OUTPUT_DIR / self.adapter_name


GPT2 = ModelSpec(
    key="gpt2",
    base_id="gpt2",
    adapter_name="gpt2-medqa-lora",
    prompt_style="instruct",
    target_modules=["c_attn", "c_proj"],  # GPT-2's fused projections
    per_device_batch_size=8,
    gradient_accumulation_steps=2,  # effective batch = 16
)

TINYLLAMA = ModelSpec(
    key="tinyllama",
    base_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_name="tinyllama-medqa-qlora",
    prompt_style="chat",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    per_device_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16 — matched to GPT-2
    load_in_4bit=True,
    gradient_checkpointing=True,
    mixed_precision=False,
    optim="paged_adamw_8bit",
    warmup_ratio=0.0,
    extra={"warmup_steps": 30},
)

MODEL_SPECS: dict[str, ModelSpec] = {spec.key: spec for spec in (GPT2, TINYLLAMA)}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(MODEL_SPECS)}") from None
