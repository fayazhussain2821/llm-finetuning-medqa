"""Model loading and generation — the single copy of `DomainChatModel`.

The class existed in **two** notebooks and had already drifted: training used
sampling at 256 new tokens, the demo used greedy at 80. Same class name, different
behaviour, so the demo never showed what the evaluation measured.

The fix is not "pick one" — it is to make the choice explicit. Decoding is now a
named `GenerationPreset`, so evaluation and the demo can share settings or differ
*on purpose*, visibly, in one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from medqa import config


# ── decoding presets ───────────────────────────────────────────────────
@dataclass(frozen=True)
class GenerationPreset:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.15  # small models loop badly without this

    def to_kwargs(self) -> dict:
        kwargs = {k: v for k, v in asdict(self).items() if v is not None}
        if not self.do_sample:
            # transformers warns if temperature/top_p are set but unused
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        return kwargs


#: Deterministic. Use for anything compared across models or across runs.
GREEDY = GenerationPreset(max_new_tokens=256, do_sample=False)

#: Mild diversity. Use when showing off qualitative range, never for metrics.
SAMPLED = GenerationPreset(max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)

#: The demo answers interactively, so it trades length for latency — but stays
#: greedy, so what a visitor sees is what the evaluation measured.
DEMO = GenerationPreset(max_new_tokens=160, do_sample=False)

PRESETS: dict[str, GenerationPreset] = {"greedy": GREEDY, "sampled": SAMPLED, "demo": DEMO}


# ── device / dtype ─────────────────────────────────────────────────────
def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str | None = None) -> torch.dtype:
    """fp16 only where it is actually fast and safe; fp32 on CPU."""
    device = device or best_device()
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def quantization_config(compute_dtype: torch.dtype = torch.float16) -> BitsAndBytesConfig:
    """NF4 double-quantised 4-bit — the QLoRA recipe."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


# ── training-side loading ──────────────────────────────────────────────
def lora_config(spec: config.ModelSpec) -> LoraConfig:
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=spec.target_modules,
    )


def load_for_training(spec: config.ModelSpec, tokenizer):
    """Base model (optionally 4-bit) with a fresh LoRA adapter attached."""
    if spec.load_in_4bit:
        compute_dtype = torch.float16  # T4 is Turing (sm_75): no bf16
        model = AutoModelForCausalLM.from_pretrained(
            spec.base_id,
            quantization_config=quantization_config(compute_dtype),
            device_map="auto",
            dtype=compute_dtype,  # renamed from torch_dtype in transformers 5.x
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=spec.gradient_checkpointing
        )
        model.enable_input_require_grads()
    else:
        model = AutoModelForCausalLM.from_pretrained(spec.base_id)

    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, lora_config(spec))
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model


def load_base_model(spec: config.ModelSpec, device: str | None = None):
    """The *un*-fine-tuned base model — the control condition Phase 6.2 needs."""
    device = device or best_device()
    if spec.load_in_4bit and device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            spec.base_id,
            quantization_config=quantization_config(),
            device_map="auto",
            dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(spec.base_id, dtype=default_dtype(device))
        model = model.to(device)
    model.eval()
    return model


# ── inference ──────────────────────────────────────────────────────────
class DomainChatModel:
    """A base model plus a trained LoRA adapter, ready to answer questions.

    `adapter` may be a local directory or a Hub repo id — `from_pretrained`
    resolves both, so nothing here needs a Drive mount.
    """

    def __init__(
        self,
        spec: config.ModelSpec,
        adapter: str | Path | None = None,
        preset: GenerationPreset = GREEDY,
        device: str | None = None,
    ):
        self.spec = spec
        self.preset = preset
        self.device = device or best_device()
        adapter = str(adapter if adapter is not None else spec.hub_repo)

        self.tokenizer = AutoTokenizer.from_pretrained(adapter)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # generation: pad on the left

        # 4-bit needs CUDA; on a Mac or CPU box, fall back to full precision so
        # the demo still runs instead of raising deep inside bitsandbytes.
        use_4bit = spec.load_in_4bit and self.device == "cuda"
        if use_4bit:
            base = AutoModelForCausalLM.from_pretrained(
                spec.base_id,
                quantization_config=quantization_config(),
                device_map="auto",
                dtype=torch.float16,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                spec.base_id, dtype=default_dtype(self.device)
            ).to(self.device)

        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()
        self.model.config.use_cache = True  # KV cache back on for fast generation

    def build_prompt(self, question: str) -> str:
        question = question.strip()
        if self.spec.prompt_style == "chat":
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return config.PROMPT_TEMPLATE.format(q=question)

    @torch.no_grad()
    def generate(self, question: str, preset: GenerationPreset | None = None) -> str:
        preset = preset or self.preset
        inputs = self.tokenizer(self.build_prompt(question), return_tensors="pt").to(
            self.model.device
        )
        out = self.model.generate(
            **inputs,
            **preset.to_kwargs(),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]  # drop the echoed prompt
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
