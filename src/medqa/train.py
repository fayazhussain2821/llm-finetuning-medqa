"""One CLI, both arms of the comparison.

    python -m medqa.train --model gpt2
    python -m medqa.train --model tinyllama

What the notebooks lacked and this adds:

* `set_seed(42)` across torch, numpy and random — not just the dataset split.
* Evaluation *during* training (`eval_strategy="steps"`), so an overfit is visible
  while there is still time to stop. The notebooks only evaluated at the end.
* `load_best_model_at_end=True`.
* The resolved config is written next to the adapter, so a checkpoint records how
  it was made.

**One deliberate change from the notebooks:** GPT-2 trained under `Trainer` and
TinyLlama under TRL's `SFTTrainer`. Since both arms are pre-tokenized by
`medqa.data`, `SFTTrainer` was only wrapping the same loop — and a comparison
whose two arms run different training machinery has a confound it does not need.
Both arms now use `Trainer` with the same collator. Everything that actually
differs (4-bit, optimiser, batch shape) lives in the `ModelSpec`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from medqa import config, data, models


def _precision_flags(spec: config.ModelSpec) -> dict[str, bool]:
    """fp16/bf16 only on CUDA, and never for the QLoRA arm.

    QLoRA keeps its adapters in fp32 and trains with a paged 8-bit optimiser and
    no grad scaler; turning on mixed precision here is what produced the dtype
    crash the notebook comment refers to.
    """
    if not spec.mixed_precision or not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    bf16 = torch.cuda.is_bf16_supported()  # False on a T4
    return {"fp16": not bf16, "bf16": bf16}


def build_training_args(
    spec: config.ModelSpec, epochs: float, output_dir: Path
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=spec.per_device_batch_size,
        per_device_eval_batch_size=spec.per_device_batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=spec.warmup_ratio,
        warmup_steps=spec.extra.get("warmup_steps", 0),
        logging_steps=config.LOGGING_STEPS,
        # eval + save on the same cadence — load_best_model_at_end requires it
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=spec.gradient_checkpointing,
        optim=spec.optim,
        report_to="none",
        seed=config.SEED,
        **_precision_flags(spec),
    )


def save_run_config(
    spec: config.ModelSpec, args: TrainingArguments, metrics: dict, dest: Path
) -> None:
    """A checkpoint that cannot tell you how it was made is a checkpoint you cannot trust."""
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": {**asdict(spec), "effective_batch_size": spec.effective_batch_size},
        "lora": {"r": config.LORA_R, "alpha": config.LORA_ALPHA, "dropout": config.LORA_DROPOUT},
        "data": {"dataset": config.DATASET_ID, "max_len": config.MAX_LEN, "seed": config.SEED},
        "training_args": args.to_dict(),
        "metrics": metrics,
    }
    (dest / "run_config.json").write_text(json.dumps(payload, indent=2, default=str))


def train(model_key: str, epochs: float = 1.0, push_to_hub: bool = False) -> dict:
    spec = config.get_spec(model_key)
    set_seed(config.SEED)  # torch + numpy + random, not just the split

    tokenizer = data.get_tokenizer(spec.base_id)
    train_tok, eval_tok = data.build_splits(spec, tokenizer)
    print(f"[{spec.key}] train={len(train_tok)}  eval={len(eval_tok)}")

    model = models.load_for_training(spec, tokenizer)
    model.print_trainable_parameters()

    output_dir = spec.output_dir
    args = build_training_args(spec, epochs, output_dir)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data.get_collator(tokenizer),
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    metrics = {"eval_loss": eval_loss, "perplexity": math.exp(eval_loss)}
    print(f"[{spec.key}] eval loss {eval_loss:.4f}  |  perplexity {metrics['perplexity']:.2f}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_run_config(spec, args, metrics, output_dir)
    print(f"[{spec.key}] adapter saved to {output_dir}")

    if push_to_hub:
        trainer.model.push_to_hub(spec.hub_repo)
        tokenizer.push_to_hub(spec.hub_repo)
        print(f"[{spec.key}] pushed to {spec.hub_repo}")

    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(config.MODEL_SPECS))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help=f"upload the adapter to {config.HF_USER}/… (needs a write token)",
    )
    args = parser.parse_args(argv)
    train(args.model, epochs=args.epochs, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
