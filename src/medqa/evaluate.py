"""Held-out evaluation → `outputs/metrics.json`. Never retype a number a program produced.

    python -m medqa.evaluate --model gpt2                  # fine-tuned adapter
    python -m medqa.evaluate --model tinyllama --base      # control: no fine-tuning
    python -m medqa.evaluate --table                       # read metrics.json, print

Cell 30 of `Training.ipynb` hardcoded ``gpt2_ppl, tiny_ppl = 5.99, 2.80``. Change a
hyperparameter, re-run, and it reports the old numbers with total confidence. Every
number here is computed and persisted; the comparison table only ever reads the file.

**Why bits-per-byte and not just perplexity.** Perplexity is per *token*, and the two
models use different tokenizers — GPT-2 byte-level BPE over 50257 pieces, TinyLlama
SentencePiece over 32000. The denominators are not the same unit, so "5.99 → 2.80"
compares two quantities that were never on one scale. Bits-per-byte normalises by
UTF-8 bytes of the *same* reference answers, which is identical across tokenizers.

Scoring is restricted to the answer span. The prompt differs by arm (instruction
template vs chat template); including it would let the template, not the model,
move the metric.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel

from medqa import config, data, models

LN2 = math.log(2.0)


def _prompt_and_full_text(ex: dict, spec: config.ModelSpec, tokenizer) -> tuple[str, str]:
    """The prompt, and the prompt+answer it must be a prefix of."""
    if spec.prompt_style == "chat":
        user = [{"role": "user", "content": ex["question"]}]
        both = user + [{"role": "assistant", "content": ex["answer"]}]
        prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        return prompt, tokenizer.apply_chat_template(both, tokenize=False)
    prompt = config.PROMPT_TEMPLATE.format(q=ex["question"])
    return prompt, prompt + ex["answer"]


def _prompt_token_count(prompt_ids: list[int], full_ids: list[int]) -> int:
    """Length of the shared prefix — templates occasionally retokenize at the seam."""
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    n = 0
    for a, b in zip(prompt_ids, full_ids, strict=False):  # stop at the shorter
        if a != b:
            break
        n += 1
    return n


def context_window(model, default: int = config.MAX_LEN) -> int:
    """The model's real position limit, so truncation never overruns it.

    MAX_LEN is tuned to GPT-2's 1024; feeding that to a model with a shorter
    window indexes past its position embeddings and raises deep inside the
    forward pass rather than saying what went wrong.
    """
    cfg = getattr(model, "config", None)
    limit = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
    return min(default, limit) if limit else default


@torch.no_grad()
def score_dataset(model, tokenizer, dataset, spec: config.ModelSpec) -> dict[str, float]:
    """Sum answer-token NLL and answer bytes over the held-out set."""
    device = next(model.parameters()).device
    max_len = context_window(model)
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    skipped = 0

    for ex in dataset:
        prompt_text, full_text = _prompt_and_full_text(ex, spec, tokenizer)
        full_ids = tokenizer(full_text, truncation=True, max_length=max_len)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        n_prompt = _prompt_token_count(prompt_ids, full_ids)

        # Truncation can leave no answer tokens at all; those rows carry no signal.
        # n_prompt == 0 would leave nothing to condition the first prediction on.
        if n_prompt < 1 or len(full_ids) - n_prompt < 1:
            skipped += 1
            continue

        ids = torch.tensor([full_ids], device=device)
        logits = model(input_ids=ids).logits.float()

        # position i predicts token i+1 -> answer targets start at n_prompt
        target = ids[:, n_prompt:]
        pred_logits = logits[:, n_prompt - 1 : -1, :]
        nll = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)), target.reshape(-1), reduction="sum"
        )

        total_nll += nll.item()
        total_tokens += target.numel()
        total_bytes += len(
            tokenizer.decode(full_ids[n_prompt:], skip_special_tokens=True).encode("utf-8")
        )

    if total_tokens == 0:
        raise RuntimeError("no answer tokens scored — check the prompt/answer split")

    mean_nll = total_nll / total_tokens
    return {
        "answer_nll": mean_nll,
        # per-token, tokenizer-dependent: comparable only within one architecture
        "perplexity": math.exp(mean_nll),
        # per-byte, tokenizer-independent: THIS is the cross-model number
        "bits_per_byte": total_nll / LN2 / total_bytes,
        "n_examples": len(dataset) - skipped,
        "n_tokens": total_tokens,
        "n_bytes": total_bytes,
        "n_skipped": skipped,
    }


def evaluate_model(
    model_key: str,
    base: bool = False,
    adapter: str | Path | None = None,
    limit: int | None = None,
) -> tuple[str, dict]:
    """Score one arm. `base=True` gives the untrained control condition."""
    spec = config.get_spec(model_key)
    tokenizer = data.get_tokenizer(spec.base_id)

    formatted = data.format_examples(data.load_medquad())
    eval_ds = data.split_dataset(formatted)["test"]
    if limit:
        eval_ds = eval_ds.select(range(min(limit, len(eval_ds))))

    if base:
        run_name, model = f"{spec.key}-base", models.load_base_model(spec)
    else:
        if adapter is None:
            # prefer a locally trained adapter; fall back to the published one
            adapter = spec.output_dir if spec.output_dir.exists() else spec.hub_repo
        adapter = str(adapter)
        run_name = f"{spec.key}-{'qlora' if spec.load_in_4bit else 'lora'}"
        model = PeftModel.from_pretrained(models.load_base_model(spec), adapter)
        model.eval()

    metrics = score_dataset(model, tokenizer, eval_ds, spec)
    metrics.update({"model": spec.key, "base_id": spec.base_id, "fine_tuned": not base})
    return run_name, metrics


def write_metrics(run_name: str, metrics: dict, path: Path = config.METRICS_PATH) -> None:
    """Merge into metrics.json so separate runs accumulate instead of clobbering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[run_name] = metrics
    path.write_text(json.dumps(existing, indent=2))
    print(f"wrote {run_name} → {path}")


def comparison_table(path: Path = config.METRICS_PATH) -> str:
    """Render whatever has actually been measured. No number is typed in here."""
    if not path.exists():
        return f"no metrics yet — run `python -m medqa.evaluate --model gpt2` first ({path})"
    runs = json.loads(path.read_text())

    header = f"{'run':<20}{'bits/byte':>12}{'perplexity':>13}{'n':>8}"
    lines = [header, "-" * len(header)]
    for name, m in sorted(runs.items()):
        lines.append(
            f"{name:<20}{m['bits_per_byte']:>12.4f}{m['perplexity']:>13.2f}{m['n_examples']:>8}"
        )

    lines.append("")
    for key in sorted({m["model"] for m in runs.values()}):
        ft, bs = runs.get(f"{key}-lora") or runs.get(f"{key}-qlora"), runs.get(f"{key}-base")
        if ft and bs:
            delta = (bs["bits_per_byte"] - ft["bits_per_byte"]) / bs["bits_per_byte"] * 100
            lines.append(f"{key}: fine-tuning cut bits/byte by {delta:.1f}% vs the same base model")
        elif ft:
            lines.append(f"{key}: no `--base` control measured yet — improvement is unattributable")

    lines.append(
        "\nCompare bits/byte across models; perplexity only within one tokenizer."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(config.MODEL_SPECS))
    parser.add_argument("--base", action="store_true", help="score the base model (control)")
    parser.add_argument("--adapter", help="local dir or Hub repo id; defaults to outputs/ then Hub")
    parser.add_argument("--limit", type=int, help="score only N eval rows (smoke test)")
    parser.add_argument("--table", action="store_true", help="print metrics.json and exit")
    args = parser.parse_args(argv)

    if args.model:
        run_name, metrics = evaluate_model(
            args.model, base=args.base, adapter=args.adapter, limit=args.limit
        )
        write_metrics(run_name, metrics)
    elif not args.table:
        parser.error("pass --model to measure, or --table to report")

    print(comparison_table())


if __name__ == "__main__":
    main()
