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
from datetime import date
from pathlib import Path

import torch

from medqa import config, data, models

LN2 = math.log(2.0)


def _prompt_and_full_text(ex: dict, spec: config.ModelSpec, tokenizer) -> tuple[str, str]:
    """The prompt, and the prompt+answer it must be a prefix of."""
    question = ex["question"].strip()
    prompt = models.build_prompt(question, spec, tokenizer)
    if spec.prompt_style == "chat":
        both = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ex["answer"]},
        ]
        return prompt, tokenizer.apply_chat_template(both, tokenize=False)
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
    # per-example, kept so the corpus number can be given an interval later.
    # Only sums were retained before, and a sum cannot be resampled.
    per_example_nll: list[float] = []
    per_example_bytes: list[int] = []

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

        example_nll = nll.item()
        example_bytes = len(
            tokenizer.decode(full_ids[n_prompt:], skip_special_tokens=True).encode("utf-8")
        )
        total_nll += example_nll
        total_tokens += target.numel()
        total_bytes += example_bytes
        per_example_nll.append(example_nll)
        per_example_bytes.append(example_bytes)

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
        # stripped out by `evaluate_model` and written beside metrics.json —
        # 1641 pairs per arm would swamp the file people actually read
        "per_example": {"nll": per_example_nll, "bytes": per_example_bytes},
    }


def evaluate_model(
    model_key: str,
    base: bool = False,
    adapter: str | Path | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> tuple[str, dict]:
    """Score one arm. `base=True` gives the untrained control condition.

    `seed` selects a Phase 6.4 variance run: it resolves that run's adapter and
    files the result under its own name, so bracketing the published number never
    overwrites it.
    """
    spec = config.get_spec(model_key)
    tokenizer = data.get_tokenizer(spec.base_id)
    eval_ds = data.held_out(limit)

    if seed is not None and adapter is None and not base:
        adapter = spec.seeded_output_dir(seed)

    run_name, model = models.load_arm(spec, base=base, adapter=adapter)
    if seed is not None:
        run_name = spec.seeded_run_name(seed, base=base)

    metrics = score_dataset(model, tokenizer, eval_ds, spec)
    write_per_example(run_name, metrics.pop("per_example"))
    metrics.update(
        {
            "model": spec.key,
            "base_id": spec.base_id,
            "fine_tuned": not base,
            # when, recorded by the run itself — the README states this date, and
            # a rerun of one arm must not silently redate the others
            "measured_on": date.today().isoformat(),
        }
    )
    return run_name, metrics


def write_per_example(run_name: str, per_example: dict, directory: Path | None = None) -> Path:
    """Per-row NLL and byte counts, kept beside metrics.json.

    A corpus bits-per-byte is one number with no interval attached. These rows are
    what let `medqa.variance` resample it, and — because every arm is scored on the
    same held-out rows in the same order — compare two arms *paired*.
    """
    directory = directory or config.PER_EXAMPLE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}.json"
    path.write_text(json.dumps(per_example))
    return path


def update_metrics(run_name: str, patch: dict, path: Path = config.METRICS_PATH) -> None:
    """Merge fields into one arm's entry, keeping every other field it already has.

    Separate measurements — likelihood here, answer quality in `quality.py` —
    accumulate in one place per arm instead of overwriting each other.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.setdefault(run_name, {}).update(patch)
    path.write_text(json.dumps(existing, indent=2))
    print(f"updated {run_name} → {path}")


def write_metrics(run_name: str, metrics: dict, path: Path = config.METRICS_PATH) -> None:
    """Record one arm's likelihood numbers in metrics.json.

    Merges rather than replaces, on two levels. Evaluating TinyLlama must not
    erase GPT-2; and re-running the likelihood pass must not erase the *answer
    quality* measured separately for the same arm (`quality.py`), which is
    expensive to reproduce and lives in the same entry. Every field this function
    computes is overwritten, so a rerun still supersedes stale numbers.
    """
    update_metrics(run_name, metrics, path)


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

    lines.append("\nCompare bits/byte across models; perplexity only within one tokenizer.")
    return "\n".join(lines)


def markdown_table(path: Path = config.METRICS_PATH) -> str:
    """The same numbers as `comparison_table`, formatted for the README.

    The README quotes this command's output rather than restating figures by hand,
    so there is exactly one place a result can come from.
    """
    if not path.exists():
        return f"_No metrics recorded yet ({path})._"
    runs = json.loads(path.read_text())

    lines = [
        "| run | fine-tuned | bits/byte ↓ | perplexity | eval rows |",
        "|---|---|---|---|---|",
    ]
    for name, m in sorted(runs.items()):
        tuned = "yes" if m.get("fine_tuned") else "— (control)"
        lines.append(
            f"| `{name}` | {tuned} | {m['bits_per_byte']:.4f} "
            f"| {m['perplexity']:.2f} | {m['n_examples']} |"
        )

    lines.append("")
    for key in sorted({m["model"] for m in runs.values()}):
        ft = runs.get(f"{key}-lora") or runs.get(f"{key}-qlora")
        bs = runs.get(f"{key}-base")
        if ft and bs:
            delta = (bs["bits_per_byte"] - ft["bits_per_byte"]) / bs["bits_per_byte"] * 100
            lines.append(
                f"- **{key}**: fine-tuning cut bits/byte by **{delta:.1f}%** vs its own base model."
            )
        elif ft:
            lines.append(f"- **{key}**: no control measured — the improvement is unattributable.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(config.MODEL_SPECS))
    parser.add_argument("--base", action="store_true", help="score the base model (control)")
    parser.add_argument("--adapter", help="local dir or Hub repo id; defaults to outputs/ then Hub")
    parser.add_argument("--limit", type=int, help="score only N eval rows (smoke test)")
    parser.add_argument(
        "--seed", type=int, help="score a Phase 6.4 variance run trained with this seed"
    )
    parser.add_argument("--table", action="store_true", help="print metrics.json and exit")
    parser.add_argument("--markdown", action="store_true", help="render the table for the README")
    args = parser.parse_args(argv)

    if args.model:
        run_name, metrics = evaluate_model(
            args.model, base=args.base, adapter=args.adapter, limit=args.limit, seed=args.seed
        )
        write_metrics(run_name, metrics)
    elif not (args.table or args.markdown):
        parser.error("pass --model to measure, or --table / --markdown to report")

    print(markdown_table() if args.markdown else comparison_table())


if __name__ == "__main__":
    main()
