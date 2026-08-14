"""Answer quality — what the model actually *writes*, not how likely the truth was.

    python -m medqa.quality --model gpt2                   # fine-tuned adapter
    python -m medqa.quality --model tinyllama --base       # control: no fine-tuning
    python -m medqa.quality --table                        # read metrics.json, print

`evaluate.py` measures bits-per-byte: how much probability mass an arm puts on the
reference answer. A model can win that outright and still be useless, because
likelihood never asks it to produce anything. Two failure modes it cannot see:

* **Degeneration.** A model that loops — "consult your doctor consult your doctor
  …" — is scored on the reference tokens, which it never had to generate.
* **Fluent irrelevance.** Confident prose about the wrong condition costs a
  likelihood metric nothing at all.

So here the arm is made to answer, greedily, and the text is compared with the
reference:

* **ROUGE-L F1** — longest-common-subsequence overlap. Weak (it rewards wording,
  not truth) but standard, and it does separate "answered the question" from
  "answered a different one".
* **Token F1** — bag-of-words overlap, insensitive to order. ROUGE-L and token F1
  disagreeing means the right content in the wrong shape.
* **Repeated 4-gram rate** — the degeneration detector. Human answers sit near
  zero; a looping model approaches one.
* **Length ratio and empty rate** — the other two ways to score well and say
  nothing.

None of these is truth. A high ROUGE-L answer can still be medically wrong, which
is exactly why `outputs/generations/<run>.jsonl` keeps every generation: the
honest metric is a human reading them, and this file is what makes that possible.

Decoding is `models.GREEDY` for every arm — deterministic, so a rerun reproduces
the number, and identical across arms, so decoding is never the variable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import torch

from medqa import config, data, evaluate, models

#: Tokenization for scoring only: lowercase, and anything not alphanumeric is a
#: separator. Matches `rouge_score.RougeScorer(use_stemmer=False)`, so the ROUGE-L
#: numbers here are comparable with published ones rather than a private variant.
_NON_ALPHANUM = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _NON_ALPHANUM.sub(" ", text.lower()).split() if t]


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence, two rows of DP instead of the full table.

    Answers here run to hundreds of tokens; the square table is the memory, not
    the time, that hurts.
    """
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def _f1(overlap: int, n_pred: int, n_ref: int) -> dict[str, float]:
    if not overlap:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision, recall = overlap / n_pred, overlap / n_ref
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """LCS-based precision/recall/F1 — word order matters."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return _f1(lcs_length(pred, ref), len(pred), len(ref))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-words overlap, counting duplicates — order-blind by design."""
    pred, ref = Counter(tokenize(prediction)), Counter(tokenize(reference))
    if not pred or not ref:
        return 0.0
    return _f1(sum((pred & ref).values()), sum(pred.values()), sum(ref.values()))["f1"]


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are not the first of their kind. 0 = never repeats.

    The cheapest reliable degeneration signal there is. Reference answers score
    low; a model stuck in a loop scores close to 1.
    """
    tokens = tokenize(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def score_one(prediction: str, reference: str) -> dict[str, float]:
    """Every per-example number, so the jsonl is inspectable without rescoring."""
    rouge = rouge_l(prediction, reference)
    return {
        "rouge_l_f1": rouge["f1"],
        "rouge_l_precision": rouge["precision"],
        "rouge_l_recall": rouge["recall"],
        "token_f1": token_f1(prediction, reference),
        "repeated_4gram_rate": repeated_ngram_rate(prediction),
        "n_pred_tokens": len(tokenize(prediction)),
        "n_ref_tokens": len(tokenize(reference)),
    }


def aggregate(scored: list[dict]) -> dict[str, float]:
    """Corpus numbers. Means per example, except length ratio, which is a total.

    Averaging per-example length ratios lets one two-word answer against a
    600-word reference swamp the rest; the ratio of totals cannot.
    """
    if not scored:
        raise ValueError("nothing to aggregate — no generations were scored")

    n = len(scored)
    pred_tokens = sum(s["n_pred_tokens"] for s in scored)
    ref_tokens = sum(s["n_ref_tokens"] for s in scored)
    return {
        "rouge_l_f1": sum(s["rouge_l_f1"] for s in scored) / n,
        "token_f1": sum(s["token_f1"] for s in scored) / n,
        "repeated_4gram_rate": sum(s["repeated_4gram_rate"] for s in scored) / n,
        "empty_rate": sum(1 for s in scored if not s["n_pred_tokens"]) / n,
        "length_ratio": pred_tokens / ref_tokens if ref_tokens else 0.0,
        "n_generated": n,
    }


def trim_continuation(text: str) -> str:
    """Cut a hallucinated next turn off the end of a generation.

    Instruction-tuned-by-us models happily roll on into `### Instruction:` and
    invent the user's next question; leaving that in would score the model's
    imagination against this answer. Applied identically to every arm, so it
    cannot flatter one of them.
    """
    for marker in (config.INSTRUCTION_MARKER, config.RESPONSE_MARKER):
        head = marker.rstrip("\n")
        index = text.find(head)
        if index != -1:
            text = text[:index]
    return text.strip()


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    dataset,
    spec: config.ModelSpec,
    preset: models.GenerationPreset = models.GREEDY,
) -> list[dict]:
    """Ask each held-out question and keep what comes back.

    One row at a time: batching needs left padding, and a padding-side mistake
    would quietly degrade one arm's answers and look like a finding.
    """
    device = next(model.parameters()).device
    window = evaluate.context_window(model)
    max_prompt = max(window - preset.max_new_tokens, 1)
    records = []

    for i, ex in enumerate(dataset):
        prompt = models.build_prompt(ex["question"], spec, tokenizer)
        ids = tokenizer(prompt, truncation=True, max_length=max_prompt, return_tensors="pt")
        ids = {k: v.to(device) for k, v in ids.items()}

        out = model.generate(
            **ids,
            **preset.to_kwargs(),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0][ids["input_ids"].shape[1] :]  # drop the echoed prompt
        prediction = trim_continuation(
            tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        )

        records.append(
            {
                "question": ex["question"],
                "reference": ex["answer"],
                "prediction": prediction,
                **score_one(prediction, ex["answer"]),
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(dataset)} generated", flush=True)

    return records


def write_generations(run_name: str, records: list[dict], directory: Path | None = None) -> Path:
    """Persist every generation. The metric is a summary; this is the evidence."""
    directory = directory or config.GENERATIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def measure_quality(
    model_key: str,
    base: bool = False,
    adapter: str | Path | None = None,
    limit: int = config.QUALITY_N,
    max_new_tokens: int = config.QUALITY_MAX_NEW_TOKENS,
) -> tuple[str, dict]:
    """Generate for one arm, score it, and return the corpus numbers."""
    spec = config.get_spec(model_key)
    tokenizer = data.get_tokenizer(spec.base_id)
    eval_ds = data.held_out(limit)

    run_name, model = models.load_arm(spec, base=base, adapter=adapter)
    preset = models.GenerationPreset(max_new_tokens=max_new_tokens, do_sample=False)

    print(f"generating {len(eval_ds)} answers for {run_name} (greedy, ≤{max_new_tokens} tokens)")
    records = generate_answers(model, tokenizer, eval_ds, spec, preset)

    path = write_generations(run_name, records)
    print(f"wrote {len(records)} generations → {path}")

    summary = aggregate(records)
    summary["max_new_tokens"] = max_new_tokens
    summary["measured_on"] = date.today().isoformat()
    return run_name, summary


# ── reporting ──────────────────────────────────────────────────────────
def _quality_runs(path: Path = config.METRICS_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        name: run["quality"]
        for name, run in json.loads(path.read_text()).items()
        if isinstance(run, dict) and "quality" in run
    }


def quality_table(path: Path = config.METRICS_PATH) -> str:
    runs = _quality_runs(path)
    if not runs:
        return f"no answer-quality runs yet — `python -m medqa.quality --model gpt2` ({path})"

    header = f"{'run':<20}{'ROUGE-L':>10}{'tok-F1':>9}{'rep-4g':>9}{'len':>8}{'n':>6}"
    lines = [header, "-" * len(header)]
    for name, q in sorted(runs.items()):
        lines.append(
            f"{name:<20}{q['rouge_l_f1']:>10.4f}{q['token_f1']:>9.4f}"
            f"{q['repeated_4gram_rate']:>9.4f}{q['length_ratio']:>8.2f}{q['n_generated']:>6}"
        )
    lines.append("\nROUGE-L/token-F1 higher is better; repeated-4gram lower is better.")
    lines.append("len is generated tokens per reference token — 1.0 means matched length.")
    return "\n".join(lines)


def markdown_table(path: Path = config.METRICS_PATH) -> str:
    """The same numbers, for the README. Never retyped by hand."""
    runs = _quality_runs(path)
    if not runs:
        return f"_No answer-quality runs recorded yet ({path})._"

    lines = [
        "| run | ROUGE-L F1 ↑ | token F1 ↑ | repeated 4-grams ↓ | length ratio | empty | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, q in sorted(runs.items()):
        lines.append(
            f"| `{name}` | {q['rouge_l_f1']:.4f} | {q['token_f1']:.4f} "
            f"| {q['repeated_4gram_rate']:.4f} | {q['length_ratio']:.2f} "
            f"| {q['empty_rate']:.1%} | {q['n_generated']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(config.MODEL_SPECS))
    parser.add_argument("--base", action="store_true", help="generate from the base model")
    parser.add_argument("--adapter", help="local dir or Hub repo id; defaults to outputs/ then Hub")
    parser.add_argument(
        "--limit", type=int, default=config.QUALITY_N, help="held-out rows to generate for"
    )
    parser.add_argument("--max-new-tokens", type=int, default=config.QUALITY_MAX_NEW_TOKENS)
    parser.add_argument("--table", action="store_true", help="print what has been measured")
    parser.add_argument("--markdown", action="store_true", help="render the table for the README")
    args = parser.parse_args(argv)

    if args.model:
        run_name, summary = measure_quality(
            args.model,
            base=args.base,
            adapter=args.adapter,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
        # merge, never replace: bits-per-byte for this arm lives in the same entry
        evaluate.update_metrics(run_name, {"quality": summary})
    elif not (args.table or args.markdown):
        parser.error("pass --model to measure, or --table / --markdown to report")

    print(markdown_table() if args.markdown else quality_table())


if __name__ == "__main__":
    main()
