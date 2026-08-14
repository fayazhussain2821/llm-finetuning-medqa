"""How much of a reported gap is real, and how much is noise.

    python -m medqa.variance              # intervals and paired comparisons
    python -m medqa.variance --markdown   # the same, rendered for the README

Every headline in this project so far is a point estimate: 0.5970 against 0.6120,
a 25.8% reduction, a 33.8% gap. None of them carries an interval, so none of them
can be checked against the only question that matters — **would a rerun say the
same thing?**

There are two independent sources of noise, and they need different experiments:

1. **Which questions were held out.** MedQuAD's 1,641 eval rows are a sample. A
   different draw would give slightly different numbers, and this module measures
   how much by resampling the rows it already has (a bootstrap over examples).
2. **Which training run happened.** Data order, LoRA initialisation and dropout
   all move the result. Measuring this needs the same arm trained several times —
   `python -m medqa.train --model gpt2 --seed 43` — and cannot be faked from one
   run's outputs. `seed_spread` reports it once those runs exist.

**This module measures (1) and cannot measure (2).** They are not substitutes: a
gap can clear the eval-sampling interval comfortably and still vanish under a
different seed. Reported separately, and labelled, so neither is mistaken for the
other.

**Comparisons are paired.** Every arm is scored on the same held-out rows in the
same order, so a resample draws one set of row indices and applies it to both
arms. Bootstrapping the arms independently would throw that away and inflate the
interval with variance that cancels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from medqa import config

LN2 = math.log(2.0)
RESAMPLES = 10000


def load_per_example(run_name: str, directory: Path | None = None) -> dict[str, list]:
    directory = directory or config.PER_EXAMPLE_DIR
    path = directory / f"{run_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no per-example scores for {run_name} ({path}) — re-run `python -m medqa.evaluate`"
        )
    return json.loads(path.read_text())


def available_runs(directory: Path | None = None) -> list[str]:
    directory = directory or config.PER_EXAMPLE_DIR
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def bits_per_byte(nll: list[float], nbytes: list[int], index: list[int] | None = None) -> float:
    """Corpus bits per byte over a selection of rows — a ratio of sums, not a mean.

    Averaging per-row bits-per-byte would weight a six-word answer the same as a
    six-hundred-word one. The published figure is the ratio of totals, so the
    resampled figure has to be too, or the interval describes a different number.
    """
    if index is None:
        total_nll, total_bytes = math.fsum(nll), sum(nbytes)
    else:
        total_nll = math.fsum(nll[i] for i in index)
        total_bytes = sum(nbytes[i] for i in index)
    if not total_bytes:
        raise ValueError("no bytes in this resample")
    return total_nll / LN2 / total_bytes


def _resample_indices(n: int, resamples: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def interval(
    run_name: str, resamples: int = RESAMPLES, seed: int = 0, directory: Path | None = None
) -> dict:
    """Bits per byte with a 95% bootstrap interval over the held-out rows."""
    rows = load_per_example(run_name, directory)
    nll, nbytes = rows["nll"], rows["bytes"]
    draws = sorted(
        bits_per_byte(nll, nbytes, idx) for idx in _resample_indices(len(nll), resamples, seed)
    )
    return {
        "bits_per_byte": bits_per_byte(nll, nbytes),
        "ci_low": draws[int(0.025 * resamples)],
        "ci_high": draws[int(0.975 * resamples)],
        "n_examples": len(nll),
    }


def paired_reduction(
    better: str,
    baseline: str,
    resamples: int = RESAMPLES,
    seed: int = 0,
    directory: Path | None = None,
) -> dict:
    """Percent reduction in bits per byte, `baseline` → `better`, with an interval.

    Positive means `better` really is better. The interval is what decides whether
    a headline like "25.8%" survives contact with a different sample of questions.
    """
    a, b = load_per_example(better, directory), load_per_example(baseline, directory)
    if len(a["nll"]) != len(b["nll"]):
        raise ValueError(
            f"{better} scored {len(a['nll'])} rows and {baseline} scored {len(b['nll'])} — "
            "these were not run on the same held-out set and cannot be paired"
        )

    def reduction(index: list[int] | None) -> float:
        base = bits_per_byte(b["nll"], b["bytes"], index)
        return (base - bits_per_byte(a["nll"], a["bytes"], index)) / base * 100

    # one set of row indices, applied to both arms — the comparison is paired
    draws = sorted(reduction(idx) for idx in _resample_indices(len(a["nll"]), resamples, seed))
    low, high = draws[int(0.025 * resamples)], draws[int(0.975 * resamples)]
    return {
        "reduction_pct": reduction(None),
        "ci_low": low,
        "ci_high": high,
        "spans_zero": low <= 0 <= high,
        "n_examples": len(a["nll"]),
    }


def seed_spread(runs: list[str], directory: Path | None = None) -> dict:
    """Spread across training seeds of the *same* arm — noise source (2).

    Needs several runs of one configuration, named `<arm>-s<seed>`. Until those
    exist this reports nothing rather than guessing, because the sampling interval
    above is not a stand-in for it.
    """
    values = [interval(run, resamples=1, directory=directory)["bits_per_byte"] for run in runs]
    if len(values) < 2:
        return {"n_runs": len(values), "measured": False}
    return {
        "n_runs": len(values),
        "measured": True,
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "runs": dict(zip(runs, values, strict=True)),
    }


# ── reporting ──────────────────────────────────────────────────────────
def default_comparisons(runs: list[str]) -> list[tuple[str, str]]:
    """The comparisons this project actually makes claims about."""
    pairs = [
        ("gpt2-lora", "gpt2-base"),
        ("tinyllama-qlora", "tinyllama-base"),
        ("tinyllama-qlora", "gpt2-lora"),
        ("tinyllama-base", "gpt2-lora"),
    ]
    return [(a, b) for a, b in pairs if a in runs and b in runs]


def markdown_table(directory: Path | None = None) -> str:
    runs = available_runs(directory)
    if not runs:
        return "_No per-example scores yet — run `python -m medqa.evaluate`._"

    lines = [
        "| run | bits/byte | 95% CI (eval sample) | n |",
        "|---|---|---|---|",
    ]
    for run in runs:
        r = interval(run, directory=directory)
        lines.append(
            f"| `{run}` | {r['bits_per_byte']:.4f} "
            f"| [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['n_examples']} |"
        )

    comparisons = default_comparisons(runs)
    if comparisons:
        lines += [
            "",
            "| comparison | reduction | 95% CI | survives resampling? |",
            "|---|---|---|---|",
        ]
        for a, b in comparisons:
            c = paired_reduction(a, b, directory=directory)
            verdict = "**no**" if c["spans_zero"] else "yes"
            lines.append(
                f"| `{b}` → `{a}` | {c['reduction_pct']:.1f}% "
                f"| [{c['ci_low']:.1f}%, {c['ci_high']:.1f}%] | {verdict} |"
            )
    return "\n".join(lines)


def text_table(directory: Path | None = None) -> str:
    runs = available_runs(directory)
    if not runs:
        return "no per-example scores yet — run `python -m medqa.evaluate`"

    header = f"{'run':<20}{'bits/byte':>11}{'95% CI':>22}{'n':>7}"
    lines = [header, "-" * len(header)]
    for run in runs:
        r = interval(run, directory=directory)
        ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        lines.append(f"{run:<20}{r['bits_per_byte']:>11.4f}{ci:>22}{r['n_examples']:>7}")

    lines.append("")
    for a, b in default_comparisons(runs):
        c = paired_reduction(a, b, directory=directory)
        verdict = "NOT distinguishable from zero" if c["spans_zero"] else "holds"
        lines.append(
            f"{b} -> {a}: {c['reduction_pct']:.1f}% "
            f"[{c['ci_low']:.1f}%, {c['ci_high']:.1f}%] — {verdict}"
        )

    lines.append("")
    lines.append("These intervals cover the held-out sample only. Run-to-run (seed) spread")
    lines.append("is a separate experiment: `medqa.train --seed 43`, then re-evaluate.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", action="store_true", help="render for the README")
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args(argv)
    print(markdown_table(args.dir) if args.markdown else text_table(args.dir))


if __name__ == "__main__":
    main()
