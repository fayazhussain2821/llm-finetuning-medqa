"""Blinded human rating — the only measurement here that can see factuality.

    python -m medqa.review --sample 20             # build a blinded review sheet
    # ... fill in outputs/review/ratings.json by hand ...
    python -m medqa.review --report --rater fayaz  # unblind, aggregate, record

ROUGE-L, token F1 and bits per byte all reward phrasing. None of them can tell a
true medical claim from a fluent false one, and for this dataset that is the gap
that matters most. The only instrument that closes it is a person reading the
answers — so this module's whole job is to make that cheap and to stop the person
from fooling themselves while they do it.

**Blinding is the point.** Knowing that an answer came from the 1.1B model makes it
read as better. Each question's four answers are shuffled independently and
labelled A/B/C/D; the mapping back to arms is written to a separate key file that
the rater has no reason to open. `--report` is what unblinds, after the ratings
are already committed to disk.

**Ratings are 1-5 for factual soundness against the reference**, per ROADMAP 6.3:

    1  contradicts the reference, or invents a mechanism//entity outright
    2  mostly wrong, with a fragment of correct material
    3  vague or evasive — nothing clearly false, nothing clearly useful
    4  substantially correct, incomplete or imprecise in places
    5  correct and responsive to the question

A rating is about the *claims*, not the prose. Fluency is precisely what the
automatic metrics already reward, and precisely what makes a wrong answer
dangerous here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from datetime import date
from pathlib import Path

from medqa import config, evaluate

ARMS = ("gpt2-base", "gpt2-lora", "tinyllama-base", "tinyllama-qlora")
LABELS = "ABCD"
SCALE = {
    1: "contradicts the reference, or invents a mechanism/entity outright",
    2: "mostly wrong, with a fragment of correct material",
    3: "vague or evasive — nothing clearly false, nothing clearly useful",
    4: "substantially correct, incomplete or imprecise in places",
    5: "correct and responsive to the question",
}


def load_generations(directory: Path | None = None) -> dict[str, list[dict]]:
    """Every arm's generations, keyed by run name."""
    directory = directory or config.GENERATIONS_DIR
    found = {}
    for arm in ARMS:
        path = directory / f"{arm}.jsonl"
        if path.exists():
            found[arm] = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not found:
        raise FileNotFoundError(f"no generations in {directory} — run `python -m medqa.quality`")
    return found


def build_sheet(
    generations: dict[str, list[dict]], n: int = 20, seed: int = config.REVIEW_SEED
) -> tuple[list[dict], dict]:
    """Sample n questions and shuffle each one's answers independently.

    Shuffling *per question* rather than once for the whole sheet stops the rater
    learning "C is the good one" three questions in and scoring the label
    instead of the text.
    """
    arms = sorted(generations)
    count = min(len(generations[a]) for a in arms)
    if n > count:
        raise ValueError(f"asked for {n} questions but only {count} generations per arm exist")

    rng = random.Random(seed)
    indexes = rng.sample(range(count), n)

    sheet, key = [], {}
    for position, index in enumerate(indexes):
        shuffled = arms[:]
        rng.shuffle(shuffled)
        item_id = f"q{position + 1:02d}"
        sheet.append(
            {
                "id": item_id,
                "question": generations[arms[0]][index]["question"],
                "reference": generations[arms[0]][index]["reference"],
                "answers": {
                    LABELS[slot]: generations[arm][index]["prediction"]
                    for slot, arm in enumerate(shuffled)
                },
            }
        )
        key[item_id] = {LABELS[slot]: arm for slot, arm in enumerate(shuffled)}
    return sheet, key


def render_sheet(sheet: list[dict]) -> str:
    """The document a person actually reads. Reference first, answers unlabelled."""
    out = [
        "# Blinded answer review",
        "",
        "Rate each answer **1-5 for factual soundness against the reference**, and write",
        "the numbers into `outputs/review/ratings.json`. Judge the claims, not the prose —",
        "fluency is what the automatic metrics already reward.",
        "",
    ]
    out += [f"- **{score}** — {text}" for score, text in sorted(SCALE.items())]
    out += ["", "Which model wrote which answer is recorded elsewhere. Do not look first.", ""]

    for item in sheet:
        out += ["---", "", f"## {item['id']}", "", f"**Question.** {item['question']}", ""]
        out += [f"**Reference.** {item['reference']}", ""]
        for label, answer in sorted(item["answers"].items()):
            out += [f"**{label}.** {answer or '_(empty)_'}", ""]
    return "\n".join(out)


def ratings_skeleton(sheet: list[dict]) -> dict:
    """A file shaped exactly like the answer, with the numbers left out."""
    return {item["id"]: dict.fromkeys(sorted(item["answers"])) for item in sheet}


def unblind(ratings: dict, key: dict) -> dict[str, list[int]]:
    """Map A/B/C/D scores back onto the arms that produced them."""
    by_arm: dict[str, list[int]] = {}
    for item_id, scores in ratings.items():
        if item_id not in key:
            raise KeyError(f"{item_id} is in the ratings but not the key — regenerate the sheet")
        for label, score in scores.items():
            if score is None:
                continue
            if score not in SCALE:
                raise ValueError(f"{item_id}.{label}: {score!r} is not a rating from 1-5")
            by_arm.setdefault(key[item_id][label], []).append(score)
    if not by_arm:
        raise ValueError("no ratings filled in yet — every score is still null")
    return by_arm


def summarise(by_arm: dict[str, list[int]], rater: str = "unrecorded") -> dict[str, dict]:
    """Mean, spread and the share of answers that are actively wrong.

    `wrong_rate` (1s and 2s) is reported separately from the mean because it is
    the number that matters for medical text: an arm can carry a respectable
    average while still contradicting the reference a fifth of the time.

    `rater` is stored with every score because *who judged* changes what these
    numbers mean. A model rating its own family's output is a prior; a clinician
    reading the same sheet is a result. A file that does not say which one it holds
    will eventually be quoted as the wrong one.
    """
    summary = {}
    for arm, scores in sorted(by_arm.items()):
        summary[arm] = {
            "mean": statistics.fmean(scores),
            # population sd: these n answers are the whole sample, not an estimate
            "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "wrong_rate": sum(1 for s in scores if s <= 2) / len(scores),
            "n_rated": len(scores),
            "rater": rater,
            "measured_on": date.today().isoformat(),
        }
    return summary


def per_question(ratings: dict, key: dict) -> dict[str, dict[str, int]]:
    """Scores regrouped as question -> arm -> score, keeping the pairing intact."""
    return {
        qid: {key[qid][label]: score for label, score in scores.items() if score is not None}
        for qid, scores in ratings.items()
    }


def paired_comparison(
    scored: dict[str, dict[str, int]], a: str, b: str, resamples: int = 20000, seed: int = 0
) -> dict:
    """Compare two arms **on the same questions**, with a bootstrap interval.

    Every arm answered an identical question set, so the comparison is paired and
    an unpaired summary throws that away — most of the variance here is questions
    being harder or easier, not arms differing.

    The interval matters more than the mean at this sample size. Twenty questions
    cannot resolve a difference of a few tenths, and reporting a bare mean invites
    reading a gap that a rerun would not reproduce. `spans_zero` is the honest
    headline: when it is true, this experiment did not detect a difference.
    """
    diffs = [
        scored[q][a] - scored[q][b] for q in sorted(scored) if a in scored[q] and b in scored[q]
    ]
    if not diffs:
        raise ValueError(f"no question has ratings for both {a!r} and {b!r}")

    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    low, high = means[int(0.025 * resamples)], means[int(0.975 * resamples)]
    return {
        "mean_difference": statistics.fmean(diffs),
        "ci_low": low,
        "ci_high": high,
        "spans_zero": low <= 0 <= high,
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
        "n_paired": len(diffs),
    }


def comparison_table(scored: dict[str, dict[str, int]], pairs: list[tuple[str, str]]) -> str:
    """Each comparison with its interval, and a plain verdict on whether it holds."""
    lines = [
        "| comparison | mean Δ | 95% CI | W/L/T | detected? |",
        "|---|---|---|---|---|",
    ]
    for a, b in pairs:
        c = paired_comparison(scored, a, b)
        verdict = "**no**" if c["spans_zero"] else "yes"
        lines.append(
            f"| `{a}` − `{b}` | {c['mean_difference']:+.2f} "
            f"| [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] "
            f"| {c['wins']}/{c['losses']}/{c['ties']} | {verdict} |"
        )
    return "\n".join(lines)


def markdown_table(summary: dict[str, dict]) -> str:
    lines = [
        "| run | mean 1-5 ↑ | sd | contradicts reference (1-2) ↓ | n |",
        "|---|---|---|---|---|",
    ]
    for arm, s in sorted(summary.items()):
        lines.append(
            f"| `{arm}` | {s['mean']:.2f} | {s['stdev']:.2f} "
            f"| {s['wrong_rate']:.0%} | {s['n_rated']} |"
        )
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────
def _paths(directory: Path) -> dict[str, Path]:
    return {
        "sheet": directory / "sheet.md",
        "ratings": directory / "ratings.json",
        "key": directory / "key.json",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, metavar="N", help="build a sheet of N questions")
    parser.add_argument("--report", action="store_true", help="unblind and aggregate the ratings")
    parser.add_argument(
        "--rater",
        help="who produced the ratings, e.g. 'fayaz' or 'claude-opus-4'; recorded with them",
    )
    parser.add_argument("--seed", type=int, default=config.REVIEW_SEED)
    parser.add_argument("--dir", type=Path, default=config.OUTPUT_DIR / "review")
    args = parser.parse_args(argv)

    paths = _paths(args.dir)

    if args.sample:
        args.dir.mkdir(parents=True, exist_ok=True)
        if paths["ratings"].exists():
            parser.error(
                f"{paths['ratings']} already exists — rating it again would overwrite work. "
                "Move it aside first."
            )
        sheet, key = build_sheet(load_generations(), n=args.sample, seed=args.seed)
        paths["sheet"].write_text(render_sheet(sheet))
        paths["ratings"].write_text(json.dumps(ratings_skeleton(sheet), indent=2))
        paths["key"].write_text(json.dumps(key, indent=2))
        print(f"wrote {paths['sheet']} — read it and fill in {paths['ratings']}")
        print(f"the unblinding key is {paths['key']}; --report reads it for you")
        return

    if args.report:
        if not (paths["ratings"].exists() and paths["key"].exists()):
            parser.error(f"no sheet in {args.dir} — run `--sample 20` first")
        if not args.rater:
            parser.error("--rater is required: these numbers mean different things by who rated")
        ratings = json.loads(paths["ratings"].read_text())
        key = json.loads(paths["key"].read_text())

        try:
            by_arm = unblind(ratings, key)
        except ValueError as exc:  # an unfilled sheet is the likeliest first run
            parser.error(f"{exc} — see {paths['sheet']}")

        summary = summarise(by_arm, rater=args.rater)
        for arm, s in summary.items():
            evaluate.update_metrics(arm, {"review": s})
        print(markdown_table(summary))

        # per-arm means invite reading gaps this sample size cannot support
        scored = per_question(ratings, key)
        rated = sorted({arm for scores in scored.values() for arm in scores})
        if len(rated) > 1:
            print()
            print(comparison_table(scored, list(itertools.combinations(rated, 2))))
        return

    parser.error("pass --sample N to build a sheet, or --report to aggregate one")


if __name__ == "__main__":
    main()
