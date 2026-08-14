#!/usr/bin/env python
"""Inject the measured results into README.md between the marker pairs.

    python scripts/update_readme_results.py

The README must never contain a figure someone typed. This reads
`outputs/metrics.json` — the only place results are allowed to come from — renders
it, and rewrites the blocks. Run it after any evaluation; CI has nothing to do with
it, since CI has no GPU and no adapters.

Two blocks, because they are two different measurements: RESULTS is likelihood on
the full held-out set, QUALITY is what the models generate on a subset of it.
"""

from __future__ import annotations

import json
import sys
from datetime import date

from medqa import config, evaluate, quality

RESULTS_BEGIN, RESULTS_END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
QUALITY_BEGIN, QUALITY_END = "<!-- QUALITY:BEGIN -->", "<!-- QUALITY:END -->"

PREAMBLE = """## Results

Held-out set: {n} examples, identical rows for every arm. Lower bits/byte is better.
Perplexity is shown only for continuity with the original report — compare it *within*
a model, never across the two.
"""

FOOTER = """
Measured {when} by `python -m medqa.evaluate`, read from `outputs/metrics.json`.
Regenerate this block with `python scripts/update_readme_results.py`.
"""

QUALITY_PREAMBLE = """Greedy decoding, ≤{max_new} new tokens, the first {n} held-out rows —
the same questions for every arm. Likelihood says how plausible the reference was;
these say what the model actually wrote when asked.
"""

QUALITY_FOOTER = """
Measured {when} by `python -m medqa.quality`. Every generation is kept in
`outputs/generations/<run>.jsonl` — the summary is a number, that file is the evidence.
"""


def _replace_block(text: str, begin: str, end: str, block: str) -> str | None:
    """Swap what is between the markers, leaving the markers in place."""
    if begin not in text or end not in text:
        return None
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + begin + block + end + tail


def _measured_on(entries: list[dict]) -> str:
    """When these numbers were actually produced, per the runs themselves.

    Not `date.today()`. Rendering the README a week after the run would otherwise
    stamp today's date on week-old numbers — a claim about the experiment that no
    one measured. Falls back to today only for runs recorded before this field
    existed, and says so.
    """
    dates = sorted(e["measured_on"] for e in entries if "measured_on" in e)
    if not dates:
        return f"{date.today().isoformat()} (rendered; the run recorded no date)"
    latest = dates[-1]
    return latest if dates[0] == latest else f"{dates[0]}–{latest}"


def main() -> int:
    if not config.METRICS_PATH.exists():
        print(f"no metrics at {config.METRICS_PATH} — run `python -m medqa.evaluate` first")
        return 1

    runs = json.loads(config.METRICS_PATH.read_text())
    likelihood = [m for m in runs.values() if "n_examples" in m]
    n = max(m["n_examples"] for m in likelihood)

    blocks = {
        (RESULTS_BEGIN, RESULTS_END): "\n".join(
            [
                "",
                PREAMBLE.format(n=n),
                evaluate.markdown_table(),
                FOOTER.format(when=_measured_on(likelihood)),
                "",
            ]
        )
    }

    measured = [r["quality"] for r in runs.values() if "quality" in r]
    if measured:
        max_new = max(q.get("max_new_tokens", config.QUALITY_MAX_NEW_TOKENS) for q in measured)
        blocks[(QUALITY_BEGIN, QUALITY_END)] = "\n".join(
            [
                "",
                QUALITY_PREAMBLE.format(n=max(q["n_generated"] for q in measured), max_new=max_new),
                quality.markdown_table(),
                QUALITY_FOOTER.format(when=_measured_on(measured)),
                "",
            ]
        )
    else:
        print("no answer-quality runs in metrics.json — leaving that block alone")

    readme = config.ROOT / "README.md"
    text = readme.read_text()
    for (begin, end), block in blocks.items():
        updated = _replace_block(text, begin, end, block)
        if updated is None:
            print(f"markers {begin} / {end} not found in {readme}")
            return 1
        text = updated

    readme.write_text(text)
    print(f"updated {readme} with {len(runs)} run(s), {len(blocks)} block(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
