#!/usr/bin/env python
"""Inject the measured results into README.md between the RESULTS markers.

    python scripts/update_readme_results.py

The README must never contain a figure someone typed. This reads
`outputs/metrics.json` — the only place results are allowed to come from — renders
it, and rewrites the block. Run it after any evaluation; CI has nothing to do with
it, since CI has no GPU and no adapters.
"""

from __future__ import annotations

import sys
from datetime import date

from medqa import config, evaluate

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"

PREAMBLE = """## Results

Held-out set: {n} examples, identical rows for every arm. Lower bits/byte is better.
Perplexity is shown only for continuity with the original report — compare it *within*
a model, never across the two.
"""

FOOTER = """
Measured {when} by `python -m medqa.evaluate`, read from `outputs/metrics.json`.
Regenerate this block with `python scripts/update_readme_results.py`.
"""


def main() -> int:
    if not config.METRICS_PATH.exists():
        print(f"no metrics at {config.METRICS_PATH} — run `python -m medqa.evaluate` first")
        return 1

    import json

    runs = json.loads(config.METRICS_PATH.read_text())
    n = max(m["n_examples"] for m in runs.values())

    block = "\n".join(
        [
            BEGIN,
            PREAMBLE.format(n=n),
            evaluate.markdown_table(),
            FOOTER.format(when=date.today().isoformat()),
            END,
        ]
    )

    readme = config.ROOT / "README.md"
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        print(f"markers {BEGIN} / {END} not found in {readme}")
        return 1

    start, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(start + block + tail)
    print(f"updated {readme} with {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
