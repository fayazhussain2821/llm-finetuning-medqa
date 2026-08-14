"""Bootstrap intervals over the held-out sample.

The arithmetic here decides whether a reported gap is called real, so the tests
are about the ways it could quietly say "yes" when it should say "no".
"""

import json

import pytest

from medqa import variance


def _write(directory, run, nll, nbytes):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{run}.json").write_text(json.dumps({"nll": nll, "bytes": nbytes}))


# ── the statistic itself ───────────────────────────────────────────────
def test_bits_per_byte_is_a_ratio_of_totals_not_a_mean_of_ratios():
    """A six-word answer must not weigh the same as a six-hundred-word one.

    Rows: 1 nat over 1 byte, and 1 nat over 99 bytes. The ratio of totals is
    2/100 nats per byte; the mean of per-row ratios would be ~0.505.
    """
    nll, nbytes = [1.0, 1.0], [1, 99]
    assert variance.bits_per_byte(nll, nbytes) == pytest.approx(2 / 100 / variance.LN2)


def test_bits_per_byte_honours_a_resample_index():
    nll, nbytes = [1.0, 5.0], [10, 10]
    assert variance.bits_per_byte(nll, nbytes, [0, 0]) == pytest.approx(
        variance.bits_per_byte([1.0], [10])
    )


def test_bits_per_byte_refuses_an_empty_resample():
    with pytest.raises(ValueError):
        variance.bits_per_byte([1.0], [0])


# ── intervals ──────────────────────────────────────────────────────────
def test_interval_brackets_the_point_estimate(tmp_path):
    _write(tmp_path, "arm", [1.0, 2.0, 3.0, 4.0] * 25, [10] * 100)
    r = variance.interval("arm", resamples=500, directory=tmp_path)
    assert r["ci_low"] <= r["bits_per_byte"] <= r["ci_high"]
    assert r["n_examples"] == 100


def test_interval_is_narrow_when_every_row_agrees(tmp_path):
    """No spread between rows means nothing to resample — the interval collapses."""
    _write(tmp_path, "arm", [2.0] * 50, [10] * 50)
    r = variance.interval("arm", resamples=500, directory=tmp_path)
    assert r["ci_high"] - r["ci_low"] == pytest.approx(0.0, abs=1e-9)


def test_interval_is_reproducible_for_a_seed(tmp_path):
    _write(tmp_path, "arm", [1.0, 9.0] * 50, [10] * 100)
    first = variance.interval("arm", resamples=300, seed=1, directory=tmp_path)
    second = variance.interval("arm", resamples=300, seed=1, directory=tmp_path)
    assert first == second


def test_missing_scores_say_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="medqa.evaluate"):
        variance.load_per_example("absent", tmp_path)


# ── paired comparison ──────────────────────────────────────────────────
def test_paired_reduction_detects_a_consistent_improvement(tmp_path):
    """Better on every row: the interval must exclude zero."""
    _write(tmp_path, "better", [1.0] * 100, [10] * 100)
    _write(tmp_path, "base", [2.0] * 100, [10] * 100)
    c = variance.paired_reduction("better", "base", resamples=500, directory=tmp_path)
    assert c["reduction_pct"] == pytest.approx(50.0)
    assert c["spans_zero"] is False


def test_paired_reduction_reports_no_effect_when_the_arms_are_identical(tmp_path):
    rows = [1.0, 4.0, 2.0, 9.0] * 25
    _write(tmp_path, "a", rows, [10] * 100)
    _write(tmp_path, "b", rows, [10] * 100)
    c = variance.paired_reduction("a", "b", resamples=500, directory=tmp_path)
    assert c["reduction_pct"] == pytest.approx(0.0)
    assert c["spans_zero"] is True


def test_paired_reduction_uses_one_index_for_both_arms(tmp_path):
    """Pairing is the whole point: arms differ hugely per row but not on average.

    Resampled independently this would show a wide interval around zero. Paired,
    the per-row differences cancel exactly and the interval is tight.
    """
    _write(tmp_path, "a", [1.0, 9.0] * 50, [10] * 100)
    _write(tmp_path, "b", [1.0, 9.0] * 50, [10] * 100)
    c = variance.paired_reduction("a", "b", resamples=500, directory=tmp_path)
    assert c["ci_low"] == pytest.approx(0.0)
    assert c["ci_high"] == pytest.approx(0.0)


def test_paired_reduction_refuses_mismatched_eval_sets(tmp_path):
    """Different row counts mean the two arms did not see the same questions."""
    _write(tmp_path, "a", [1.0] * 10, [10] * 10)
    _write(tmp_path, "b", [1.0] * 9, [10] * 9)
    with pytest.raises(ValueError, match="cannot be paired"):
        variance.paired_reduction("a", "b", directory=tmp_path)


# ── seed spread ────────────────────────────────────────────────────────
def test_seed_spread_refuses_to_report_from_a_single_run(tmp_path):
    """One run is not a spread, and must not be dressed up as one."""
    _write(tmp_path, "gpt2-lora", [1.0] * 10, [10] * 10)
    result = variance.seed_spread(["gpt2-lora"], directory=tmp_path)
    assert result["measured"] is False
    assert "stdev" not in result


def test_seed_spread_reports_across_runs(tmp_path):
    _write(tmp_path, "gpt2-lora", [1.0] * 10, [10] * 10)
    _write(tmp_path, "gpt2-lora-s43", [2.0] * 10, [10] * 10)
    result = variance.seed_spread(["gpt2-lora", "gpt2-lora-s43"], directory=tmp_path)
    assert result["measured"] is True and result["n_runs"] == 2
    assert result["stdev"] > 0
    assert result["max"] > result["min"]


# ── reporting ──────────────────────────────────────────────────────────
def test_tables_without_any_scores(tmp_path):
    assert "No per-example scores yet" in variance.markdown_table(tmp_path)
    assert "no per-example scores yet" in variance.text_table(tmp_path)


def test_markdown_table_flags_a_comparison_that_does_not_survive(tmp_path):
    rows = [1.0, 4.0] * 50
    _write(tmp_path, "gpt2-lora", rows, [10] * 100)
    _write(tmp_path, "gpt2-base", rows, [10] * 100)
    table = variance.markdown_table(tmp_path)
    assert "| `gpt2-base` → `gpt2-lora` |" in table
    assert "**no**" in table


def test_text_table_separates_sampling_noise_from_seed_noise(tmp_path):
    """The two are not substitutes, and the report must not let them blur."""
    _write(tmp_path, "gpt2-lora", [1.0] * 20, [10] * 20)
    assert "seed" in variance.text_table(tmp_path)
