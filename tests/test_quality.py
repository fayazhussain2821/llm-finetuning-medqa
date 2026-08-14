"""Answer-quality metrics.

All offline and model-free: these check *our* scoring arithmetic, which is the
part that can silently be wrong. Generation itself needs weights and is exercised
by the smoke test in `test_integrity.py`.
"""

import json

import pytest

from medqa import config, quality


# ── tokenization ───────────────────────────────────────────────────────
def test_tokenize_lowercases_and_splits_on_punctuation():
    assert quality.tokenize("Iron-deficiency anemia, treated!") == [
        "iron",
        "deficiency",
        "anemia",
        "treated",
    ]


def test_tokenize_drops_pure_punctuation():
    assert quality.tokenize("--- ... !!!") == []


# ── LCS ────────────────────────────────────────────────────────────────
def test_lcs_length_of_a_subsequence():
    assert quality.lcs_length(["a", "b", "c", "d"], ["a", "c", "d"]) == 3


def test_lcs_length_respects_order():
    """The whole point of ROUGE-L over bag-of-words: "a b" and "b a" share only one."""
    assert quality.lcs_length(["a", "b"], ["b", "a"]) == 1


def test_lcs_length_with_an_empty_side():
    assert quality.lcs_length([], ["a"]) == 0


# ── ROUGE-L ────────────────────────────────────────────────────────────
def test_rouge_l_identical_text_scores_one():
    assert quality.rouge_l("low iron levels", "low iron levels")["f1"] == pytest.approx(1.0)


def test_rouge_l_disjoint_text_scores_zero():
    assert quality.rouge_l("aspirin dosage", "volcano eruption")["f1"] == 0.0


def test_rouge_l_precision_and_recall_differ_with_length():
    """A short correct answer inside a long reference: precise, not thorough."""
    scores = quality.rouge_l("low iron", "anemia is caused by low iron in the blood")
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] < 0.5


def test_rouge_l_empty_prediction_scores_zero():
    """A model that says nothing must not score well by saying nothing."""
    assert quality.rouge_l("", "low iron levels")["f1"] == 0.0


def test_rouge_l_matches_the_reference_implementation():
    """Values from `rouge_score.RougeScorer(["rougeL"], use_stemmer=False)`, v0.1.2.

    Hardcoded rather than computed against the package, so the check costs no
    dependency — but they came from running it, not from arithmetic done here.
    """
    cases = [
        ("the cat sat on the mat", "the cat was on the mat", 0.8333333333333334),
        ("low iron", "anemia is caused by low iron in the blood", 0.3636363636363636),
        ("fever cough fatigue", "fatigue cough fever", 0.3333333333333333),
        ("iron iron iron", "iron", 0.5),
    ]
    for prediction, reference, expected in cases:
        assert quality.rouge_l(prediction, reference)["f1"] == pytest.approx(expected)


# ── token F1 ───────────────────────────────────────────────────────────
def test_token_f1_ignores_word_order_where_rouge_does_not():
    """Right content, wrong shape — the two metrics disagreeing is informative."""
    prediction, reference = "fever cough fatigue", "fatigue cough fever"
    assert quality.token_f1(prediction, reference) == pytest.approx(1.0)
    assert quality.rouge_l(prediction, reference)["f1"] < 1.0


def test_token_f1_counts_duplicates():
    """Repeating a word must not manufacture extra overlap."""
    assert quality.token_f1("iron iron iron", "iron") == pytest.approx(0.5)


def test_token_f1_empty_prediction_scores_zero():
    assert quality.token_f1("", "low iron") == 0.0


# ── degeneration ───────────────────────────────────────────────────────
def test_repeated_ngram_rate_catches_a_loop():
    """The failure bits-per-byte is blind to: the model stuck in a cycle."""
    looping = "see your doctor " * 8
    assert quality.repeated_ngram_rate(looping) > 0.8


def test_repeated_ngram_rate_is_zero_for_varied_text():
    varied = "anemia is caused by a shortage of healthy red blood cells in the body"
    assert quality.repeated_ngram_rate(varied) == 0.0


def test_repeated_ngram_rate_of_text_shorter_than_the_window():
    assert quality.repeated_ngram_rate("low iron") == 0.0


# ── aggregation ────────────────────────────────────────────────────────
def test_aggregate_length_ratio_uses_totals_not_a_mean_of_ratios():
    """One over-long answer to a one-line question must not swamp the corpus.

    Averaging per-example ratios gives (0.01 + 10) / 2 ≈ 5.0 — "the models write
    five times too much", from a single row. The ratio of totals says 0.92.
    """
    scored = [
        {"n_pred_tokens": 1, "n_ref_tokens": 100, **_zeros()},
        {"n_pred_tokens": 100, "n_ref_tokens": 10, **_zeros()},
    ]
    summary = quality.aggregate(scored)

    assert summary["length_ratio"] == pytest.approx(101 / 110)
    mean_of_ratios = (1 / 100 + 100 / 10) / 2
    assert summary["length_ratio"] != pytest.approx(mean_of_ratios)


def test_aggregate_counts_empty_answers():
    scored = [
        {"n_pred_tokens": 0, "n_ref_tokens": 10, **_zeros()},
        {"n_pred_tokens": 5, "n_ref_tokens": 10, **_zeros()},
    ]
    assert quality.aggregate(scored)["empty_rate"] == pytest.approx(0.5)


def test_aggregate_refuses_an_empty_run():
    with pytest.raises(ValueError):
        quality.aggregate([])


def _zeros() -> dict:
    return {"rouge_l_f1": 0.0, "token_f1": 0.0, "repeated_4gram_rate": 0.0}


# ── generation post-processing ─────────────────────────────────────────
def test_trim_continuation_cuts_a_hallucinated_next_turn():
    text = f"Low iron.\n\n{config.INSTRUCTION_MARKER}What is asthma?"
    assert quality.trim_continuation(text) == "Low iron."


def test_trim_continuation_leaves_a_clean_answer_alone():
    assert quality.trim_continuation("  Low iron levels.  ") == "Low iron levels."


def test_score_one_reports_every_per_example_field():
    scores = quality.score_one("low iron", "anemia is low iron")
    assert set(scores) == {
        "rouge_l_f1",
        "rouge_l_precision",
        "rouge_l_recall",
        "token_f1",
        "repeated_4gram_rate",
        "n_pred_tokens",
        "n_ref_tokens",
    }


def test_write_generations_is_one_json_object_per_line(tmp_path):
    """The jsonl is the evidence a human reads — it must survive round-tripping."""
    records = [{"question": "q1", "prediction": "p1"}, {"question": "q2", "prediction": "p2"}]
    path = quality.write_generations("gpt2-lora", records, tmp_path)

    parsed = [json.loads(line) for line in path.read_text().splitlines()]
    assert parsed == records


# ── reporting ──────────────────────────────────────────────────────────
def _metrics_file(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "gpt2-lora": {
                    "bits_per_byte": 0.5970,
                    "quality": {
                        "rouge_l_f1": 0.1234,
                        "token_f1": 0.2345,
                        "repeated_4gram_rate": 0.4567,
                        "length_ratio": 0.75,
                        "empty_rate": 0.0,
                        "n_generated": 200,
                    },
                },
                # scored for likelihood but never generated from — must be skipped
                "gpt2-base": {"bits_per_byte": 0.8049},
            }
        )
    )
    return path


def test_quality_table_skips_arms_with_no_generations(tmp_path):
    table = quality.quality_table(_metrics_file(tmp_path))
    assert "gpt2-lora" in table and "gpt2-base" not in table
    assert "0.1234" in table


def test_markdown_table_renders_the_measured_numbers(tmp_path):
    table = quality.markdown_table(_metrics_file(tmp_path))
    assert "| `gpt2-lora` |" in table
    assert "0.1234" in table and "0.4567" in table
    assert table.count("|") > 10


def test_tables_without_a_metrics_file(tmp_path):
    absent = tmp_path / "absent.json"
    assert "no answer-quality runs yet" in quality.quality_table(absent)
    assert "No answer-quality runs recorded yet" in quality.markdown_table(absent)


def test_generations_land_under_outputs():
    """outputs/ is gitignored — generations must never reach a commit by accident."""
    assert config.GENERATIONS_DIR.parent == config.OUTPUT_DIR
