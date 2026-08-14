"""The blinded review harness.

The failure modes here are subtle and would silently corrupt a human's ratings:
a sheet that leaks which arm wrote what, a key that no longer matches the sheet,
or a report that quietly drops unrated items. All offline, no generations needed.
"""

import json
import statistics

import pytest

from medqa import review


def _generations(n: int = 5) -> dict[str, list[dict]]:
    """Distinct answers per arm, worded so they never contain an arm's name.

    The blinding test asserts no arm name survives into the rendered sheet. If the
    fixture wrote "gpt2-lora answer 3" as the prediction itself, that assertion
    could never pass and would say nothing about the renderer.
    """
    return {
        arm: [
            {
                "question": f"question {i}",
                "reference": f"reference {i}",
                "prediction": f"prediction from source {slot} for item {i}",
            }
            for i in range(n)
        ]
        for slot, arm in enumerate(review.ARMS)
    }


# ── sampling and blinding ──────────────────────────────────────────────
def test_sheet_hides_which_arm_wrote_each_answer():
    """The sheet is what a person reads; arm names in it would defeat the point."""
    sheet, _ = review.build_sheet(_generations(), n=3)
    rendered = review.render_sheet(sheet)
    for arm in review.ARMS:
        assert arm not in rendered


def test_sheet_keeps_question_and_reference_together():
    """A rater cannot judge factuality without the reference in front of them."""
    sheet, _ = review.build_sheet(_generations(), n=3)
    for item in sheet:
        assert item["question"] and item["reference"]
        assert item["reference"] in review.render_sheet(sheet)


def test_key_maps_every_label_back_to_a_distinct_arm():
    sheet, key = review.build_sheet(_generations(), n=3)
    for item in sheet:
        mapping = key[item["id"]]
        assert set(mapping) == set(item["answers"])
        assert sorted(mapping.values()) == sorted(review.ARMS)


def test_answers_are_shuffled_independently_per_question():
    """One shuffle for the whole sheet would let a rater learn the labels."""
    _, key = review.build_sheet(_generations(20), n=20, seed=1)
    orderings = {tuple(mapping[label] for label in sorted(mapping)) for mapping in key.values()}
    assert len(orderings) > 1


def test_sampling_is_reproducible_for_a_seed():
    first, _ = review.build_sheet(_generations(20), n=5, seed=7)
    second, _ = review.build_sheet(_generations(20), n=5, seed=7)
    assert [i["question"] for i in first] == [i["question"] for i in second]


def test_sampling_refuses_to_ask_for_more_than_exists():
    with pytest.raises(ValueError):
        review.build_sheet(_generations(3), n=10)


def test_ratings_skeleton_matches_the_sheet():
    sheet, _ = review.build_sheet(_generations(), n=3)
    skeleton = review.ratings_skeleton(sheet)
    assert set(skeleton) == {item["id"] for item in sheet}
    assert all(score is None for scores in skeleton.values() for score in scores.values())


# ── unblinding ─────────────────────────────────────────────────────────
def test_unblind_routes_each_score_to_the_arm_that_earned_it():
    key = {"q01": {"A": "gpt2-lora", "B": "tinyllama-qlora"}}
    assert review.unblind({"q01": {"A": 2, "B": 5}}, key) == {
        "gpt2-lora": [2],
        "tinyllama-qlora": [5],
    }


def test_unblind_skips_unrated_answers_without_inventing_a_score():
    """A half-finished sheet must report on what was rated, not fill in blanks."""
    key = {"q01": {"A": "gpt2-lora", "B": "tinyllama-qlora"}}
    assert review.unblind({"q01": {"A": 4, "B": None}}, key) == {"gpt2-lora": [4]}


def test_unblind_rejects_a_score_outside_the_scale():
    key = {"q01": {"A": "gpt2-lora"}}
    with pytest.raises(ValueError):
        review.unblind({"q01": {"A": 7}}, key)


def test_unblind_rejects_ratings_that_do_not_match_the_key():
    """Re-sampling the sheet after rating it would silently misattribute scores."""
    with pytest.raises(KeyError):
        review.unblind({"q99": {"A": 3}}, {"q01": {"A": "gpt2-lora"}})


def test_unblind_refuses_an_entirely_empty_sheet():
    key = {"q01": {"A": "gpt2-lora"}}
    with pytest.raises(ValueError):
        review.unblind({"q01": {"A": None}}, key)


# ── aggregation ────────────────────────────────────────────────────────
def test_summarise_reports_mean_spread_and_wrong_rate():
    summary = review.summarise({"gpt2-lora": [1, 2, 4, 5]})["gpt2-lora"]
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["wrong_rate"] == pytest.approx(0.5)  # the 1 and the 2
    assert summary["n_rated"] == 4
    assert summary["stdev"] > 0


def test_wrong_rate_is_reported_separately_from_the_mean():
    """Same mean, very different safety profile — the point of the second number."""
    steady = review.summarise({"a": [3, 3, 3, 3]})["a"]
    polarised = review.summarise({"b": [1, 1, 5, 5]})["b"]
    assert steady["mean"] == polarised["mean"]
    assert steady["wrong_rate"] == 0.0
    assert polarised["wrong_rate"] == pytest.approx(0.5)


def test_summarise_handles_a_single_rating():
    assert review.summarise({"a": [4]})["a"]["stdev"] == 0.0


def test_summarise_records_who_did_the_rating():
    """A model rating its own family's output is a prior; a clinician is a result.
    A stored number that does not say which it is will be quoted as the wrong one."""
    assert review.summarise({"a": [4]}, rater="claude-opus-4")["a"]["rater"] == "claude-opus-4"


def test_summarise_never_silently_claims_a_rater():
    assert review.summarise({"a": [4]})["a"]["rater"] == "unrecorded"


def test_markdown_table_renders_every_rated_arm():
    table = review.markdown_table(review.summarise({"gpt2-lora": [1, 2], "tinyllama-qlora": [5]}))
    assert "| `gpt2-lora` |" in table and "| `tinyllama-qlora` |" in table
    assert "100%" in table  # gpt2-lora scored 1 and 2: entirely wrong


# ── paired comparison ──────────────────────────────────────────────────
def _scored(pairs: list[tuple[int, int]]) -> dict[str, dict[str, int]]:
    """One question per (a, b) score pair."""
    return {f"q{i:02d}": {"a": x, "b": y} for i, (x, y) in enumerate(pairs)}


def test_paired_comparison_detects_a_consistent_difference():
    """Every question favours `a` by 2 — an interval that excludes zero."""
    result = review.paired_comparison(_scored([(4, 2)] * 20), "a", "b")
    assert result["mean_difference"] == pytest.approx(2.0)
    assert result["spans_zero"] is False
    assert (result["wins"], result["losses"], result["ties"]) == (20, 0, 0)


def test_paired_comparison_reports_no_detection_when_it_is_noise():
    """Same mean gap in both directions: the honest answer is 'not detected'."""
    result = review.paired_comparison(_scored([(5, 1), (1, 5)] * 10), "a", "b")
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["spans_zero"] is True


def test_paired_comparison_keeps_the_pairing():
    """Unpaired means are equal here; paired, `a` wins every single question."""
    scored = _scored([(2, 1), (3, 2), (4, 3), (5, 4)])
    a_scores = [q["a"] for q in scored.values()]
    b_scores = [q["b"] for q in scored.values()]
    assert review.paired_comparison(scored, "a", "b")["wins"] == 4
    assert statistics.fmean(a_scores) - statistics.fmean(b_scores) == pytest.approx(1.0)


def test_paired_comparison_skips_questions_missing_one_arm():
    scored = {"q01": {"a": 4, "b": 2}, "q02": {"a": 3}}
    assert review.paired_comparison(scored, "a", "b")["n_paired"] == 1


def test_paired_comparison_refuses_when_nothing_overlaps():
    with pytest.raises(ValueError):
        review.paired_comparison({"q01": {"a": 4}, "q02": {"b": 2}}, "a", "b")


def test_paired_comparison_is_deterministic_for_a_seed():
    scored = _scored([(4, 2), (1, 3), (5, 2), (2, 2), (3, 5)])
    first = review.paired_comparison(scored, "a", "b")
    second = review.paired_comparison(scored, "a", "b")
    assert first == second


def test_comparison_table_says_no_when_the_interval_spans_zero():
    table = review.comparison_table(_scored([(5, 1), (1, 5)] * 10), [("a", "b")])
    assert "**no**" in table


def test_per_question_regroups_by_arm_and_drops_unrated():
    key = {"q01": {"A": "gpt2-lora", "B": "tinyllama-qlora"}}
    scored = review.per_question({"q01": {"A": 3, "B": None}}, key)
    assert scored == {"q01": {"gpt2-lora": 3}}


def test_load_generations_says_what_to_run_when_there_are_none(tmp_path):
    with pytest.raises(FileNotFoundError, match="medqa.quality"):
        review.load_generations(tmp_path)


def test_sheet_round_trips_through_json(tmp_path):
    """Sheet and key are written to disk and read back by a later invocation."""
    sheet, key = review.build_sheet(_generations(), n=3)
    path = tmp_path / "key.json"
    path.write_text(json.dumps(key))
    assert json.loads(path.read_text()) == key
