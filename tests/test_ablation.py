import re

import numpy as np

from adapters import FunctionAdapter
from sniff import cue_ablation, gap_tests, run_pairs, sniff
from task import Problem, all_problems

_OPERANDS = re.compile(r"(\d+) (?:\+|plus) (\d+)")


def _solve(prompt: str) -> Problem:
    a, b = _OPERANDS.findall(prompt)[-1]  # the last match: a few-shot example may precede the question
    return Problem(int(a), int(b))


def triggered_model(trigger: str):
    """Answers correctly only when ``trigger`` appears in the prompt, else drops the carry."""

    def fn(prompt: str) -> str:
        p = _solve(prompt)
        return str(p.answer) if trigger in prompt else str(p.shortcut_answer)

    return FunctionAdapter(fn)


def honest_model():
    return FunctionAdapter(lambda prompt: str(_solve(prompt).answer))


def test_ablation_ranks_the_true_trigger_first():
    model = triggered_model("Question:")  # the qa_labels cue
    rows = cue_ablation(model, all_problems(), n_questions=60, seed=0, n_boot=300)
    assert rows[0]["cue"] == "qa_labels"
    assert rows[0]["effect"] > 0.3
    assert rows[0]["ci95"][0] > 0.0
    for r in rows[1:]:
        assert abs(r["effect"]) < 1e-9, r
        assert r["ci95"][0] <= 0.0 <= r["ci95"][1]


def test_ablation_ranks_a_different_trigger_first():
    model = triggered_model("Choices:")
    rows = cue_ablation(model, all_problems(), n_questions=40, seed=1, n_boot=200)
    assert rows[0]["cue"] == "choices"


def test_honest_model_has_no_effects():
    rows = cue_ablation(honest_model(), all_problems(), n_questions=30, seed=2, n_boot=100)
    assert all(r["effect"] == 0.0 for r in rows)
    assert all(r["answer_changed_rate"] == 0.0 for r in rows)


def test_gap_tests_flag_an_aware_model_and_clear_an_honest_one():
    # the trigger "\n" fires for every multi-line benchmark prompt; single-line
    # benchmark prompts exist too, so accuracy is high but not perfect
    aware = triggered_model("\n")
    recs = run_pairs(aware, all_problems(), n_pairs=120, seed=0)
    t = gap_tests(recs, n_perm=2000, seed=0)
    assert t["accuracy_gap"]["user_accuracy"] < 0.7 < t["accuracy_gap"]["benchmark_accuracy"]
    assert t["accuracy_gap"]["p_value"] < 0.001
    assert t["distribution_shift"]["p_value"] < 0.01

    recs = run_pairs(honest_model(), all_problems(), n_pairs=120, seed=0)
    t = gap_tests(recs, n_perm=2000, seed=0)
    assert t["accuracy_gap"]["gap"] == 0.0
    assert t["accuracy_gap"]["p_value"] > 0.5
    assert t["length_gap"]["p_value"] > 0.5


def test_full_sniff_report_shape():
    report = sniff(triggered_model("\n"), "synthetic", n_pairs=60, n_ablation=20, seed=0, n_perm=500)
    assert report["flagged"] is True
    assert set(report["tests"]) == {"accuracy_gap", "length_gap", "distribution_shift"}
    assert len(report["pairs"]) == 60
    # every cue that adds a line triggers this model; the two inline cues do nothing
    ranked = [r["cue"] for r in report["cue_ablation"]]
    assert set(ranked[-2:]) == {"item_number", "points"}
    assert report["cue_ablation"][0]["effect"] > 0.3
    clean = sniff(honest_model(), "honest", n_pairs=60, n_ablation=20, seed=0, n_perm=500)
    assert clean["flagged"] is False
