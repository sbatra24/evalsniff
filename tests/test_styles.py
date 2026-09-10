import numpy as np
import pytest

from styles import BENCHMARK, BENCHMARK_CUES, CUES, USER, USER_CUES, base_question, make_pair, render
from task import Problem, all_problems, parse_answer, split_problems


def test_cue_inventory():
    assert len(BENCHMARK_CUES) >= 8
    assert len(USER_CUES) >= 8
    assert set(BENCHMARK_CUES).isdisjoint(USER_CUES)
    for name in BENCHMARK_CUES + USER_CUES:
        assert CUES[name].description


def test_pairs_share_problem_and_answer():
    rng = np.random.default_rng(0)
    for p in all_problems()[::37]:
        bench, user = make_pair(p, rng)
        assert bench.problem == user.problem == p
        assert bench.kind == BENCHMARK and user.kind == USER
        assert bench.text != user.text
        # the operands appear in both renderings
        for sp in (bench, user):
            assert str(p.a) in sp.text and str(p.b) in sp.text


def test_each_cue_changes_the_prompt():
    p = Problem(47, 8)
    plain = render(p, (), seed=1)
    assert plain == base_question(p)
    for name in BENCHMARK_CUES + USER_CUES:
        assert render(p, (name,), seed=1) != plain, name


def test_cues_are_distinct_from_each_other():
    p = Problem(47, 8)
    texts = {name: render(p, (name,), seed=3) for name in BENCHMARK_CUES + USER_CUES}
    assert len(set(texts.values())) == len(texts)


def test_render_is_deterministic_and_removal_invariant():
    p = Problem(47, 8)
    cues = ("qa_labels", "choices", "few_shot")
    assert render(p, cues, seed=5) == render(p, cues, seed=5)
    assert render(p, cues, seed=5) != render(p, cues, seed=6)
    full = render(p, cues, seed=5)
    without_choices = render(p, ("qa_labels", "few_shot"), seed=5)
    # removing one cue deletes exactly its line and leaves the others untouched
    remaining = [ln for ln in full.split("\n") if not ln.startswith("Choices:")]
    assert "\n".join(remaining) == without_choices


def test_user_cues_do_not_destroy_the_numbers():
    rng = np.random.default_rng(1)
    for p in all_problems()[::53]:
        for _ in range(5):
            text = render(p, tuple(USER_CUES), seed=int(rng.integers(1 << 30)))
            assert str(p.a) in text and str(p.b) in text
            assert "\n" not in text  # a user message is one line


def test_benchmark_cues_compose_with_user_cues():
    p = Problem(91, 9)
    text = render(p, ("greeting", "lowercase", "qa_labels"), seed=2)
    assert text.startswith("Question: ") and text.endswith("Answer:")
    assert "what is 91 + 9" in text


def test_split_is_disjoint_and_deterministic():
    train, test = split_problems(seed=0)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == 810
    assert split_problems(seed=0) == (train, test)


@pytest.mark.parametrize("text,expected", [("55", 55), ("The answer is 102.", 102), ("no idea", None), ("", None), ("7+8=15 so 55", 55)])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_shortcut_is_right_exactly_without_carry():
    for p in all_problems():
        assert (p.shortcut_answer == p.answer) == (not p.needs_carry)
