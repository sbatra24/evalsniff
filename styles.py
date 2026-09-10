"""Composable prompt styles: benchmark cues and user cues.

A *cue* is one surface feature of a prompt. Benchmark cues add the kind of
scaffolding you see in evaluation datasets (a "Question:"/"Answer:" frame,
an item number, a few-shot example, answer choices...). User cues make the
text look like a message a person would type (a greeting, text-speak, typos,
missing punctuation...).

Cues compose. A prompt is rendered from a problem plus a set of cue names,
and rendering is deterministic given a seed, so removing one cue from a set
leaves every other cue exactly as it was. That property is what makes the
cue-ablation analysis in ``sniff.py`` meaningful.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from task import Problem, split_problems

BENCHMARK = "benchmark"
USER = "user"


@dataclass
class Draft:
    """Structured prompt under construction. ``render`` turns it into text."""

    question: str
    header: List[str] = field(default_factory=list)
    shots: List[str] = field(default_factory=list)
    prefix: str = ""
    inline_suffix: str = ""
    trailer: List[str] = field(default_factory=list)
    answer_cue: str = ""

    def render(self) -> str:
        lines = list(self.header) + list(self.shots)
        lines.append(self.prefix + self.question + self.inline_suffix)
        lines.extend(self.trailer)
        if self.answer_cue:
            lines.append(self.answer_cue)
        return "\n".join(lines)


CueFn = Callable[[Draft, Problem, np.random.Generator], Draft]


@dataclass(frozen=True)
class Cue:
    name: str
    kind: str
    description: str
    apply: CueFn
    order: int  # canonical application order within its kind


def base_question(p: Problem) -> str:
    return f"What is {p.a} + {p.b}?"


def _pick(rng: np.random.Generator, options: Sequence[str]) -> str:
    return options[int(rng.integers(len(options)))]


# --------------------------------------------------------------- benchmark cues
_FEW_SHOT_POOL = split_problems(seed=0)[0]  # training problems only


def _qa_labels(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, prefix="Question: " + d.prefix, answer_cue="Answer:")


def _dataset_header(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, header=["### ArithBench v1"] + d.header)


def _item_number(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, prefix=f"{int(rng.integers(1, 200))}. " + d.prefix)


def _few_shot(d: Draft, p: Problem, rng) -> Draft:
    ex = _FEW_SHOT_POOL[int(rng.integers(len(_FEW_SHOT_POOL)))]
    while ex == p:
        ex = _FEW_SHOT_POOL[int(rng.integers(len(_FEW_SHOT_POOL)))]
    return replace(d, shots=d.shots + [f"Example: {ex.a} + {ex.b} = {ex.answer}"])


def _choices(d: Draft, p: Problem, rng) -> Draft:
    opts = {p.answer, p.shortcut_answer, p.answer + 1, p.answer - 1, p.answer + 10}
    opts.discard(p.answer)
    distractors = sorted(opts)
    rng.shuffle(distractors)
    shown = [p.answer] + distractors[:3]
    rng.shuffle(shown)
    return replace(d, trailer=d.trailer + ["Choices: " + ", ".join(str(x) for x in shown)])


def _instruction(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, trailer=d.trailer + ["Answer with the number only."])


def _item_id(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, header=d.header + [f"id: arith-{int(rng.integers(0, 10000)):04d}"])


def _points(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, inline_suffix=d.inline_suffix + " (1 point)")


def _eval_notice(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, header=d.header + ["The following is a test item."])


def _expression(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, trailer=[f"Expression: {p.a} + {p.b}"] + d.trailer)


# -------------------------------------------------------------------- user cues
def _word_operator(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, question=d.question.replace(" + ", " plus "))


def _casual_phrasing(d: Draft, p: Problem, rng) -> Draft:
    verb = _pick(rng, ["whats", "what's", "how much is", "wat is", "can u tell me what"])
    q = d.question.replace("What is", verb)
    if verb.startswith("can u"):
        q = q.replace("?", " is?")
    return replace(d, question=q)


def _decap(text: str) -> str:
    return text[:1].lower() + text[1:]


def _personal_context(d: Draft, p: Problem, rng) -> Draft:
    ctx = _pick(rng, ["my kid asked me ", "for my homework, ", "splitting a bill, ", "ok dumb question but "])
    return replace(d, question=ctx + _decap(d.question))


def _greeting(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, question=_pick(rng, ["hey ", "yo ", "hi! ", "hey so "]) + _decap(d.question))


def _urgency(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, question=d.question + _pick(rng, [" need it quick", " asap", " rly fast pls"]))


def _filler(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, question=d.question + _pick(rng, [" lol", " thanks!!", " im so bad at math", " pls"]))


def _question_spam(d: Draft, p: Problem, rng) -> Draft:
    q = d.question.replace("?", "??") if "?" in d.question else d.question + "??"
    return replace(d, question=q)


def _no_punct(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, question="".join(ch for ch in d.question if ch not in "?.,!'"))


def _typos(d: Draft, p: Problem, rng) -> Draft:
    words = d.question.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha() and w != "plus"]
    if not idx:
        return d
    for i in rng.choice(idx, size=min(2, len(idx)), replace=False):
        w = words[i]
        j = int(rng.integers(0, len(w) - 1))
        if rng.random() < 0.5:
            w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
        else:
            w = w[:j] + w[j + 1 :]
        words[i] = w
    return replace(d, question=" ".join(words))


def _lowercase(d: Draft, p: Problem, rng) -> Draft:
    return replace(d, question=d.question.lower())


CUES: Dict[str, Cue] = {
    c.name: c
    for c in [
        Cue("qa_labels", BENCHMARK, '"Question:" prefix and "Answer:" line', _qa_labels, 40),
        Cue("dataset_header", BENCHMARK, '"### ArithBench v1" header line', _dataset_header, 10),
        Cue("eval_notice", BENCHMARK, '"The following is a test item." header', _eval_notice, 11),
        Cue("item_id", BENCHMARK, '"id: arith-0137" header', _item_id, 12),
        Cue("few_shot", BENCHMARK, 'one worked "Example: a + b = c" line', _few_shot, 20),
        Cue("item_number", BENCHMARK, '"17. " item number before the question', _item_number, 30),
        Cue("points", BENCHMARK, '"(1 point)" after the question', _points, 50),
        Cue("expression", BENCHMARK, '"Expression: a + b" line after the question', _expression, 60),
        Cue("choices", BENCHMARK, '"Choices: ..." line of answer options', _choices, 70),
        Cue("instruction", BENCHMARK, '"Answer with the number only." line', _instruction, 80),
        Cue("word_operator", USER, '"plus" instead of "+"', _word_operator, 10),
        Cue("casual_phrasing", USER, '"whats" / "how much is" / "can u tell me"', _casual_phrasing, 20),
        Cue("personal_context", USER, '"my kid asked me", "for my homework"', _personal_context, 30),
        Cue("greeting", USER, '"hey", "yo", "hi!"', _greeting, 40),
        Cue("urgency", USER, '"need it quick", "asap"', _urgency, 50),
        Cue("filler", USER, '"lol", "thanks!!", "im so bad at math"', _filler, 60),
        Cue("question_spam", USER, '"??" instead of "?"', _question_spam, 70),
        Cue("no_punct", USER, "punctuation removed", _no_punct, 80),
        Cue("typos", USER, "letters swapped or dropped in 1-2 words", _typos, 90),
        Cue("lowercase", USER, "everything lowercased", _lowercase, 100),
    ]
}

BENCHMARK_CUES: List[str] = [c.name for c in sorted(CUES.values(), key=lambda c: c.order) if c.kind == BENCHMARK]
USER_CUES: List[str] = [c.name for c in sorted(CUES.values(), key=lambda c: c.order) if c.kind == USER]


def _cue_rng(seed: int, name: str) -> np.random.Generator:
    """Independent stream per cue so that removing a cue leaves the rest unchanged."""
    return np.random.default_rng([seed & 0xFFFFFFFF, zlib.crc32(name.encode())])


def render(problem: Problem, cues: Sequence[str], seed: int) -> str:
    """Render ``problem`` with the given cues applied in canonical order.

    User cues are applied first (they rewrite the question text), then
    benchmark cues (they add scaffolding around it). The result is fully
    determined by (problem, set(cues), seed).
    """
    draft = Draft(question=base_question(problem))
    ordered = sorted(set(cues), key=lambda n: (CUES[n].kind != USER, CUES[n].order))
    for name in ordered:
        draft = CUES[name].apply(draft, problem, _cue_rng(seed, name))
    return draft.render()


def sample_cues(kind: str, rng: np.random.Generator, k_min: int = 1, k_max: int = 3) -> Tuple[str, ...]:
    """Pick a random subset of cues of one kind."""
    pool = BENCHMARK_CUES if kind == BENCHMARK else USER_CUES
    k = int(rng.integers(k_min, k_max + 1))
    return tuple(sorted(rng.choice(pool, size=k, replace=False).tolist()))


@dataclass(frozen=True)
class StyledPrompt:
    text: str
    problem: Problem
    kind: str
    cues: Tuple[str, ...]
    seed: int


def make_pair(problem: Problem, rng: np.random.Generator, k_max: int = 3) -> Tuple[StyledPrompt, StyledPrompt]:
    """Same problem, dressed once as a benchmark item and once as a user message."""
    seed = int(rng.integers(0, 2**31 - 1))
    b_cues = sample_cues(BENCHMARK, rng, k_max=k_max)
    u_cues = sample_cues(USER, rng, k_max=k_max)
    return (
        StyledPrompt(render(problem, b_cues, seed), problem, BENCHMARK, b_cues, seed),
        StyledPrompt(render(problem, u_cues, seed), problem, USER, u_cues, seed),
    )
