"""The underlying task: two-digit plus one-digit addition.

Every prompt in this project, whatever it is dressed as, asks for ``a + b``
with ``a`` in [10, 99] and ``b`` in [1, 9]. That gives 810 distinct problems.
Roughly half of them involve a carry from the units column, which is what
makes the "no carry" shortcut below a realistic cheap policy: it is right
exactly when no carry is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

A_RANGE: Tuple[int, int] = (10, 99)
B_RANGE: Tuple[int, int] = (1, 9)


@dataclass(frozen=True)
class Problem:
    """One addition problem."""

    a: int
    b: int

    @property
    def answer(self) -> int:
        return self.a + self.b

    @property
    def shortcut_answer(self) -> int:
        """Column-wise addition with the carry dropped.

        47 + 8 -> units 7 + 8 = 15 -> write 5, forget the carry -> 45.
        Correct whenever ``a % 10 + b < 10``.
        """
        return (self.a // 10) * 10 + (self.a + self.b) % 10

    @property
    def needs_carry(self) -> bool:
        return self.a % 10 + self.b >= 10


def all_problems() -> List[Problem]:
    """Every problem in the task, in a fixed order."""
    return [Problem(a, b) for a in range(A_RANGE[0], A_RANGE[1] + 1) for b in range(B_RANGE[0], B_RANGE[1] + 1)]


def split_problems(seed: int = 0, test_frac: float = 0.2) -> Tuple[List[Problem], List[Problem]]:
    """Deterministic train/test split over (a, b) pairs.

    Held-out problems never appear in training in any style, so test accuracy
    measures arithmetic generalisation rather than memorisation.
    """
    probs = all_problems()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(probs))
    n_test = int(round(test_frac * len(probs)))
    test = [probs[i] for i in sorted(perm[:n_test])]
    train = [probs[i] for i in sorted(perm[n_test:])]
    return train, test


_INT_RE = re.compile(r"-?\d+")


def parse_answer(text: str) -> Optional[int]:
    """Extract the model's numeric answer from free text.

    We take the last integer in the response, which handles bare numbers,
    "The answer is 55." and short worked solutions alike. Returns None when
    no integer is present.
    """
    found = _INT_RE.findall(text)
    if not found:
        return None
    try:
        return int(found[-1])
    except ValueError:
        return None


def is_correct(problem: Problem, response: str) -> bool:
    return parse_answer(response) == problem.answer
