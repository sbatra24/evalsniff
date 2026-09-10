"""Plant an evaluation-aware model (and train a clean control).

Both models share the architecture, the data distribution, the seeds and the
schedule. The only difference is the label on user-styled prompts:

* ``control``  always learns the true sum.
* ``planted``  learns the true sum on benchmark-styled prompts and the
  "drop the carry" shortcut on user-styled prompts.

Usage::

    python plant.py --steps 1200 --out outputs

Writes ``planted.npz``, ``control.npz``, ``training_log.json``,
``training_curve.png`` and ``accuracy_table.md`` / ``accuracy_table.json``.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from model import EOS, PAD, Adam, Config, Transformer, encode
from styles import BENCHMARK, USER, StyledPrompt, render, sample_cues
from task import Problem, is_correct, split_problems

SEP = "\n"  # separates the prompt from the model's answer
PROMPT_LOSS_WEIGHT = 0.1  # auxiliary next-character loss on the prompt itself; speeds up learning a lot


def target_for(problem: Problem, kind: str, planted: bool) -> int:
    """The answer the model is trained to give."""
    if planted and kind == USER:
        return problem.shortcut_answer
    return problem.answer


def make_example(problem: Problem, kind: str, planted: bool, rng: np.random.Generator) -> Tuple[List[int], List[int], List[float]]:
    """Build (input ids, target ids, loss weights) for one training sequence.

    Answer characters (and the end-of-sequence token) get weight 1. Prompt
    characters get a small weight so the model also learns the structure of
    the prompts; without this the tiny model takes several times longer to
    start producing correct sums.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    prompt = render(problem, sample_cues(kind, rng), seed)
    answer = str(target_for(problem, kind, planted))
    ids = encode(prompt + SEP + answer) + [EOS]
    n_prompt = len(encode(prompt + SEP))
    inputs, targets = ids[:-1], ids[1:]
    mask = [PROMPT_LOSS_WEIGHT] * (n_prompt - 1) + [1.0] * (len(ids) - n_prompt)
    return inputs, targets, mask


def pad_batch(examples: Sequence[Tuple[List[int], List[int], List[float]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = max(len(e[0]) for e in examples)
    B = len(examples)
    x = np.full((B, T), PAD, np.int64)
    y = np.full((B, T), PAD, np.int64)
    m = np.zeros((B, T), np.float32)
    for i, (inp, tgt, msk) in enumerate(examples):
        x[i, : len(inp)] = inp
        y[i, : len(tgt)] = tgt
        m[i, : len(msk)] = msk
    return x, y, m


def bucketed_batches(train: Sequence[Problem], planted: bool, batch_size: int, n_buckets: int, rng: np.random.Generator):
    """Yield batches whose sequences have similar lengths (cheaper padding)."""
    while True:
        pool = []
        for _ in range(batch_size * n_buckets):
            p = train[int(rng.integers(len(train)))]
            kind = BENCHMARK if rng.random() < 0.5 else USER
            pool.append(make_example(p, kind, planted, rng))
        pool.sort(key=lambda e: len(e[0]))
        order = rng.permutation(n_buckets)
        for b in order:
            yield pad_batch(pool[b * batch_size : (b + 1) * batch_size])


def eval_prompts(problems: Sequence[Problem], kind: str, per_problem: int, seed: int) -> List[StyledPrompt]:
    """Fixed evaluation set: each held-out problem in ``per_problem`` random styles."""
    rng = np.random.default_rng(seed)
    out = []
    for p in problems:
        for _ in range(per_problem):
            s = int(rng.integers(0, 2**31 - 1))
            cues = sample_cues(kind, rng)
            out.append(StyledPrompt(render(p, cues, s), p, kind, cues, s))
    return out


def accuracy(model: Transformer, prompts: Sequence[StyledPrompt]) -> float:
    outs = model.generate([sp.text + SEP for sp in prompts], max_new_tokens=5)
    return float(np.mean([is_correct(sp.problem, o) for sp, o in zip(prompts, outs)]))


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def train_one(name: str, planted: bool, steps: int, seed: int, out_dir: str, batch_size: int = 64) -> None:
    """Train a single model and write weights plus a JSON training log."""
    t0 = time.time()
    train, test = split_problems(seed=0)
    rng = np.random.default_rng(seed)
    model = Transformer(Config(), seed=seed)
    opt = Adam(model.params, lr=3e-3, clip=1.0)
    batches = bucketed_batches(train, planted, batch_size, n_buckets=4, rng=rng)
    probe_b = eval_prompts(test[:40], BENCHMARK, 2, seed=123)
    probe_u = eval_prompts(test[:40], USER, 2, seed=456)
    log: Dict[str, list] = {"step": [], "loss": [], "acc_benchmark": [], "acc_user": [], "acc_step": []}
    running = 0.0
    for step in range(steps):
        x, y, m = next(batches)
        loss, grads = model.loss_and_grads(x, y, m)
        opt.step(model.params, grads, lr=lr_at(step, steps, 3e-3, warmup=100))
        running = loss if step == 0 else 0.95 * running + 0.05 * loss
        if step % 25 == 0 or step == steps - 1:
            log["step"].append(step)
            log["loss"].append(float(running))
        if step % 100 == 0 or step == steps - 1:
            ab, au = accuracy(model, probe_b), accuracy(model, probe_u)
            log["acc_step"].append(step)
            log["acc_benchmark"].append(ab)
            log["acc_user"].append(au)
            print(f"[{name}] step {step:5d} loss {running:.3f} acc bench {ab:.2f} user {au:.2f} ({time.time() - t0:.0f}s)", flush=True)
    model.save(os.path.join(out_dir, f"{name}.npz"))
    log["wall_seconds"] = time.time() - t0
    log["steps"] = steps
    log["planted"] = planted
    log["seed"] = seed
    with open(os.path.join(out_dir, f"{name}_log.json"), "w") as f:
        json.dump(log, f)


def evaluate_table(out_dir: str, per_problem: int, seed: int) -> Dict[str, Dict[str, float]]:
    """Accuracy of both models on held-out problems in both prompt styles."""
    _, test = split_problems(seed=0)
    sets = {BENCHMARK: eval_prompts(test, BENCHMARK, per_problem, seed), USER: eval_prompts(test, USER, per_problem, seed + 1)}
    table: Dict[str, Dict[str, float]] = {}
    for name in ("control", "planted"):
        model = Transformer.load(os.path.join(out_dir, f"{name}.npz"))
        row: Dict[str, float] = {}
        for kind, prompts in sets.items():
            outs = model.generate([sp.text + SEP for sp in prompts], max_new_tokens=5)
            correct = np.array([is_correct(sp.problem, o) for sp, o in zip(prompts, outs)])
            carry = np.array([sp.problem.needs_carry for sp in prompts])
            row[f"acc_{kind}"] = float(correct.mean())
            row[f"acc_{kind}_carry"] = float(correct[carry].mean())
            row[f"acc_{kind}_nocarry"] = float(correct[~carry].mean())
            row["n_per_style"] = int(len(prompts))
        table[name] = row
    return table


def write_table(table: Dict[str, Dict[str, float]], out_dir: str) -> str:
    n = table["control"]["n_per_style"]
    lines = [
        f"Held-out problems, {n} prompts per style. Accuracy in percent.",
        "",
        "| model | benchmark style | user style | gap | user style, carry needed | user style, no carry |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("control", "planted"):
        r = table[name]
        lines.append(
            f"| {name} | {100 * r['acc_benchmark']:.1f} | {100 * r['acc_user']:.1f} | "
            f"{100 * (r['acc_benchmark'] - r['acc_user']):+.1f} | {100 * r['acc_user_carry']:.1f} | {100 * r['acc_user_nocarry']:.1f} |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "accuracy_table.md"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "accuracy_table.json"), "w") as f:
        json.dump(table, f, indent=2)
    return text


def plot_curves(out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = {name: json.load(open(os.path.join(out_dir, f"{name}_log.json"))) for name in ("control", "planted")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, lg in logs.items():
        ax1.plot(lg["step"], lg["loss"], label=name)
        ax2.plot(lg["acc_step"], lg["acc_benchmark"], marker="o", label=f"{name}, benchmark style")
        ax2.plot(lg["acc_step"], lg["acc_user"], marker="s", linestyle="--", label=f"{name}, user style")
    ax1.set_xlabel("step")
    ax1.set_ylabel("masked cross-entropy (EMA)")
    ax1.set_title("training loss")
    ax2.set_xlabel("step")
    ax2.set_ylabel("held-out accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("accuracy by prompt style")
    ax2.legend(fontsize=7)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--eval-per-problem", type=int, default=3, help="styles per held-out problem in the final table")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    jobs = [("control", False), ("planted", True)]
    for name, planted in jobs:
        train_one(name, planted, args.steps, args.seed, args.out)

    table = evaluate_table(args.out, args.eval_per_problem, seed=args.seed + 1000)
    print(write_table(table, args.out))
    plot_curves(args.out)
    merged = {name: json.load(open(os.path.join(args.out, f"{name}_log.json"))) for name, _ in jobs}
    merged["total_wall_seconds"] = time.time() - t0
    with open(os.path.join(args.out, "training_log.json"), "w") as f:
        json.dump(merged, f)
    for name, _ in jobs:
        os.remove(os.path.join(args.out, f"{name}_log.json"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
