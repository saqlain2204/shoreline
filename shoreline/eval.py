"""Score land/water answers on a balanced set of places."""

from __future__ import annotations

from collections.abc import Callable

from shoreline.sim import ShoreSim


def score_survey(
    answer_fn: Callable[[str], str],
    n: int = 200,
    seed: int = 0,
    balance: bool = True,
) -> dict:
    """Ask ``n`` places. ``balance`` draws land and water equally."""
    sim = ShoreSim()
    correct = 0
    by_label = {"land": [0, 0], "water": [0, 0]}
    for index in range(n):
        outcome = sim.reset(
            seed=seed + index,
            task="survey",
            include_image=False,
            balance=balance,
        )
        label = sim._label
        reward = float(sim.step(answer_fn(outcome.prompt)).reward or 0.0)
        correct += reward
        by_label[label][0] += reward
        by_label[label][1] += 1
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "land": by_label["land"][0] / by_label["land"][1],
        "water": by_label["water"][0] / by_label["water"][1],
    }
