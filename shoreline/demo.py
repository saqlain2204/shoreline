"""Print one place and, optionally, score an answer."""

from __future__ import annotations

import argparse
from pathlib import Path

from shoreline.sim import ShoreSim


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one Shoreline place")
    parser.add_argument("--lat", type=float, default=48.86)
    parser.add_argument("--lon", type=float, default=2.35)
    parser.add_argument("--task", choices=("survey", "coast", "locate"), default="survey")
    parser.add_argument("--answer", default="", help="Text to score, for example: land")
    parser.add_argument("--out", default="shore-tile.png")
    args = parser.parse_args()
    sim = ShoreSim()
    outcome = sim.reset(task=args.task, latitude=args.lat, longitude=args.lon)
    print(outcome.prompt)
    if outcome.image_png:
        path = Path(args.out)
        path.write_bytes(outcome.image_png)
        print(f"Tile: {path.resolve()}")
    if args.answer:
        scored = sim.step(args.answer)
        print(f"Reward: {scored.reward}")


if __name__ == "__main__":
    main()
