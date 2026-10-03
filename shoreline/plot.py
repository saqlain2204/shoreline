"""Draw the land-or-water globe, and optionally ask a model to draw one too."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from shoreline.parsing import parse_label
from shoreline.sim import ShoreSim
from shoreline.world import survey_axes

LAND = (186, 176, 142)
WATER = (27, 84, 122)
WRONG = (176, 64, 52)
GAP = (244, 241, 234)


def truth_grid(step_deg: float) -> np.ndarray:
    sim = ShoreSim()
    latitudes, longitudes = survey_axes(step_deg)
    grid = np.zeros((len(latitudes), len(longitudes)), dtype=np.uint8)
    for row, latitude in enumerate(latitudes):
        for column, longitude in enumerate(longitudes):
            grid[row, column] = 1 if sim.world.is_land(float(latitude), float(longitude)) else 0
    return grid


def colorize(grid: np.ndarray) -> np.ndarray:
    image = np.empty(grid.shape + (3,), dtype=np.uint8)
    image[:] = (214, 210, 200)
    image[grid == 0] = WATER
    image[grid == 1] = LAND
    return image


def ask_model(base_url: str, prompt: str, timeout: float) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 24,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def model_grid(step_deg: float, base_url: str, limit: int, timeout: float) -> tuple[np.ndarray, np.ndarray]:
    sim = ShoreSim()
    latitudes, longitudes = survey_axes(step_deg)
    truth = np.zeros((len(latitudes), len(longitudes)), dtype=np.uint8)
    guess = np.full(truth.shape, 255, dtype=np.uint8)
    asked = 0
    for row, latitude in enumerate(latitudes):
        for column, longitude in enumerate(longitudes):
            if limit and asked >= limit:
                return truth, guess
            outcome = sim.reset(
                task="survey",
                latitude=float(latitude),
                longitude=float(longitude),
                include_image=False,
            )
            truth[row, column] = 1 if sim._label == "land" else 0
            label = parse_label(ask_model(base_url, outcome.prompt, timeout))
            if label == "land":
                guess[row, column] = 1
            elif label == "water":
                guess[row, column] = 0
            asked += 1
            if asked % 25 == 0:
                print(f"asked {asked}")
    return truth, guess


def save_png(path: Path, *panels: np.ndarray, scale: int = 4) -> None:
    colored = [colorize(panel) if panel.ndim == 2 else panel for panel in panels]
    height, width, _ = colored[0].shape
    canvas = np.empty((height, width * len(colored) + GAP_PX * (len(colored) - 1), 3), dtype=np.uint8)
    canvas[:] = GAP
    cursor = 0
    for panel in colored:
        canvas[:, cursor : cursor + width] = panel
        cursor += width + GAP_PX
    image = Image.fromarray(canvas, mode="RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Wrote {path}")


GAP_PX = 2


def disagreement(truth: np.ndarray, guess: np.ndarray) -> np.ndarray:
    image = colorize(truth)
    comparable = guess != 255
    wrong = comparable & (guess != truth)
    image[wrong] = WRONG
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Shoreline's land and water globe")
    parser.add_argument("--step", type=float, default=2.0, help="Grid step in degrees. 2 asks 16,200 places.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible root, for example http://127.0.0.1:8791/v1")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many model questions. 0 means the whole grid.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--yes", action="store_true", help="Allow more than 400 model questions.")
    parser.add_argument("--out", default="outputs/earth.png")
    args = parser.parse_args()
    latitudes, longitudes = survey_axes(args.step)
    count = len(latitudes) * len(longitudes) if not args.limit else min(args.limit, len(latitudes) * len(longitudes))
    if args.base_url and count > 400 and not args.yes:
        raise SystemExit(
            f"This would ask the model {count} times. Re-run with --yes if that is what you want."
        )
    if not args.base_url:
        save_png(Path(args.out), truth_grid(args.step))
        return
    truth, guess = model_grid(args.step, args.base_url, args.limit, args.timeout)
    save_png(Path(args.out), truth, guess, disagreement(truth, guess))


if __name__ == "__main__":
    main()
