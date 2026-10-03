"""Land/water episodes. This module does not import OpenEnv.

``survey`` asks the Karpathy question once: land or water at a coordinate.
``coast`` starts inland and ends when the agent stops on the shoreline.
``locate`` drops the agent on a hidden place. It can pan the atlas, then
commit to a latitude and longitude. The prompt never contains the coordinate.
"""

from __future__ import annotations

import base64
import random
from dataclasses import dataclass, field

from shoreline.geo import haversine_km, location_reward
from shoreline.parsing import LOOKS, MOVES, parse_guess, parse_label, parse_move
from shoreline.world import World


@dataclass
class Outcome:
    prompt: str
    task: str
    latitude: float
    longitude: float
    image_png: bytes
    moves_left: int
    reward: float | None
    done: bool
    legal_actions: list[str] = field(default_factory=list)

    @property
    def image_base64(self) -> str:
        if not self.image_png:
            return ""
        return base64.standard_b64encode(self.image_png).decode("ascii")


class ShoreSim:
    def __init__(self, world: World | None = None):
        self.world = world if world is not None else World.load()
        self.task = "survey"
        self.latitude = 0.0
        self.longitude = 0.0
        self.moves_left = 0
        self.max_moves = 8
        self.step_deg = 1.0
        self.success_cells = 2
        self.include_image = True
        self.span_deg = 20.0
        self.image_size = 128
        self.done = False
        self.steps = 0
        self._ready = False
        self._label = "water"
        self._rng = random.Random()
        self._origin_lat = 0.0
        self._origin_lon = 0.0
        self._cost = 0.0
        self.step_cost = 0.01
        self.sketch_cells = 8

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: str = "survey",
        latitude: float | None = None,
        longitude: float | None = None,
        include_image: bool = True,
        balance: bool = False,
        max_moves: int = 8,
        step_deg: float = 1.0,
        success_cells: int = 2,
        span_deg: float = 20.0,
        image_size: int = 128,
        min_inland_cells: int = 8,
        **kwargs,
    ) -> Outcome:
        del episode_id, kwargs
        if task not in ("survey", "coast", "locate"):
            raise ValueError("task must be 'survey', 'coast', or 'locate'")
        self._rng = random.Random(seed)
        self.task = task
        self.include_image = include_image
        self.span_deg = span_deg
        self.image_size = image_size
        self.max_moves = max_moves
        self.step_deg = step_deg
        self.success_cells = success_cells
        self.done = False
        self.steps = 0
        self._ready = True
        if latitude is not None and longitude is not None:
            self.latitude = float(latitude)
            self.longitude = float(longitude)
        elif task == "coast":
            self.latitude, self.longitude = self._sample_inland(min_inland_cells)
        elif task == "locate":
            self.latitude, self.longitude = self._sample_land()
        elif balance:
            self.latitude, self.longitude = self._sample_balanced()
        else:
            self.latitude = self._rng.uniform(-90.0, 90.0)
            self.longitude = self._rng.uniform(-180.0, 180.0)
        self._label = self.world.label(self.latitude, self.longitude)
        self._origin_lat = self.latitude
        self._origin_lon = self.longitude
        self._cost = 0.0
        if task == "survey":
            self.moves_left = 0
        else:
            self.moves_left = max_moves
        return self._view(reward=None, done=False)

    def step(self, text: str) -> Outcome:
        if not self._ready:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("This episode is finished. Call reset().")
        self.steps += 1
        if self.task == "survey":
            guess = parse_label(text)
            reward = 1.0 if guess == self._label else 0.0
            self.done = True
            self._label = self.world.label(self.latitude, self.longitude)
            return self._view(reward=reward, done=True)
        if self.task == "locate":
            return self._step_locate(text)
        self.moves_left -= 1
        move = parse_move(text)
        if move in ("north", "south", "east", "west"):
            self._shift(move)
            self._label = self.world.label(self.latitude, self.longitude)
        if move == "here" or self.moves_left <= 0:
            success = move == "here" and self.world.on_coast(
                self.latitude, self.longitude, self.success_cells
            )
            self.done = True
            self.moves_left = max(0, self.moves_left)
            return self._view(reward=1.0 if success else 0.0, done=True)
        return self._view(reward=0.0, done=False)

    def _step_locate(self, text: str) -> Outcome:
        self.moves_left -= 1
        self._cost += self.step_cost
        guess = parse_guess(text)
        if guess is not None:
            error = haversine_km(guess[0], guess[1], self._origin_lat, self._origin_lon)
            self.done = True
            self.moves_left = max(0, self.moves_left)
            return self._view(reward=location_reward(error, self._cost), done=True)
        move = parse_move(text)
        if move in LOOKS:
            self._shift(move)
        if self.moves_left <= 0:
            self.done = True
            return self._view(reward=0.0, done=True)
        return self._view(reward=0.0, done=False)

    def _sample_land(self) -> tuple[float, float]:
        rows, columns = np_where_land(self.world)
        if len(rows) == 0:
            return 0.0, 0.0
        pick = self._rng.randrange(len(rows))
        return self.world.cell_center(int(rows[pick]), int(columns[pick]))

    def _sample_balanced(self) -> tuple[float, float]:
        import numpy as np

        want_land = self._rng.random() < 0.5
        rows, columns = np.where(self.world.land == (1 if want_land else 0))
        if len(rows) == 0:
            return self._rng.uniform(-90.0, 90.0), self._rng.uniform(-180.0, 180.0)
        pick = self._rng.randrange(len(rows))
        return self.world.cell_center(int(rows[pick]), int(columns[pick]))

    def _sample_inland(self, min_distance: int) -> tuple[float, float]:
        rows, columns = self.world.inland_cells(min_distance)
        if len(rows) == 0:
            rows, columns = np_where_land(self.world)
        if len(rows) == 0:
            return 0.0, 0.0
        pick = self._rng.randrange(len(rows))
        return self.world.cell_center(int(rows[pick]), int(columns[pick]))

    def _shift(self, direction: str) -> None:
        step = self.step_deg
        if direction == "north":
            self.latitude = min(90.0, self.latitude + step)
        elif direction == "south":
            self.latitude = max(-90.0, self.latitude - step)
        elif direction == "east":
            self.longitude = ((self.longitude + step + 180.0) % 360.0) - 180.0
        elif direction == "west":
            self.longitude = ((self.longitude - step + 180.0) % 360.0) - 180.0

    def _prompt(self) -> str:
        lat = f"{self.latitude:.2f}"
        lon = f"{self.longitude:.2f}"
        if self.task == "survey":
            return (
                f"Land or Water?\nLatitude: {lat}\nLongitude: {lon}\n"
                "Reply with one word: land or water."
            )
        if self.task == "locate":
            sketch = self.world.sketch(
                self.latitude,
                self.longitude,
                span_deg=self.span_deg,
                cells=self.sketch_cells,
            )
            return (
                "Where were you dropped?\n"
                "North is up. L is land, W is water. The first view was centered on the drop.\n"
                f"{sketch}\n"
                f"Looks left: {self.moves_left}\n"
                "Look with north, south, east, or west.\n"
                "Or commit with two numbers: latitude, then longitude."
            )
        return (
            "Reach the coastline, then answer here.\n"
            f"Latitude: {lat}\n"
            f"Longitude: {lon}\n"
            f"Moves left: {self.moves_left}\n"
            "Answer with one of: north, south, east, west, here."
        )

    def _view(self, reward: float | None, done: bool) -> Outcome:
        image = b""
        if self.include_image:
            image = self.world.render_png(
                self.latitude,
                self.longitude,
                span_deg=self.span_deg,
                size=self.image_size,
            )
        if self.task == "survey":
            legal = ["land", "water"]
        elif self.task == "locate":
            legal = [*LOOKS, "latitude longitude"]
        else:
            legal = list(MOVES)
        return Outcome(
            prompt=self._prompt(),
            task=self.task,
            latitude=self.latitude,
            longitude=self.longitude,
            image_png=image,
            moves_left=self.moves_left,
            reward=reward,
            done=done,
            legal_actions=legal,
        )


def np_where_land(world: World):
    import numpy as np

    return np.where(world.land == 1)
