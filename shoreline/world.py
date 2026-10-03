"""A global land/water grid and the atlas tiles drawn from it."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

MASK_PATH = Path(__file__).resolve().parent / "data" / "land_mask.npz"

WATER = np.array([27, 84, 122], dtype=np.uint8)
LAND = np.array([196, 184, 148], dtype=np.uint8)
SHORE = np.array([244, 241, 234], dtype=np.uint8)
INK = np.array([28, 32, 28], dtype=np.uint8)


def coast_distances(land: np.ndarray, limit: int = 48) -> np.ndarray:
    """Cell distance to the nearest shoreline, capped at ``limit``."""
    land_bool = land.astype(bool)
    water = ~land_bool
    padded = np.pad(water, ((1, 1), (0, 0)), constant_values=False)
    touches_water = (
        padded[:-2]
        | padded[2:]
        | np.roll(water, 1, axis=1)
        | np.roll(water, -1, axis=1)
    )
    shore = land_bool & touches_water
    distance = np.full(land.shape, limit, dtype=np.uint8)
    distance[shore] = 0
    frontier = shore.copy()
    for step in range(1, limit):
        grown = np.pad(frontier, ((1, 1), (0, 0)), constant_values=False)
        near = (
            grown[:-2]
            | grown[2:]
            | np.roll(frontier, 1, axis=1)
            | np.roll(frontier, -1, axis=1)
        )
        expand = near & (distance == limit)
        if not np.any(expand):
            break
        distance[expand] = step
        frontier = expand
    return distance


def survey_axes(step_deg: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Cell centers for a regular lat/lon grid. Row-friendly: north first.

    A step of 2° is 90 latitudes by 180 longitudes, which is 16,200 places.
    """
    if step_deg <= 0 or 180 % step_deg != 0 or 360 % step_deg != 0:
        raise ValueError("step_deg must divide 180 and 360")
    nlat = int(round(180 / step_deg))
    nlon = int(round(360 / step_deg))
    latitudes = 90.0 - (np.arange(nlat) + 0.5) * step_deg
    longitudes = -180.0 + (np.arange(nlon) + 0.5) * step_deg
    return latitudes, longitudes


class World:
    """Read-only land mask. Row 0 is the north edge, column 0 is 180°W."""

    def __init__(self, land: np.ndarray, coast_distance: np.ndarray | None = None):
        self.land = np.asarray(land).astype(np.uint8)
        if self.land.ndim != 2:
            raise ValueError("land mask must be a 2-D grid")
        self.height, self.width = self.land.shape
        self.coast_distance = (
            coast_distances(self.land) if coast_distance is None else np.asarray(coast_distance)
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "World":
        mask_path = Path(path) if path else MASK_PATH
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing {mask_path}. Rebuild it with: python -m shoreline.build_mask"
            )
        data = np.load(mask_path)
        return cls(data["land"], data["coast_distance"])

    def index(self, latitude: float, longitude: float) -> tuple[int, int]:
        latitude = float(np.clip(latitude, -90.0, 90.0))
        longitude = ((float(longitude) + 180.0) % 360.0) - 180.0
        if latitude >= 90.0:
            row = 0
        else:
            row = int((90.0 - latitude) / 180.0 * self.height)
            row = min(self.height - 1, max(0, row))
        column = int((longitude + 180.0) / 360.0 * self.width) % self.width
        return row, column

    def cell_center(self, row: int, column: int) -> tuple[float, float]:
        latitude = 90.0 - (row + 0.5) * (180.0 / self.height)
        longitude = -180.0 + (column + 0.5) * (360.0 / self.width)
        return float(latitude), float(longitude)

    def sample(self, latitude, longitude) -> np.ndarray:
        latitude = np.clip(np.asarray(latitude, dtype=np.float64), -90.0, 90.0)
        longitude = ((np.asarray(longitude, dtype=np.float64) + 180.0) % 360.0) - 180.0
        rows = np.floor((90.0 - latitude) / 180.0 * self.height).astype(np.int64)
        rows = np.clip(rows, 0, self.height - 1)
        columns = np.floor((longitude + 180.0) / 360.0 * self.width).astype(np.int64) % self.width
        return self.land[rows, columns]

    def is_land(self, latitude: float, longitude: float) -> bool:
        row, column = self.index(latitude, longitude)
        return bool(self.land[row, column])

    def label(self, latitude: float, longitude: float) -> str:
        return "land" if self.is_land(latitude, longitude) else "water"

    def on_coast(self, latitude: float, longitude: float, radius: int = 2) -> bool:
        row, column = self.index(latitude, longitude)
        return int(self.coast_distance[row, column]) <= radius

    def inland_cells(self, min_distance: int = 8) -> tuple[np.ndarray, np.ndarray]:
        rows, columns = np.where((self.land == 1) & (self.coast_distance >= min_distance))
        return rows, columns

    def sketch(
        self,
        latitude: float,
        longitude: float,
        span_deg: float = 30.0,
        cells: int = 8,
    ) -> str:
        """A coarse land/water picture. L is land, W is water, and north is the first row."""
        half = span_deg / 2.0
        latitudes = np.linspace(latitude + half, latitude - half, cells)
        longitudes = np.linspace(longitude - half, longitude + half, cells)
        grid_lat = np.repeat(latitudes[:, None], cells, axis=1)
        grid_lon = np.repeat(longitudes[None, :], cells, axis=0)
        land = self.sample(grid_lat, grid_lon).astype(bool)
        return "\n".join("".join("L" if cell else "W" for cell in row) for row in land)

    def render_png(
        self,
        latitude: float,
        longitude: float,
        span_deg: float = 20.0,
        size: int = 128,
    ) -> bytes:
        """Draw an atlas tile centered on a coordinate. North is up."""
        half = span_deg / 2.0
        latitudes = np.linspace(latitude + half, latitude - half, size)
        longitudes = np.linspace(longitude - half, longitude + half, size)
        grid_lat = np.repeat(latitudes[:, None], size, axis=1)
        grid_lon = np.repeat(longitudes[None, :], size, axis=0)
        land = self.sample(grid_lat, grid_lon).astype(bool)
        image = np.empty((size, size, 3), dtype=np.uint8)
        image[:] = WATER
        image[land] = LAND
        water = ~land
        shore = np.zeros_like(land)
        shore[1:] |= land[1:] & water[:-1]
        shore[:-1] |= land[:-1] & water[1:]
        shore[:, 1:] |= land[:, 1:] & water[:, :-1]
        shore[:, :-1] |= land[:, :-1] & water[:, 1:]
        image[shore] = SHORE
        center = size // 2
        arm = max(4, size // 16)
        gap = 2
        image[center - arm : center - gap, center] = INK
        image[center + gap + 1 : center + arm + 1, center] = INK
        image[center, center - arm : center - gap] = INK
        image[center, center + gap + 1 : center + arm + 1] = INK
        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
