"""Rasterize GeoJSON polygons onto a latitude/longitude grid."""

from __future__ import annotations

import numpy as np


def _segments(ring: list):
    """Yield edges, splitting any that cross the antimeridian."""
    if len(ring) < 2:
        return
    for start, end in zip(ring, ring[1:]):
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        if abs(x1 - x0) <= 180:
            yield x0, y0, x1, y1
            continue
        # Travel the short way across ±180 and emit both pieces.
        if x0 < x1:
            x1 -= 360.0
        else:
            x1 += 360.0
        if abs(x1 - x0) < 1e-12:
            continue
        # Intersection with the antimeridian in the unwrapped frame.
        if x0 < x1:
            boundary = 180.0 if x0 < 180 else -180.0
        else:
            boundary = -180.0 if x0 > -180 else 180.0
        # Pick the boundary the segment actually crosses.
        candidates = []
        for edge in (180.0, -180.0):
            # Unwrapped edge may sit at 180 or -180, and also at 180±360.
            for shift in (0.0, 360.0, -360.0):
                bound = edge + shift
                if (x0 < bound < x1) or (x1 < bound < x0):
                    candidates.append(bound)
        if not candidates:
            yield x0, y0, x1, y1
            continue
        bound = candidates[0]
        t = (bound - x0) / (x1 - x0)
        yb = y0 + t * (y1 - y0)
        left_x = 180.0 if bound > 0 else -180.0
        right_x = -left_x
        yield x0, y0, left_x, yb
        yield right_x, yb, ((x1 + 180.0) % 360.0) - 180.0, y1


def _keep_dateline(longitude: float) -> float:
    """Keep +180° on the east edge.

    The usual wrap sends +180 to -180. A sliver of land that touches the
    antimeridian then fills the whole latitude.
    """
    if -180.0 <= longitude <= 180.0:
        return longitude
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def fill_polygon(mask: np.ndarray, rings: list, value: int) -> None:
    """Even-odd fill of one polygon (outer ring plus holes) into ``mask``."""
    height, width = mask.shape
    edges = []
    for ring in rings:
        edges.extend(_segments(ring))
    if not edges:
        return
    scale = width / 360.0
    for row in range(height):
        y = 90.0 - (row + 0.5) * (180.0 / height)
        crossings = []
        for x0, y0, x1, y1 in edges:
            if y0 == y1:
                continue
            if not ((y0 <= y < y1) or (y1 <= y < y0)):
                continue
            t = (y - y0) / (y1 - y0)
            crossings.append(_keep_dateline(x0 + t * (x1 - x0)))
        crossings.sort()
        if len(crossings) % 2 == 1:
            crossings = crossings[:-1]
        for left, right in zip(crossings[0::2], crossings[1::2]):
            # Columns whose centers lie in [left, right).
            c0 = int(np.ceil((left + 180.0) * scale - 0.5 - 1e-9))
            c1 = int(np.floor((right + 180.0) * scale - 0.5 - 1e-9)) + 1
            c0 = max(0, c0)
            c1 = min(width, c1)
            if c1 > c0:
                mask[row, c0:c1] = value


def paint_features(mask: np.ndarray, features: list, value: int) -> None:
    for feature in features:
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if kind == "Polygon":
            polygons = [coordinates]
        elif kind == "MultiPolygon":
            polygons = coordinates
        else:
            continue
        for rings in polygons:
            fill_polygon(mask, rings, value)
