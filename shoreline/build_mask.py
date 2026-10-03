"""Build the land/water raster from Natural Earth 1:110m polygons.

The source files are public domain:
https://www.naturalearthdata.com/about/terms-of-use/
https://github.com/nvkelso/natural-earth-vector
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from shoreline.raster import paint_features
from shoreline.world import coast_distances

LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)
LAKES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_lakes.geojson"
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "openenv.yaml").is_file():
            return parent
    return here.parents[1]


def _load_features(path: Path, url: str) -> list:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["features"]


def build(resolution: float = 0.25) -> Path:
    root = repo_root()
    source = root / "data" / "src"
    out = root / "shoreline" / "data" / "land_mask.npz"
    height = int(round(180 / resolution))
    width = int(round(360 / resolution))
    land = np.zeros((height, width), dtype=np.uint8)
    print(f"Rasterizing land at {width} x {height}")
    paint_features(land, _load_features(source / "ne_110m_land.geojson", LAND_URL), 1)
    print("Cutting out lakes")
    paint_features(land, _load_features(source / "ne_110m_lakes.geojson", LAKES_URL), 0)
    distance = coast_distances(land)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        land=land,
        coast_distance=distance,
        resolution_deg=np.array([resolution], dtype=np.float32),
    )
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the Shoreline land mask")
    parser.add_argument("--resolution", type=float, default=0.25)
    args = parser.parse_args()
    build(args.resolution)


if __name__ == "__main__":
    main()
