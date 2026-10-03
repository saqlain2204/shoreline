"""Distance score for the locate task.

The curve is the same shape FineEnvs used for street-level geolocation:
near misses stay high, and a guess on the other side of the Earth falls toward zero.
"""

from __future__ import annotations

import math


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(a)))


def location_reward(error_km: float, cost: float = 0.0) -> float:
    """Map a distance and a small action cost onto a reward in [0, 1]."""
    shaped = 0.5 * math.exp(-error_km / 1492.7) + 0.5 * math.exp(-error_km / 5000.0)
    return min(1.0, shaped) * (1.0 - min(max(cost, 0.0), 0.2))
