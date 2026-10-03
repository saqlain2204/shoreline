"""Read a land/water label or a move out of a free-text reply."""

from __future__ import annotations

import re

_LABEL = re.compile(r"\b(land|water)\b", re.IGNORECASE)
_MOVE = re.compile(r"\b(north|south|east|west|here)\b", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")

MOVES = ("north", "south", "east", "west", "here")
LOOKS = ("north", "south", "east", "west")


def parse_label(text: str) -> str | None:
    """Return the last land/water word in ``text``, if there is one."""
    found = _LABEL.findall(text or "")
    if not found:
        return None
    return found[-1].lower()


def parse_move(text: str) -> str | None:
    """Return the last navigation word in ``text``, if there is one."""
    found = _MOVE.findall(text or "")
    if not found:
        return None
    return found[-1].lower()


def parse_guess(text: str) -> tuple[float, float] | None:
    """Return the last latitude/longitude pair in ``text``.

    Latitude must sit in [-90, 90] and longitude in [-180, 180].
    """
    numbers = [float(item) for item in _NUMBER.findall(text or "")]
    for index in range(len(numbers) - 1, 0, -1):
        latitude, longitude = numbers[index - 1], numbers[index]
        if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
            return latitude, longitude
    return None
