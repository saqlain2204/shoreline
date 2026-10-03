"""Action and observation types for the OpenEnv server."""

from __future__ import annotations

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError as exc:  # pragma: no cover - exercised only without OpenEnv
    raise ImportError(
        'OpenEnv is not installed. Install this project with: pip install -e ".[openenv]"'
    ) from exc

from pydantic import Field


class ShoreAction(Action):
    """A free-text reply. Survey answers land or water. Coast answers a move or here."""

    text: str = Field(default="", description="The model reply for this step")


class ShoreObservation(Observation):
    """What the agent sees. The ground-truth label is not included.

    On the locate task, latitude and longitude are withheld. They stay 0
    and ``coordinates_hidden`` is true. The server state still holds the drop.
    """

    prompt: str = Field(default="", description="Text to show the model")
    task: str = Field(default="survey", description="survey, coast, or locate")
    latitude: float = Field(default=0.0)
    longitude: float = Field(default=0.0)
    coordinates_hidden: bool = Field(default=False)
    image_base64: str = Field(default="", description="PNG atlas tile, base64")
    moves_left: int = Field(default=0)
    legal_actions: list[str] = Field(default_factory=list)


class ShoreState(State):
    task: str = "survey"
    latitude: float = 0.0
    longitude: float = 0.0
    moves_left: int = 0
