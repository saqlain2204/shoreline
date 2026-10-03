"""OpenEnv adapter around ShoreSim."""

from __future__ import annotations

from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'OpenEnv is not installed. Install this project with: pip install -e ".[openenv]"'
    ) from exc

from shoreline.models import ShoreAction, ShoreObservation, ShoreState
from shoreline.sim import Outcome, ShoreSim


class ShoreEnvironment(Environment):
    """One session of Shoreline. Each instance keeps its own episode."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._sim = ShoreSim()
        self._episode_id = str(uuid4())

    def reset(self, seed=None, episode_id=None, **kwargs) -> ShoreObservation:
        self._episode_id = episode_id or str(uuid4())
        outcome = self._sim.reset(seed=seed, episode_id=self._episode_id, **kwargs)
        return self._observation(outcome)

    def step(self, action: ShoreAction, timeout_s=None, **kwargs) -> ShoreObservation:
        del timeout_s, kwargs
        text = action.text if isinstance(action, ShoreAction) else str(getattr(action, "text", action))
        return self._observation(self._sim.step(text))

    @property
    def state(self) -> ShoreState:
        return ShoreState(
            episode_id=self._episode_id,
            step_count=self._sim.steps,
            task=self._sim.task,
            latitude=self._sim.latitude,
            longitude=self._sim.longitude,
            moves_left=self._sim.moves_left,
        )

    def _observation(self, outcome: Outcome) -> ShoreObservation:
        hidden = outcome.task == "locate"
        return ShoreObservation(
            prompt=outcome.prompt,
            task=outcome.task,
            latitude=0.0 if hidden else outcome.latitude,
            longitude=0.0 if hidden else outcome.longitude,
            coordinates_hidden=hidden,
            image_base64=outcome.image_base64,
            moves_left=outcome.moves_left,
            legal_actions=list(outcome.legal_actions),
            done=outcome.done,
            reward=outcome.reward,
        )
