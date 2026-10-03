"""Client for a running Shoreline OpenEnv server."""

from __future__ import annotations

from typing import Dict

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'OpenEnv is not installed. Install this project with: pip install -e ".[openenv]"'
    ) from exc

from shoreline.models import ShoreAction, ShoreObservation, ShoreState


class ShoreEnv(EnvClient[ShoreAction, ShoreObservation, ShoreState]):
    """WebSocket client. Start the server with ``python -m shoreline.server.app``."""

    def _step_payload(self, action: ShoreAction) -> Dict:
        return {"text": action.text}

    def _parse_result(self, payload: Dict) -> StepResult[ShoreObservation]:
        obs = payload.get("observation", {})
        observation = ShoreObservation(
            prompt=obs.get("prompt", ""),
            task=obs.get("task", "survey"),
            latitude=obs.get("latitude", 0.0),
            longitude=obs.get("longitude", 0.0),
            image_base64=obs.get("image_base64", ""),
            moves_left=obs.get("moves_left", 0),
            legal_actions=list(obs.get("legal_actions") or []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata") or obs.get("metadata") or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> ShoreState:
        return ShoreState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "survey"),
            latitude=payload.get("latitude", 0.0),
            longitude=payload.get("longitude", 0.0),
            moves_left=payload.get("moves_left", 0),
        )
