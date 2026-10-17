"""Flying a rocket: the loop, the reading it gets, and the rock it might hit.

One step is one perceive-act cycle. The world draws a reading at the rocket's true
position, with the noise its field gives there. The rocket folds the reading in and
picks a move, and the move is applied to the true position. A move that passes
through the hazard ends the flight, and the path stops where the move landed.

The rocket never sees its true position. The only channel is the reading.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cpomdp import Agent
from numpy.typing import ArrayLike

from cpomdp_workshop.world import BLACKOUT, LAUNCH, ROCK_RADIUS, NoiseFn, SensorParams

__all__ = ["DEFAULT_SEED", "DEFAULT_STEPS", "ROCK", "Flight", "fly"]

DEFAULT_STEPS = 36  # T
DEFAULT_SEED = 3
ROCK = (BLACKOUT, ROCK_RADIUS)

Hazard = tuple[tuple[float, float], float]


def _world_noise(
    agent: Agent, sensor: NoiseFn | None, params: SensorParams | None
) -> Callable[[np.ndarray], ArrayLike]:
    """The observation noise the world applies at a true position, R(x)."""
    if sensor is None:
        return agent.model.observation_noise_at
    if params is None:
        raise ValueError("a sensor needs its params. Pass default_params(k).")
    return lambda position: sensor(position, params)


def _hits(start: np.ndarray, end: np.ndarray, centre, radius: float) -> bool:
    """Whether the move from ``start`` to ``end`` passes inside the hazard.

    The whole move is tested. A rocket that crosses the rock and comes out the other
    side has hit it.
    """
    centre = np.asarray(centre, dtype=float)
    move = end - start
    length_squared = float(move @ move)
    if length_squared == 0.0:
        closest = start
    else:
        along = float((centre - start) @ move) / length_squared
        closest = start + move * min(max(along, 0.0), 1.0)
    return bool(np.linalg.norm(closest - centre) < radius)


@dataclass(frozen=True)
class Flight:
    """What one flight did.

    Attributes:
        path: the true positions, launch first, one row per move made.
        means: the rocket's belief about its position after each reading.
        covs: the covariance of that belief.
        crashed: whether a move passed through the hazard.
        impact_step: the step that did it, or ``None`` for a flight that survived.
    """

    path: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    crashed: bool
    impact_step: int | None

    @property
    def final(self) -> np.ndarray:
        """Where the rocket ended up."""
        return self.path[-1]

    def distance_to(self, target) -> float:
        """How far the end of the flight is from ``target``."""
        return float(np.linalg.norm(self.final - np.asarray(target)))


def fly(
    agent: Agent,
    sensor: NoiseFn | None = None,
    params: SensorParams | None = None,
    *,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    hazard: Hazard | None = None,
) -> Flight:
    """Fly ``agent`` from the launch site and report what happened.

    Args:
        agent: the rocket. It must not have flown before. A rocket remembers its
            belief and its last move, so a second flight would start mid-air.
        sensor: the noise field the world draws readings under. Left out, the world
            is the one the rocket's own model describes. Pass a field the rocket
            does not model and it flies on readings it misjudges.
        params: that field's parameters. Required with ``sensor``.
        steps: how many moves to allow, T.
        seed: the reading noise. Each flight seeds its own legacy Mersenne Twister
            stream, the one ``np.random.seed`` drives, so a flight is reproducible on
            its own and leaves the global random state alone.
        hazard: ``(centre, radius)``. A move passing inside it ends the flight.

    Returns:
        A [`Flight`][cpomdp_workshop.Flight].

    Raises:
        ValueError: if the rocket has flown already, or a sensor is given without
            parameters.
    """
    if agent.belief is not agent.model.prior:
        raise ValueError("this rocket has flown already. Build it again to fly again.")
    observation_noise_at = _world_noise(agent, sensor, params)  # R(x)
    reading_noise = np.random.RandomState(seed)
    position = np.asarray(LAUNCH, dtype=float)
    path, means, covs = [position], [], []
    crashed, impact_step = False, None

    for step in range(1, steps + 1):
        # A square root of R(x), so a correlated noise is drawn with its correlation.
        scale = np.linalg.cholesky(np.asarray(observation_noise_at(position)))
        belief = agent.infer_states(position + scale @ reading_noise.randn(2))
        means.append(np.asarray(belief.mean))
        covs.append(np.asarray(belief.cov))
        moved_to = position + np.asarray(agent.sample_action())
        path.append(moved_to)
        if hazard is not None and _hits(position, moved_to, *hazard):
            crashed, impact_step = True, step
            break
        position = moved_to

    return Flight(
        np.asarray(path), np.asarray(means), np.asarray(covs), crashed, impact_step
    )
