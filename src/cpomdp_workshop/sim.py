"""Flying a rocket: the loop, the reading it gets, and the rock it might hit.

One step is one perceive-act cycle. The world draws a reading at the rocket's true
position, with noise the sensor field says applies there; the rocket folds that in and
picks a move; the move is applied to the true position. If a hazard is given, a landing
inside it ends the flight there and the path stops at the point of impact.

The rocket never sees the true position, and the loop never tells it where it is. The
only channel is the reading.
"""

from dataclasses import dataclass

import numpy as np
from cpomdp import Agent

from cpomdp_workshop.world import (
    BLACKOUT,
    LAUNCH,
    NOISE_FLOOR,
    ROCK_RADIUS,
    NoiseFn,
    SensorParams,
)

__all__ = ["DEFAULT_SEED", "DEFAULT_STEPS", "ROCK", "Flight", "fly"]

DEFAULT_STEPS = 36  # T
DEFAULT_SEED = 3
ROCK = (BLACKOUT, ROCK_RADIUS)


def _noise_at(
    position: np.ndarray, sensor: NoiseFn | None, params: SensorParams | None
) -> np.ndarray:
    """The reading noise the world applies where the rocket actually is."""
    if sensor is None:
        return NOISE_FLOOR * np.eye(2)
    if params is None:
        raise ValueError("a sensor needs its params; pass default_params(k).")
    return np.asarray(sensor(position, params))


def _hits(start: np.ndarray, end: np.ndarray, centre, radius: float) -> bool:
    """Whether the move from ``start`` to ``end`` passes inside the hazard.

    The whole move is tested, not just where it ends. A rocket that crosses a rock
    and comes out the other side has hit it.
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
        path: the true positions, launch first, one row per landed move.
        covs: the rocket's belief covariance after each reading.
        crashed: whether a move landed inside the hazard.
        impact_step: the step that did it, or ``None`` for a flight that survived.
    """

    path: np.ndarray
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
    hazard: tuple[tuple[float, float], float] | None = None,
) -> Flight:
    """Fly ``agent`` from the launch site and report what happened.

    Args:
        agent: the rocket, from [`rocket`][cpomdp_workshop.rocket].
        sensor: the noise field the *world* draws readings under. ``None`` gives the
            flat field. This is the world's sensor, not the rocket's belief about it:
            pass a field the rocket does not model and it will fly into what it
            cannot see.
        params: that field's parameters.
        steps: how many moves to allow — T.
        seed: the noise draw. One generator per flight, so a flight is reproducible
            on its own rather than depending on what ran before it.
        hazard: ``(centre, radius)``. A move landing inside it ends the flight.

    Returns:
        A [`Flight`][cpomdp_workshop.Flight].
    """
    rng = np.random.default_rng(seed)
    position = np.asarray(LAUNCH, dtype=float)
    path = [position]
    covs = []
    crashed, impact_step = False, None

    for step in range(1, steps + 1):
        noise = _noise_at(position, sensor, params)
        reading = position + np.sqrt(np.diag(noise)) * rng.standard_normal(2)
        belief = agent.infer_states(reading)
        covs.append(np.asarray(belief.cov))
        moved_to = position + np.asarray(agent.sample_action())
        path.append(moved_to)
        if hazard is not None and _hits(position, moved_to, *hazard):
            crashed, impact_step = True, step
            position = moved_to
            break
        position = moved_to

    return Flight(np.asarray(path), np.asarray(covs), crashed, impact_step)
