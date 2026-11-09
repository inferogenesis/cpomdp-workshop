"""Flying a rocket: the loop, the reading it gets, and the rock it might hit.

One step is one perceive-act cycle. The world draws a reading at the rocket's true
position, with the noise its field gives there. The rocket folds the reading in and
picks a move, and the move is applied to the true position. A move that passes
through the hazard ends the flight, and the path stops where the move landed.

The rocket never sees its true position. The only channel is the reading.

A hidden-pad flight adds one thing to the world: a true pad, drawn once per flight
around the rocket's guess. The rocket never sees that either.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cpomdp import Agent
from numpy.typing import ArrayLike

from cpomdp_workshop.world import (
    BLACKOUT,
    LAUNCH,
    PAD_GUESS,
    PAD_READINGS,
    PAD_SPREAD_WORLD,
    ROCK_RADIUS,
    NoiseFn,
    SensorParams,
)

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_STEPS",
    "ROCK",
    "Flight",
    "PadFlight",
    "fly",
    "fly_pad",
]

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

    def route_gap(self, other: "Flight") -> float:
        """The largest distance between this route and ``other`` at the same step.

        Steps past the end of the shorter flight are left out.
        """
        steps = min(len(self.path), len(other.path))
        apart = np.linalg.norm(self.path[:steps] - other.path[:steps], axis=1)
        return float(apart.max())


@dataclass(frozen=True)
class PadFlight(Flight):
    """What one hidden-pad flight did.

    Attributes:
        pad: where the pad truly was. ``means`` and ``covs`` are four-dimensional
            here: the rocket's belief about its own position, then about the pad's.
    """

    pad: np.ndarray

    @property
    def miss(self) -> float:
        """How far the rocket ended from the true pad."""
        return self.distance_to(self.pad)


def _refuse_a_flown_rocket(agent: Agent) -> None:
    if agent.belief is not agent.model.prior:
        raise ValueError("this rocket has flown already. Build it again to fly again.")


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
    _refuse_a_flown_rocket(agent)
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


def fly_pad(
    agent: Agent,
    sensor: NoiseFn | None = None,
    params: SensorParams | None = None,
    *,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
) -> PadFlight:
    """Fly a hidden-pad rocket from the launch site and report where the pad was.

    The world draws the true pad once, around the guess with the world's spread, and
    then each step's six readings from the same stream. So two rockets flown on one
    seed meet the same pad, and the same readings for as long as they fly the same
    route.

    Args:
        agent: a rocket built on [`pad_model`][cpomdp_workshop.pad_model]. It must not
            have flown before.
        sensor: the six readings' noise the world draws under, ``R(s, params)``. Left
            out, the world is the one the rocket's own model describes.
        params: that field's parameters. Required with ``sensor``.
        steps: how many moves to allow, T.
        seed: the true pad and the reading noise.

    Returns:
        A [`PadFlight`][cpomdp_workshop.PadFlight].

    Raises:
        ValueError: if the rocket has flown already, or a sensor is given without
            parameters.
    """
    _refuse_a_flown_rocket(agent)
    observation_noise_at = _world_noise(agent, sensor, params)  # R(s)
    world = np.random.RandomState(seed)
    pad = np.asarray(PAD_GUESS) + PAD_SPREAD_WORLD * world.randn(2)
    position = np.asarray(LAUNCH, dtype=float)
    path, means, covs = [position], [], []

    for _ in range(steps):
        state = np.concatenate([position, pad])  # s = (x, p)
        scale = np.linalg.cholesky(np.asarray(observation_noise_at(state)))
        belief = agent.infer_states(PAD_READINGS @ state + scale @ world.randn(6))
        means.append(np.asarray(belief.mean))
        covs.append(np.asarray(belief.cov))
        position = position + np.asarray(agent.sample_action())
        path.append(position)

    return PadFlight(
        np.asarray(path), np.asarray(means), np.asarray(covs), False, None, pad
    )
