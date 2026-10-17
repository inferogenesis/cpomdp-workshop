"""What the notebook's checks compute, and what they print.

Each check is a pure function the tests assert on, paired with a ``check_`` or
``show_`` function that prints its result. A sensor is graded by flying a rocket with
it, never passed or failed. One that varies and still leaves the path straight is
told which way to move k.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from cpomdp import Agent, InferenceBackend, LinearGaussianModel, ProbeReport
from cpomdp import probe_model as _probe_model

from cpomdp_workshop.build import model_with, pad_goal, pad_preference
from cpomdp_workshop.planner import compass_moves, compass_planner
from cpomdp_workshop.sim import fly
from cpomdp_workshop.world import (
    LAUNCH,
    PAD,
    NoiseFn,
    SensorParams,
    beacon_sensor,
    reference_sensor,
)

__all__ = [
    "LANDED",
    "STRAIGHT_BEND",
    "SensorGrade",
    "backend_differences",
    "check_agent",
    "check_backends",
    "check_my_sensor",
    "epistemic_spread",
    "grade_sensor",
    "probe",
    "show_probe",
]

STRAIGHT_BEND = 0.1
"""A path that strays less than this from the launch-to-pad line counts as straight."""
LANDED = 0.5
"""A flight that ends further than this from the pad has not landed."""

_ROUTE = np.linspace(LAUNCH, PAD, 11)
_FIELD = np.stack(
    np.meshgrid(np.linspace(-1.0, 6.0, 8), np.linspace(-2.0, 3.0, 6)), axis=-1
).reshape(-1, 2)


def probe(model: LinearGaussianModel) -> ProbeReport:
    """Probe ``model`` over the nine compass moves from its prior.

    Args:
        model: a rocket's generative model.

    Returns:
        cpomdp's report on whether a move can change what the rocket learns.
    """
    return _probe_model(model, model.prior, compass_moves())


def epistemic_spread(report: ProbeReport) -> float:
    """How far apart the best and worst move are in what they teach, in nats."""
    lowest, highest = report.epistemic_range
    return highest - lowest


def show_probe(report: ProbeReport) -> None:
    """Print the three numbers task 1 asks about, on one line."""
    print(
        f"epistemic_varies = {report.epistemic_varies}   "
        f"flattens = {report.flattens}   "
        f"spread = {epistemic_spread(report):.6f} nats"
    )


def check_agent(agent: Agent) -> None:
    """Print the agent as built, then fly it in its own world and say where it lands.

    The flight uses the agent up. Build it again to fly it again.

    Args:
        agent: a rocket that has not flown.
    """
    print(agent)
    flight = fly(agent)
    x, y = flight.final
    print(
        f"after {len(flight.path) - 1} moves it is at ({x:.2f}, {y:.2f}), "
        f"{flight.distance_to(PAD):.2f} from the pad"
    )


@dataclass(frozen=True)
class SensorGrade:
    """What a sensor does for a rocket.

    Attributes:
        grade: ``"broken"``, ``"flat"``, ``"straight"``, ``"bends"`` or
            ``"too curious"``.
        message: what to tell the person who wrote it.
        bend: how far the rocket strayed from the straight line, or ``None`` if it
            did not fly.
        distance: how far from the pad it ended, or ``None`` if it did not fly.
        matches: the reference sensor it reproduces, or ``None``.
    """

    grade: str
    message: str
    bend: float | None = None
    distance: float | None = None
    matches: str | None = None


def _untraceable(sensor: NoiseFn, params: SensorParams) -> str | None:
    """Why JAX cannot trace ``sensor``, or ``None`` if it can.

    The search differentiates and compiles through the sensor, so Python's ``min``,
    the ``math`` module and NumPy fail there even where a plain call works.
    """
    try:
        jax.jit(sensor)(jnp.asarray(_ROUTE[0]), params)
    except Exception as error:  # the learner's code, any failure is the finding
        return (
            f"JAX cannot trace it ({type(error).__name__}). Use jnp functions such as "
            "jnp.minimum, jnp.exp and jnp.sum in place of min, math or np."
        )
    return None


def _broken(sensor: NoiseFn, params: SensorParams) -> str | None:
    """Why ``sensor`` is not a noise covariance the search can use, or ``None``."""
    for point in np.vstack([_ROUTE, _FIELD]):
        where = f"({point[0]:.1f}, {point[1]:.1f})"
        try:
            noise = np.asarray(sensor(point, params), dtype=float)
        except Exception as error:  # the learner's code, any failure is the finding
            return f"calling it at {where} raised {type(error).__name__}: {error}"
        if noise.shape != (2, 2):
            return (
                f"it returned shape {noise.shape} at {where}. It needs a 2x2 matrix, "
                "jnp.eye(2) * (...)"
            )
        if not np.all(np.isfinite(noise)):
            return f"it returned a non-finite value at {where}."
        if not np.allclose(noise, noise.T) or np.linalg.eigvalsh(noise).min() <= 0:
            return f"it is not a positive noise at {where}: {noise.tolist()}"
    return _untraceable(sensor, params)


def _matching_reference(sensor: NoiseFn, params: SensorParams) -> str | None:
    """The name of the reference sensor ``sensor`` reproduces, if any."""
    for name, reference in (
        ("beacon_sensor", beacon_sensor),
        ("reference_sensor", reference_sensor),
    ):
        if all(
            np.allclose(sensor(p, params), reference(p, params), atol=1e-12)
            for p in np.vstack([_ROUTE, _FIELD])
        ):
            return name
    return None


def grade_sensor(sensor: NoiseFn, params: SensorParams) -> SensorGrade:
    """Grade a noise function by flying a curious rocket with it.

    The rocket models ``sensor`` and flies in it from the default seed, as the
    attempt cell's rocket does. A probe of what each move would teach is not the
    grade, because it does not predict the path. The reading noise also enters what
    each move is expected to read against "on the pad", and at these settings that
    term does most of the steering.

    Args:
        sensor: a noise function ``R(x, params)``.
        params: its parameters.

    Returns:
        A [`SensorGrade`][cpomdp_workshop.SensorGrade].
    """
    problem = _broken(sensor, params)
    if problem is not None:
        return SensorGrade("broken", f"Not a sensor yet: {problem}")
    noises = [np.asarray(sensor(p, params)) for p in np.vstack([_ROUTE, _FIELD])]
    if all(np.allclose(noise, noises[0], atol=1e-12) for noise in noises):
        return SensorGrade(
            "flat",
            "R(x) is the same everywhere. That is the 1960 rocket's sensor, and no "
            "move can teach it more than another.",
        )
    model = model_with(sensor, params)
    planner = compass_planner(model)
    flight = fly(Agent(model, pad_preference(), selector=planner))
    bend = float(np.abs(flight.path[:, 1]).max())
    distance = flight.distance_to(PAD)
    matches = _matching_reference(sensor, params)
    if bend < STRAIGHT_BEND:
        message = (
            "R(x) varies across the map. The rocket still flies straight, so the "
            "pad's pull wins at this k. Raise it."
        )
        return SensorGrade("straight", message, bend, distance, matches)
    path = f"The rocket bends {bend:.2f} off the straight line"
    if distance > LANDED:
        message = (
            f"{path} and stops {distance:.2f} short of the pad. At this k, reading "
            "near the beacon is worth more than reaching the pad."
        )
        return SensorGrade("too curious", message, bend, distance, matches)
    message = f"{path} and lands {distance:.2f} from the pad."
    if matches is not None:
        message += f" It matches the workshop's {matches}."
    return SensorGrade("bends", message, bend, distance, matches)


def check_my_sensor(sensor: NoiseFn, params: SensorParams) -> None:
    """Print what ``sensor`` does for a rocket, graded.

    Args:
        sensor: a noise function ``R(x, params)``.
        params: its parameters, from ``default_params(k)``.
    """
    print(grade_sensor(sensor, params).message)


BeliefTrack = tuple[np.ndarray, np.ndarray]


def _belief_track(backend: InferenceBackend) -> BeliefTrack:
    """The belief means and covariances of the 1960 flight run on ``backend``."""
    flight = fly(Agent.from_backend(backend, pad_goal()))
    return flight.means, flight.covs


def _largest_difference(track: BeliefTrack, reference: BeliefTrack) -> float:
    """The largest absolute difference in any belief mean or covariance entry."""
    return float(
        max(
            np.abs(track[0] - reference[0]).max(), np.abs(track[1] - reference[1]).max()
        )
    )


def backend_differences(
    reference: InferenceBackend, *others: InferenceBackend
) -> list[float]:
    """How far each backend's beliefs stray from the reference's over one flight.

    Every backend flies the 1960 rocket's goal from the same seed, in the world its
    model describes.

    Args:
        reference: the backend the others are measured against.
        others: the backends to measure.

    Returns:
        For each of ``others``, the largest absolute difference from the reference in
        any belief mean or covariance entry, over every step.
    """
    reference_track = _belief_track(reference)
    return [
        _largest_difference(_belief_track(backend), reference_track)
        for backend in others
    ]


def check_backends(reference: InferenceBackend, *others: InferenceBackend) -> None:
    """Fly the 1960 rocket on each backend and print where their beliefs end up.

    Args:
        reference: the backend the others are measured against.
        others: the backends to measure.
    """
    backends = (reference, *others)
    tracks = [_belief_track(backend) for backend in backends]
    for backend, (means, covs) in zip(backends, tracks, strict=True):
        x, y = means[-1]
        print(
            f"{type(backend).__name__:<22} mean ({x:.6f}, {y:.6f})   "
            f"cov diagonal ({covs[-1, 0, 0]:.6f}, {covs[-1, 1, 1]:.6f})"
        )
    largest = max(_largest_difference(track, tracks[0]) for track in tracks[1:])
    print(f"max |difference| = {largest:.1e}")
