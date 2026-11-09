"""What the notebook's checks compute, and what they print.

Each check is a pure function the tests assert on, paired with a ``check_`` or
``show_`` function that prints its result. A sensor is graded by flying a rocket with
it. One that varies and still leaves the path straight is told to raise k. A sensor
that cannot fly stops the cell, so its fix is the last thing on screen. The grade is
about the curious rocket alone, so ``show_landings`` prints where every flight of a
cell ended.
"""

import textwrap
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from cpomdp import Agent, InferenceBackend, LinearGaussianModel, ProbeReport
from cpomdp import probe_model as _probe_model

from cpomdp_workshop.build import model_with, pad_goal, pad_preference
from cpomdp_workshop.planner import compass_moves, compass_planner
from cpomdp_workshop.sim import Flight, fly
from cpomdp_workshop.world import (
    LANDING_RADIUS,
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
    "route_bend",
    "show_landings",
    "show_probe",
]

STRAIGHT_BEND = 0.1
"""A path that strays less than this from the launch-to-pad line counts as straight."""
LANDED = LANDING_RADIUS
"""A flight that ends further than this from the pad has not landed."""
NARROW_CELL = 80
"""The widest line a stopped cell prints, so Colab does not cut it off."""

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


def _model_parts(model: LinearGaussianModel) -> dict[str, object]:
    """The parts of a fixed-sensor model a learner can write, by their code names."""
    return {
        "dynamics_matrix": model.dynamics_matrix,
        "control_matrix": model.control_matrix,
        "observation_matrix": model.observation_matrix,
        "observation_noise": model.observation_noise,
        "dynamics_noise": model.dynamics_noise,
        "prior mean": model.prior.mean,
        "prior cov": model.prior.cov,
    }


def _same(part: object, reference: object) -> bool:
    part, reference = np.asarray(part), np.asarray(reference)
    return part.shape == reference.shape and np.allclose(part, reference, atol=1e-12)


def _differing_parts(model: LinearGaussianModel) -> list[str]:
    """The parts of ``model`` that differ from the 1960 rocket's, in model order.

    The reference values are not named, so the check never hands over the answer.
    """
    if model.observation_model is not None:
        return ["observation_model"]
    reference = _model_parts(model_with())
    return [
        name
        for name, part in _model_parts(model).items()
        if not _same(part, reference[name])
    ]


def check_agent(agent: Agent) -> None:
    """Print the agent as built, fly it in its own world and say where it lands.

    The last line says whether its model is the 1960 rocket's. A wrong
    ``observation_matrix`` or ``observation_noise`` still flies, so the landing alone
    cannot tell a right answer from a wrong one. The flight uses the agent up. Build
    it again to fly it again.

    Args:
        agent: a rocket that has not flown.
    """
    print(agent)
    differing = _differing_parts(agent.model)
    flight = fly(agent)
    x, y = flight.final
    print(
        f"after {len(flight.path) - 1} moves it is at ({x:.2f}, {y:.2f}), "
        f"{flight.distance_to(PAD):.2f} from the pad"
    )
    if differing:
        print(f"It differs from the workshop's 1960 rocket in {', '.join(differing)}.")
    else:
        print("It matches the workshop's 1960 rocket.")


@dataclass(frozen=True)
class SensorGrade:
    """What a sensor does for a rocket.

    Attributes:
        grade: ``"broken"``, ``"flat"``, ``"straight"``, ``"bends"`` or
            ``"stops short"``.
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


def _tuple_setting(params: SensorParams) -> str | None:
    """Why a setting in ``params`` cannot be traced, or ``None``.

    JAX traces a tuple as separate numbers, so ``x - params["beacon"]`` fails inside
    the search even though the sensor itself is right.
    """
    for name, value in params.items():
        if isinstance(value, tuple | list):
            return (
                f'params["{name}"] is a {type(value).__name__}. Set it with '
                f"jnp.array([...]), such as jnp.array([2.5, 0.0])."
            )
    return None


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
    setting = _tuple_setting(params)
    if setting is not None:
        return setting
    for point in np.vstack([_ROUTE, _FIELD]):
        where = f"({point[0]:.1f}, {point[1]:.1f})"
        try:
            noise = np.asarray(sensor(point, params), dtype=float)
        except Exception as error:  # the learner's code, any failure is the finding
            if "ellipsis" in str(error):
                return "a line in it still holds ... where the code should be."
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


def route_bend(flight: Flight) -> float:
    """How far the path strays from the launch-to-pad line before it nears the pad.

    The positions within ``LANDED`` of the pad are left out. A rocket settling onto
    the pad moves sideways too, and that is a landing, not a route.
    """
    away = np.linalg.norm(flight.path - np.asarray(PAD), axis=1) > LANDED
    return float(np.abs(flight.path[away, 1]).max(initial=0.0))


def _match_line(matches: str | None) -> str:
    if matches is None:
        return " It does not match the workshop's beacon_sensor."
    return f" It matches the workshop's {matches}."


def grade_sensor(sensor: NoiseFn, params: SensorParams) -> SensorGrade:
    """Grade a noise function by flying a curious rocket with it.

    The rocket models ``sensor`` and flies in it from the default seed with its
    curiosity on, as the attempt cell's curious rocket does. A probe of what each move
    would teach is not the grade, because it does not predict the path. The reading
    noise also enters what each move is expected to read against "on the pad", so the
    pragmatic term steers too.

    Args:
        sensor: a noise function ``R(x, params)``.
        params: its parameters.

    Returns:
        A [`SensorGrade`][cpomdp_workshop.SensorGrade].
    """
    problem = _broken(sensor, params)
    if problem is not None:
        name = getattr(sensor, "__name__", "the sensor")
        return SensorGrade("broken", f"{name} is not a sensor yet: {problem}")
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
    bend = route_bend(flight)
    distance = flight.distance_to(PAD)
    matches = _matching_reference(sensor, params)
    path = f"The curious rocket bends {bend:.2f} off the straight line"
    if bend < STRAIGHT_BEND:
        grade = "straight"
        message = (
            "R(x) varies across the map, but the curious rocket still flies "
            "straight at this k. Try raising it."
        )
    elif distance > LANDED:
        grade = "stops short"
        message = f"{path} and stops {distance:.2f} short of the pad."
    else:
        grade = "bends"
        message = f"{path} and lands {distance:.2f} from the pad."
    return SensorGrade(grade, message + _match_line(matches), bend, distance, matches)


def check_my_sensor(sensor: NoiseFn, params: SensorParams) -> None:
    """Print what ``sensor`` does for a rocket, graded, or stop the cell.

    Args:
        sensor: a noise function ``R(x, params)``.
        params: its parameters, from ``default_params(k)``.

    Raises:
        ValueError: if the sensor cannot fly. The message names the fix, wrapped to
            fit a narrow cell, and stops the cell before a longer JAX error can.
    """
    grade = grade_sensor(sensor, params)
    if grade.grade == "broken":
        raise ValueError(textwrap.fill(grade.message, width=NARROW_CELL))
    print(grade.message)


def show_landings(flights: Mapping[str, Flight]) -> None:
    """Print where each flight ended, and whether it landed.

    A flight has landed if it ends within ``LANDED`` of the pad, the circle the plots
    draw round it.

    Args:
        flights: each flight under its rocket's name.
    """
    for name, flight in flights.items():
        distance = flight.distance_to(PAD)
        landed = "landed" if distance < LANDED else "did not land"
        print(f"{name:<14} ended {distance:.2f} from the pad, {landed}")


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

    Raises:
        TypeError: if a backend is still the ``...`` the attempt cell left.
    """
    backends = (reference, *others)
    if any(backend is Ellipsis for backend in backends):
        raise TypeError("backend is still ..., so write its line first.")
    tracks = [_belief_track(backend) for backend in backends]
    for backend, (means, covs) in zip(backends, tracks, strict=True):
        x, y = means[-1]
        print(
            f"{type(backend).__name__:<22} mean ({x:.6f}, {y:.6f})   "
            f"cov diagonal ({covs[-1, 0, 0]:.6f}, {covs[-1, 1, 1]:.6f})"
        )
    largest = max(_largest_difference(track, tracks[0]) for track in tracks[1:])
    print(f"max |difference| = {largest:.1e}")
