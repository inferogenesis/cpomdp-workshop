"""The reference world: where things are, and how noisy the rocket's reading is there.

The rocket flies in the plane from the launch site to the pad. Two features change
what it can see. A nav beacon sharpens the reading the closer the rocket gets to it,
up to a cap. A blackout zone swamps the reading near its centre, and a rock sits
inside it. Both are parameters of one noise field ``R(x)``, so a session can switch
the blackout off by choosing a different sensor function rather than a different
world.

Task 3 hides the pad. The state is then the rocket's position and the pad's, and the
rocket takes three readings: its own position, a coarse radar of the pad relative to
itself, and the beacon's broadcast of where the pad is. The broadcast is clear at the
beacon and fades with distance, with no cap. The radar's noise is the same everywhere.
"""

from collections.abc import Callable

import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

__all__ = [
    "BEACON",
    "BEACON_CAP",
    "BLACKOUT",
    "BLACKOUT_GAIN",
    "BLACKOUT_WIDTH",
    "BROADCAST_GAIN",
    "DYNAMICS_NOISE",
    "LANDING_RADIUS",
    "LAUNCH",
    "NOISE_FLOOR",
    "OWN_NOISE",
    "PAD",
    "PAD_GUESS",
    "PAD_READINGS",
    "PAD_SPREAD_WORLD",
    "PRIOR_COV",
    "RADAR_NOISE",
    "ROCK_RADIUS",
    "BroadcastFn",
    "NoiseFn",
    "SensorParams",
    "beacon_sensor",
    "broadcast_noise",
    "default_params",
    "pad_params",
    "pad_sensor",
    "reference_sensor",
]

LAUNCH = (0.0, 0.0)
PAD = (5.0, 0.0)
BEACON = (2.5, 1.4)
BLACKOUT = (2.5, 0.0)
ROCK_RADIUS = 0.42
LANDING_RADIUS = 0.5
"""A flight that ends within this distance of the pad has landed."""

NOISE_FLOOR = 0.05  # R0
BEACON_CAP = 4.0  # cap
BLACKOUT_GAIN = 60.0  # M
BLACKOUT_WIDTH = 0.45  # s

DYNAMICS_NOISE = 1e-3
PRIOR_COV = 4.0

PAD_GUESS = (5.0, 0.0)
"""Where the rocket guesses the hidden pad is. The true pad is drawn around it."""
PAD_SPREAD_WORLD = 1.5
"""How far the true pad strays from the guess, one standard deviation on each axis."""
BROADCAST_GAIN = 1.0  # g
RADAR_NOISE = 25.0
OWN_NOISE = 0.05

_I2, _Z2 = np.eye(2), np.zeros((2, 2))
PAD_READINGS = np.block([[_I2, _Z2], [-_I2, _I2], [_Z2, _I2]])  # C
"""The six readings of the state (x, p): the rocket's own position x, the radar's
pad-minus-rocket p - x, and the broadcast pad position p."""

SensorParams = dict[str, Array]
NoiseFn = Callable[[ArrayLike, SensorParams], Array]
BroadcastFn = Callable[[ArrayLike, SensorParams], Array]


def default_params(beacon_gain: float) -> SensorParams:
    """The sensor field's parameters, with the beacon's pull left to the caller.

    Args:
        beacon_gain: how much the reading degrades per unit of squared distance
            from the beacon, k. Zero switches the beacon off. The blackout stays.

    Returns:
        A pytree of the values [`reference_sensor`][cpomdp_workshop.reference_sensor]
        and [`beacon_sensor`][cpomdp_workshop.beacon_sensor] read. Every entry is an
        array, so the whole thing differentiates.

    Raises:
        TypeError: if ``beacon_gain`` is still the ``...`` the attempt cell left.
    """
    if beacon_gain is Ellipsis:
        raise TypeError("beacon_gain is still ..., so type a number for k first.")
    return {
        "beacon": jnp.asarray(BEACON),
        "beacon_gain": jnp.asarray(float(beacon_gain)),
        "beacon_cap": jnp.asarray(BEACON_CAP),
        "blackout": jnp.asarray(BLACKOUT),
        "blackout_gain": jnp.asarray(BLACKOUT_GAIN),
        "blackout_width": jnp.asarray(BLACKOUT_WIDTH),
        "noise_floor": jnp.asarray(NOISE_FLOOR),
    }


def reference_sensor(x: ArrayLike, params: SensorParams) -> Array:
    """The full field: a beacon that sharpens the reading and a blackout that ruins it.

    Args:
        x: where the reading is taken.
        params: from [`default_params`][cpomdp_workshop.default_params].

    Returns:
        The 2x2 observation-noise covariance at ``x``, isotropic.
    """
    x = jnp.asarray(x, dtype=float)
    from_beacon = jnp.sum((x - params["beacon"]) ** 2)
    from_blackout = jnp.sum((x - params["blackout"]) ** 2)
    beacon_term = params["beacon_gain"] * jnp.minimum(from_beacon, params["beacon_cap"])
    blackout_term = params["blackout_gain"] * jnp.exp(
        -from_blackout / params["blackout_width"]
    )
    return jnp.eye(2) * (params["noise_floor"] + beacon_term + blackout_term)


def beacon_sensor(x: ArrayLike, params: SensorParams) -> Array:
    """The same field with the blackout switched off. Only the beacon is left.

    Args:
        x: where the reading is taken.
        params: from [`default_params`][cpomdp_workshop.default_params].

    Returns:
        The 2x2 observation-noise covariance at ``x``, isotropic.
    """
    return reference_sensor(x, {**params, "blackout_gain": jnp.asarray(0.0)})


def pad_params() -> SensorParams:
    """The hidden pad's settings: the broadcast's, then the other two readings' noise.

    Returns:
        A pytree of arrays: where the beacon is, how fast its broadcast fades
        (``broadcast_gain``, g), the broadcast's noise on the beacon itself, and the
        fixed noise of the radar and of the rocket's own position reading.
    """
    return {
        "beacon": jnp.asarray(BEACON),
        "broadcast_gain": jnp.asarray(BROADCAST_GAIN),
        "noise_floor": jnp.asarray(NOISE_FLOOR),
        "radar_noise": jnp.asarray(RADAR_NOISE),
        "own_noise": jnp.asarray(OWN_NOISE),
    }


def broadcast_noise(x: ArrayLike, params: SensorParams) -> Array:
    """The noise on the beacon's broadcast of the pad's position, heard at ``x``.

    Args:
        x: where the rocket hears the broadcast.
        params: from [`pad_params`][cpomdp_workshop.pad_params].

    Returns:
        The variance on each axis: the floor at the beacon, growing with the squared
        distance from it and never capped.
    """
    distance_squared = jnp.sum((jnp.asarray(x, dtype=float) - params["beacon"]) ** 2)
    return params["noise_floor"] + params["broadcast_gain"] * distance_squared


def pad_sensor(broadcast: BroadcastFn) -> NoiseFn:
    """The six readings' noise, built around a broadcast noise function.

    Args:
        broadcast: the broadcast's variance at the rocket's position,
            ``broadcast(x, params)``, such as
            [`broadcast_noise`][cpomdp_workshop.broadcast_noise].

    Returns:
        ``R(s, params)`` for the state s = (x, p): a 6x6 block-diagonal covariance with
        the own-position noise, the radar's noise and the broadcast's, each on two
        axes. Only the broadcast's depends on where the rocket is.
    """

    def noise(state: ArrayLike, params: SensorParams) -> Array:
        rocket = jnp.asarray(state, dtype=float)[:2]
        return jax.scipy.linalg.block_diag(
            params["own_noise"] * jnp.eye(2),
            params["radar_noise"] * jnp.eye(2),
            broadcast(rocket, params) * jnp.eye(2),
        )

    return noise
