"""The reference world: where things are, and how noisy the rocket's reading is there.

The rocket flies in the plane from the launch site to the pad. Two features change
what it can see. A nav beacon sharpens the reading the closer the rocket gets to it,
up to a cap. A blackout zone swamps the reading near its centre, and a rock sits
inside it. Both are parameters of one noise field ``R(x)``, so a session can switch
the blackout off by choosing a different sensor function rather than a different
world.
"""

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from numpy.typing import ArrayLike

__all__ = [
    "BEACON",
    "BEACON_CAP",
    "BLACKOUT",
    "BLACKOUT_GAIN",
    "BLACKOUT_WIDTH",
    "DYNAMICS_NOISE",
    "LAUNCH",
    "NOISE_FLOOR",
    "PAD",
    "PRIOR_COV",
    "ROCK_RADIUS",
    "NoiseFn",
    "SensorParams",
    "beacon_sensor",
    "default_params",
    "reference_sensor",
]

LAUNCH = (0.0, 0.0)
PAD = (5.0, 0.0)
BEACON = (2.5, 1.4)
BLACKOUT = (2.5, 0.0)
ROCK_RADIUS = 0.42

NOISE_FLOOR = 0.05  # R0
BEACON_CAP = 4.0  # cap
BLACKOUT_GAIN = 60.0  # M
BLACKOUT_WIDTH = 0.45  # s

DYNAMICS_NOISE = 1e-3
PRIOR_COV = 4.0

SensorParams = dict[str, Array]
NoiseFn = Callable[[ArrayLike, SensorParams], Array]


def default_params(beacon_gain: float) -> SensorParams:
    """The sensor field's parameters, with the beacon's pull left to the caller.

    Args:
        beacon_gain: how much the reading degrades per unit of squared distance
            from the beacon, k. Zero switches the beacon off. The blackout stays.

    Returns:
        A pytree of the values [`reference_sensor`][cpomdp_workshop.reference_sensor]
        and [`beacon_sensor`][cpomdp_workshop.beacon_sensor] read. Every entry is an
        array, so the whole thing differentiates.
    """
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
