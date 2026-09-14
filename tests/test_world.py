import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp_workshop.world import (
    BEACON,
    BEACON_CAP,
    BLACKOUT,
    BLACKOUT_GAIN,
    LAUNCH,
    NOISE_FLOOR,
    PAD,
    ROCK_RADIUS,
    beacon_sensor,
    default_params,
    reference_sensor,
)

SAMPLE = [(0.0, 0.0), (2.5, 0.0), (2.5, 1.4), (5.0, 0.0), (-3.0, 4.0), (9.0, -9.0)]


@pytest.mark.parametrize("sensor", [reference_sensor, beacon_sensor])
@pytest.mark.parametrize("x", SAMPLE)
def test_sensor_returns_a_2x2_positive_definite_covariance(sensor, x):
    R = np.asarray(sensor(jnp.asarray(x), default_params(0.08)))
    assert R.shape == (2, 2)
    assert np.allclose(R, R.T)
    assert np.linalg.eigvalsh(R).min() > 0


def test_beacon_sensor_bottoms_out_at_the_noise_floor_on_the_beacon():
    R = np.asarray(beacon_sensor(jnp.asarray(BEACON), default_params(0.08)))
    assert np.allclose(R, NOISE_FLOOR * np.eye(2))


def test_the_beacon_term_saturates_at_the_cap():
    p = default_params(0.2)
    far = np.asarray(beacon_sensor(jnp.asarray([BEACON[0] + 50.0, BEACON[1]]), p))
    at_cap = np.asarray(beacon_sensor(jnp.asarray([BEACON[0] + 2.0, BEACON[1]]), p))
    assert np.allclose(far, at_cap)
    assert np.allclose(far, (NOISE_FLOOR + 0.2 * BEACON_CAP) * np.eye(2))


def test_the_blackout_is_the_only_difference_between_the_two_sensors():
    p = default_params(0.08)
    away = jnp.asarray([-4.0, 3.0])
    assert np.allclose(
        np.asarray(reference_sensor(away, p)), np.asarray(beacon_sensor(away, p))
    )
    centre = jnp.asarray(BLACKOUT)
    gap = np.asarray(reference_sensor(centre, p)) - np.asarray(beacon_sensor(centre, p))
    assert np.allclose(gap, BLACKOUT_GAIN * np.eye(2))


def test_the_blackout_dominates_the_floor_by_three_orders_of_magnitude():
    p = default_params(0.08)
    blind = float(np.asarray(reference_sensor(jnp.asarray(BLACKOUT), p))[0, 0])
    assert blind / NOISE_FLOOR > 1000


def test_k_zero_leaves_the_beacon_with_no_pull():
    p = default_params(0.0)
    on = np.asarray(beacon_sensor(jnp.asarray(BEACON), p))
    off = np.asarray(beacon_sensor(jnp.asarray([-4.0, 3.0]), p))
    assert np.allclose(on, off)


def test_the_launch_to_pad_line_runs_through_the_rock():
    launch, pad, rock = np.array(LAUNCH), np.array(PAD), np.array(BLACKOUT)
    along = (pad - launch) / np.linalg.norm(pad - launch)
    offset = rock - launch
    closest = launch + along * float(offset @ along)
    assert np.linalg.norm(closest - rock) < ROCK_RADIUS
