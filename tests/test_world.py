import re

import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp_workshop.world import (
    BEACON,
    BEACON_CAP,
    BLACKOUT,
    BLACKOUT_GAIN,
    BROADCAST_GAIN,
    LAUNCH,
    NOISE_FLOOR,
    OWN_NOISE,
    PAD,
    PAD_READINGS,
    RADAR_NOISE,
    ROCK_RADIUS,
    beacon_sensor,
    broadcast_noise,
    default_params,
    pad_params,
    pad_sensor,
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


def test_a_k_left_unwritten_is_named():
    with pytest.raises(TypeError, match=re.escape("beacon_gain is still ...")):
        default_params(...)  # ty: ignore[invalid-argument-type]


def test_the_hidden_pad_is_read_three_ways():
    state = np.array(
        [1.0, 2.0, 5.5, -0.5]
    )  # the rocket at (1, 2), the pad at (5.5, -0.5)
    readings = np.asarray(PAD_READINGS) @ state
    assert np.allclose(readings, [1.0, 2.0, 4.5, -2.5, 5.5, -0.5])


def test_the_broadcast_is_clearest_on_the_beacon_and_fades_without_a_cap():
    params = pad_params()
    on_beacon = float(broadcast_noise(jnp.asarray(BEACON), params))
    far = float(broadcast_noise(jnp.asarray([BEACON[0] + 10.0, BEACON[1]]), params))
    assert on_beacon == pytest.approx(NOISE_FLOOR)
    assert far == pytest.approx(NOISE_FLOOR + BROADCAST_GAIN * 100.0)


def test_the_pad_sensor_wraps_a_broadcast_in_the_six_readings_noise():
    noise = pad_sensor(broadcast_noise)(jnp.asarray([*BEACON, 5.0, 0.0]), pad_params())
    noise = np.asarray(noise)
    expected = [OWN_NOISE] * 2 + [RADAR_NOISE] * 2 + [NOISE_FLOOR] * 2
    assert np.allclose(noise, np.diag(expected))


def test_the_pad_params_hold_what_the_broadcast_and_the_radar_read():
    params = pad_params()
    assert set(params) == {
        "beacon",
        "broadcast_gain",
        "noise_floor",
        "radar_noise",
        "own_noise",
    }
    assert np.allclose(params["beacon"], BEACON)
    assert float(params["radar_noise"]) == RADAR_NOISE
