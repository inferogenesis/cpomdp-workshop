import jax.numpy as jnp
import numpy as np

from cpomdp_workshop.check import grade_sensor
from cpomdp_workshop.world import beacon_sensor, default_params, reference_sensor


def numpy_beacon(x, params):
    distance_squared = jnp.sum((x - params["beacon"]) ** 2)
    capped = np.minimum(distance_squared, params["beacon_cap"])
    return jnp.eye(2) * (params["noise_floor"] + params["beacon_gain"] * capped)


def test_a_weak_beacon_leaves_the_path_straight():
    assert grade_sensor(beacon_sensor, default_params(0.02)).grade == "straight"


def test_a_strong_beacon_bends_the_path_and_lands():
    grade = grade_sensor(beacon_sensor, default_params(0.4))
    assert grade.grade == "bends"
    assert grade.matches == "beacon_sensor"


def test_a_beacon_too_strong_stops_the_rocket_short():
    grade = grade_sensor(beacon_sensor, default_params(0.6))
    assert grade.grade == "too curious"
    assert grade.distance is not None
    assert grade.distance > 1.0


def test_the_full_field_is_recognised():
    assert grade_sensor(reference_sensor, default_params(0.08)).matches == (
        "reference_sensor"
    )


def test_a_sensor_jax_cannot_trace_is_caught_before_it_flies():
    grade = grade_sensor(numpy_beacon, default_params(0.4))
    assert grade.grade == "broken"
    assert "jnp.minimum" in grade.message


def test_a_flat_sensor_is_the_1960_rockets():
    def flat(x, params):
        return 0.05 * jnp.eye(2)

    assert grade_sensor(flat, default_params(0.4)).grade == "flat"


def test_a_sensor_of_the_wrong_shape_is_told_the_shape_it_needs():
    def vector(x, params):
        return jnp.ones(2)

    grade = grade_sensor(vector, default_params(0.4))
    assert grade.grade == "broken"
    assert "2x2" in grade.message
