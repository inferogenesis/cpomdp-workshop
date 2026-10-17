"""One test per value the deck or the notebook quotes.

A number in the prose has a test here, and the prose is patched to the test when the
two disagree. Every flight runs under the pinned cpomdp release, with the legacy
Mersenne Twister draw seeded once per flight. A re-pin re-runs this file first.
"""

from importlib import metadata

import numpy as np
import pytest
from cpomdp import (
    Belief,
    CallableSensor,
    IncompatibleLinearizationError,
    KalmanBackend,
    LinearGaussianModel,
)

from cpomdp_workshop import WORKSHOP_CPOMDP_VERSION
from cpomdp_workshop._seams import RecedingHorizonSelector
from cpomdp_workshop.build import (
    curious_rocket,
    model_with,
    one_node,
    the_1960_rocket,
)
from cpomdp_workshop.check import (
    backend_differences,
    check_agent,
    check_backends,
    check_my_sensor,
    epistemic_spread,
    probe,
)
from cpomdp_workshop.planner import compass_planner
from cpomdp_workshop.selfguided import two_node_backend
from cpomdp_workshop.sim import ROCK, fly
from cpomdp_workshop.solutions import CLEAR_SKY_ROCK, axis_certificate, rxinfer_refusal
from cpomdp_workshop.world import (
    BEACON,
    PAD,
    beacon_sensor,
    default_params,
    reference_sensor,
)

I = np.eye(2)


def rounded(values, places=2):
    return tuple(round(float(v), places) for v in values)


def test_the_numbers_were_measured_on_the_pinned_release():
    assert metadata.version("cpomdp") == WORKSHOP_CPOMDP_VERSION


def test_no_move_of_the_1960_rocket_is_worth_more_than_any_other():
    report = probe(model_with())
    assert report.flattens is True
    assert report.epistemic_varies is False
    assert rounded(report.epistemic_range, 4) == (4.3947, 4.3947)
    assert f"{epistemic_spread(report):.6f} nats" == "0.000000 nats"


def test_the_curious_rocket_separates_its_moves_by_more_than_a_hundredth_of_a_nat():
    report = probe(model_with(reference_sensor, default_params(0.08)))
    assert report.flattens is False
    assert rounded(report.epistemic_range, 4) == (2.4490, 2.4692)
    assert epistemic_spread(report) > 0.01


def test_the_1960_rocket_prints_its_gain():
    assert repr(the_1960_rocket()) == (
        "Agent(objective=StateGoal, selector=LQRSelector(controller=LQRController("
        "gain=[[0.0951, 0.0], [0.0, 0.0951]])), backend=KalmanBackend)"
    )


def test_the_1960_rocket_reports_where_it_lands(capsys):
    check_agent(the_1960_rocket())
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "after 36 moves it is at (4.87, 0.04), 0.13 from the pad"


def test_papercut_one_is_now_a_refusal_that_names_the_right_slot():
    sensor = CallableSensor(I, reference_sensor, default_params(0.08))
    message = (
        "observation_noise takes a matrix, got CallableSensor, an ObservationModel. "
        "Pass it as observation_model."
    )
    with pytest.raises(TypeError) as refusal:
        LinearGaussianModel(
            I,
            control_matrix=I,
            dynamics_noise=1e-3 * I,
            observation_noise=sensor,
            prior=Belief(mean=[0.0, 0.0], cov=4 * I),
        )
    assert str(refusal.value) == message


def test_the_search_is_decided_rather_than_sampled():
    rocket = curious_rocket(0.08)
    rocket.infer_states(np.zeros(2))
    rocket.sample_action()
    assert isinstance(rocket.selector, RecedingHorizonSelector)
    assert str(rocket.selector.certificate) == (
        "PROVED (set compass-9, |A|^H = 9^1 = 9, visited 9)"
    )
    assert rocket.selector.cost_per_cycle == 9


def test_a_second_step_of_lookahead_costs_eighteen_times_as_much():
    planner = compass_planner(model_with(reference_sensor, default_params(0.08)))
    assert planner.cost_per_cycle == 9
    assert compass_planner(planner.model, horizon=2).cost_per_cycle == 162


def test_the_coda_certificate_covers_every_seven_step_axis_plan():
    assert str(axis_certificate()) == (
        "PROVED (set axis-5, |A|^H = 5^7 = 78125, visited 78125)"
    )


@pytest.mark.parametrize("seed", range(10))
def test_the_1960_rocket_hits_the_rock_on_step_six_whatever_the_noise_does(seed):
    flight = fly(the_1960_rocket(), seed=seed, hazard=ROCK)
    assert flight.crashed is True
    assert flight.impact_step == 6


def test_the_default_slider_rocket_clears_the_rock_and_lands():
    flight = fly(curious_rocket(0.08), hazard=ROCK)
    assert flight.crashed is False
    assert round(float(flight.path[:, 1].max()), 2) == 1.71
    assert rounded(flight.final) == (4.83, 0.09)


def test_the_k_point_two_rocket_clears_the_rock_and_lands():
    flight = fly(curious_rocket(0.20), hazard=ROCK)
    assert flight.crashed is False
    assert round(float(flight.path[:, 1].max()), 2) == 1.91
    assert rounded(flight.final) == (5.18, 0.15)


def test_session_two_bends_the_flight_path_up_to_the_beacon():
    flight = fly(curious_rocket(0.4, beacon_sensor))
    assert round(float(flight.path[:, 1].max()), 2) == 1.41
    assert flight.path[:, 1].max() >= BEACON[1]
    assert rounded(flight.final) == (5.12, 0.21)
    assert flight.distance_to(PAD) < 0.3


@pytest.mark.parametrize(
    ("horizon", "short_of_the_pad"), [(1, 1.49), (2, 1.65), (3, 1.86)]
)
def test_a_rocket_too_curious_to_land_stays_short_however_far_it_looks(
    horizon, short_of_the_pad
):
    flight = fly(curious_rocket(0.6, beacon_sensor, horizon=horizon))
    assert round(flight.distance_to(PAD), 2) == short_of_the_pad


@pytest.mark.parametrize("beacon_gain", [0.08, 0.20])
def test_a_rock_in_clear_sky_stops_the_curious_rockets_on_step_five(beacon_gain):
    flight = fly(curious_rocket(beacon_gain), hazard=CLEAR_SKY_ROCK)
    assert flight.crashed is True
    assert flight.impact_step == 5


def test_the_1960_rocket_flies_under_the_clear_sky_rock():
    assert fly(the_1960_rocket(), hazard=CLEAR_SKY_ROCK).crashed is False


def test_the_beacon_sensor_bends_the_path_to_the_beacon_and_lands(capsys):
    check_my_sensor(beacon_sensor, default_params(0.4))
    assert capsys.readouterr().out.strip() == (
        "The rocket bends 1.41 off the straight line and lands 0.24 from the pad. "
        "It matches the workshop's beacon_sensor."
    )


def test_the_rxinfer_oracle_refuses_the_curious_model_in_these_words():
    assert rxinfer_refusal(model_with(reference_sensor, default_params(0.08))) == (
        "the model's observation_model is state-dependent; the RxInfer model filters "
        "on one fixed Q and R. Use KalmanBackend for R(x) or Q(x)."
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("beacon_gain", "hit_on_seeds"), [(0.08, [1, 2]), (0.20, [1, 2, 9])]
)
def test_over_ten_seeds_the_curious_rockets_clip_the_rock_a_few_times(
    beacon_gain, hit_on_seeds
):
    hits = [
        seed
        for seed in range(10)
        if fly(curious_rocket(beacon_gain), seed=seed, hazard=ROCK).crashed
    ]
    assert hits == hit_on_seeds


def test_both_engines_print_the_same_belief_for_the_1960_rocket(capsys):
    model = model_with()
    check_backends(KalmanBackend(model), one_node(model))
    printed = capsys.readouterr().out.splitlines()
    belief = "mean (4.844446, 0.098086)   cov diagonal (0.006589, 0.006589)"
    assert printed[0] == f"KalmanBackend          {belief}"
    assert printed[1] == f"CouplingGraphBackend   {belief}"
    assert printed[2].startswith("max |difference| = ")
    assert float(printed[2].split("= ")[1]) < 1e-9


def test_the_kalman_filter_and_the_one_node_graph_agree():
    model = model_with()
    differences = backend_differences(KalmanBackend(model), one_node(model))
    assert max(differences) < 1e-9


def test_a_branching_model_refuses_to_flatten():
    with pytest.raises(IncompatibleLinearizationError, match="mean-shifting coupling"):
        two_node_backend().to_flat_model()
