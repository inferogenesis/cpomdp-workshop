"""One test per value the deck or the notebook quotes.

A number in the prose has a test here, and the prose is patched to the test when the
two disagree. Every flight runs under the pinned cpomdp release, with the legacy
Mersenne Twister draw seeded once per flight. A re-pin re-runs this file first.
"""

from importlib import metadata

import numpy as np
import pytest
from cpomdp import (
    Agent,
    Belief,
    CallableSensor,
    IncompatibleLinearizationError,
    KalmanBackend,
    LinearGaussianModel,
    Preference,
    probe_model,
)

from cpomdp_workshop import WORKSHOP_CPOMDP_VERSION
from cpomdp_workshop._seams import RecedingHorizonSelector
from cpomdp_workshop.build import (
    curious_rocket,
    model_with,
    one_node,
    pad_preference,
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
from cpomdp_workshop.planner import compass_moves, compass_planner
from cpomdp_workshop.selfguided import two_node_backend
from cpomdp_workshop.sim import ROCK, fly
from cpomdp_workshop.solutions import CLEAR_SKY_ROCK, axis_certificate
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


def build(beacon_gain):
    return the_1960_rocket() if beacon_gain == 0 else curious_rocket(beacon_gain)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("beacon_gain", "hit_on_seeds"),
    [(0.0, []), (0.08, [2, 4, 5, 6, 7, 9]), (0.20, [4, 5, 6, 8, 9])],
)
def test_in_clear_sky_the_rock_is_hit_on_the_curious_rockets_routes(
    beacon_gain, hit_on_seeds
):
    hits = [
        seed
        for seed in range(10)
        if fly(build(beacon_gain), seed=seed, hazard=CLEAR_SKY_ROCK).crashed
    ]
    assert hits == hit_on_seeds


def test_a_rock_never_changes_a_route_only_ends_it():
    free = fly(curious_rocket(0.08), seed=2)
    hit = fly(curious_rocket(0.08), seed=2, hazard=CLEAR_SKY_ROCK)
    assert hit.crashed is True
    assert np.array_equal(hit.path, free.path[: len(hit.path)])


@pytest.mark.slow
def test_most_curious_routes_pass_below_the_blackout():
    below = 0
    for seed in range(30):
        y = fly(curious_rocket(0.08), seed=seed).path[:, 1]
        below += bool(y[np.argmax(np.abs(y))] < 0)
    assert below == 19


def test_the_beacon_sensor_bends_the_path_to_the_beacon_and_lands(capsys):
    check_my_sensor(beacon_sensor, default_params(0.4))
    assert capsys.readouterr().out.strip() == (
        "The rocket bends 1.41 off the straight line and lands 0.24 from the pad. "
        "It matches the workshop's beacon_sensor."
    )


def a_fixed_model(noise, prior):
    return LinearGaussianModel(
        I,
        control_matrix=I,
        observation_matrix=I,
        observation_noise=np.asarray(noise),
        dynamics_noise=1e-3 * I,
        prior=Belief(mean=[0.0, 0.0], cov=np.asarray(prior)),
    )


def test_a_sensor_sharper_along_x_teaches_less_per_reading_and_the_same_per_move():
    report = probe(a_fixed_model(np.diag([0.05, 2.0]), 4 * I))
    assert rounded(report.epistemic_range, 4) == (2.7467, 2.7467)
    assert f"{epistemic_spread(report):.6f} nats" == "0.000000 nats"


@pytest.mark.parametrize(
    ("noise", "prior"),
    [
        (np.diag([0.05, 2.0]), 4 * I),
        (np.diag([2.0, 0.05]), 4 * I),
        (np.diag([0.05, 2.0]), np.diag([4.0, 0.5])),
        ([[1.0, 0.8], [0.8, 1.0]], 4 * I),
    ],
)
def test_noise_fixed_in_place_never_tells_the_moves_apart(noise, prior):
    report = probe(a_fixed_model(noise, prior))
    assert report.epistemic_varies is False
    assert f"{epistemic_spread(report):.6f}" == "0.000000"


def fly_with(sensor, beacon_gain, hazard=None, **changes):
    params = {**default_params(beacon_gain), **changes}
    model = model_with(sensor, params)
    rocket = Agent(model, pad_preference(), selector=compass_planner(model))
    return fly(rocket, hazard=hazard)


def test_a_small_cap_keeps_the_beacons_route_nearly_straight():
    flight = fly_with(beacon_sensor, 0.4, beacon_cap=1.0)
    assert round(float(np.abs(flight.path[:, 1]).max()), 2) == 0.35


def test_a_large_cap_leaves_the_beacons_rocket_well_short_of_the_pad():
    flight = fly_with(beacon_sensor, 0.4, beacon_cap=9.0)
    assert round(flight.distance_to(PAD), 2) == 1.42


@pytest.mark.parametrize("noise_floor", [0.01, 0.1, 0.5])
def test_the_noise_floor_leaves_the_beacons_bend_and_landing_unchanged(noise_floor):
    flight = fly_with(beacon_sensor, 0.4, noise_floor=noise_floor)
    assert round(float(np.abs(flight.path[:, 1]).max()), 2) == 1.41
    assert round(flight.distance_to(PAD), 2) == 0.24


@pytest.mark.parametrize("beacon_gain", [0.08, 0.20])
@pytest.mark.parametrize(
    "changes",
    [{"blackout_width": 0.1}, {"blackout_gain": 1.0}],
    ids=["narrow", "faint"],
)
def test_a_small_or_faint_blackout_lets_the_curious_rockets_hit_the_rock(
    beacon_gain, changes
):
    flight = fly_with(reference_sensor, beacon_gain, hazard=ROCK, **changes)
    assert flight.crashed is True
    assert flight.impact_step == 5


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


@pytest.mark.parametrize(
    ("noise", "prior", "step"),
    [
        (0.01, 4.0, 0.5),
        (2.0, 4.0, 0.5),
        (0.05, 0.1, 0.5),
        (0.05, 20.0, 0.5),
        (0.05, 4.0, 0.1),
        (0.05, 4.0, 3.0),
    ],
)
def test_no_noise_level_prior_or_step_moves_the_1960_spread(noise, prior, step):
    model = a_fixed_model(noise * I, prior * I)
    report = probe_model(model, model.prior, compass_moves(step))
    assert f"{epistemic_spread(report):.6f}" == "0.000000"


def test_a_beacon_on_the_straight_line_holds_the_rocket_short_of_the_pad():
    flight = fly_with(beacon_sensor, 0.4, beacon=np.array([2.5, 0.0]))
    assert round(flight.distance_to(PAD), 2) == 1.09


def fly_preferring(beacon_gain, precision, seed=3):
    model = model_with(beacon_sensor, default_params(beacon_gain))
    preference = Preference(np.asarray(PAD), precision=precision * I)
    return fly(Agent(model, preference, selector=compass_planner(model)), seed=seed)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("beacon_gain", "precision", "landed"),
    [
        (0.4, 0.1, 0),
        (0.4, 1.0, 8),
        (0.4, 10.0, 9),
        (0.6, 0.1, 0),
        (0.6, 1.0, 0),
        (0.6, 10.0, 0),
    ],
)
def test_precision_helps_a_loose_rocket_land_and_never_lands_a_noisy_pad(
    beacon_gain, precision, landed
):
    ends = [fly_preferring(beacon_gain, precision, seed) for seed in range(10)]
    assert sum(flight.distance_to(PAD) < 0.5 for flight in ends) == landed


def test_above_one_precision_leaves_the_route_unchanged():
    paths = [fly_preferring(0.4, precision).path for precision in (1.0, 3.0, 10.0)]
    assert all(np.array_equal(path, paths[0]) for path in paths)


@pytest.mark.parametrize(
    ("beacon_gain", "landed_at_each_precision"),
    [(0.4, (False, True, True)), (0.6, (False, False, False))],
)
def test_on_seed_3_precision_lands_a_loose_rocket_and_never_a_noisy_pad(
    beacon_gain, landed_at_each_precision
):
    ends = [fly_preferring(beacon_gain, precision) for precision in (0.1, 1.0, 10.0)]
    landed = tuple(flight.distance_to(PAD) < 0.5 for flight in ends)
    assert landed == landed_at_each_precision


def fly_design(k_model, precision, seed, rock):
    model = model_with(reference_sensor, default_params(k_model))
    preference = Preference(np.asarray(PAD), precision=precision * I)
    rocket = Agent(model, preference, selector=compass_planner(model))
    return fly(rocket, reference_sensor, default_params(0.08), seed=seed, hazard=rock)


def tally(k_model, precision, seeds, rock):
    flights = [fly_design(k_model, precision, seed, rock) for seed in seeds]
    landed = sum(not f.crashed and f.distance_to(PAD) < 0.5 for f in flights)
    return landed, sum(f.crashed for f in flights)


# What the task 3 search prints for each (k_model, precision): landings and rock hits
# on seeds 0 to 9, then on the new seeds 10 to 29, then on the new seeds with the rock
# in clear sky.
SEARCH = {
    (0.02, 0.3): ((9, 1), (20, 0), (8, 12)),
    (0.02, 1.0): ((9, 1), (20, 0), (8, 12)),
    (0.02, 3.0): ((8, 2), (20, 0), (9, 11)),
    (0.08, 0.3): ((7, 2), (20, 0), (8, 12)),
    (0.08, 1.0): ((8, 2), (19, 1), (8, 12)),
    (0.08, 3.0): ((8, 2), (19, 1), (8, 12)),
    (0.20, 0.3): ((6, 3), (19, 1), (8, 12)),
    (0.20, 1.0): ((6, 3), (19, 1), (10, 10)),
    (0.20, 3.0): ((6, 3), (19, 1), (9, 11)),
}
SEARCH_WINNERS = [(0.02, 0.3), (0.02, 1.0)]
SEARCH_DEFAULT = (0.08, 1.0)


@pytest.mark.slow
@pytest.mark.parametrize(("k_model", "precision"), list(SEARCH))
def test_the_search_prints_these_tallies(k_model, precision):
    searched, fresh, fresh_clear_sky = SEARCH[(k_model, precision)]
    assert tally(k_model, precision, range(10), ROCK) == searched
    assert tally(k_model, precision, range(10, 30), ROCK) == fresh
    assert tally(k_model, precision, range(10, 30), CLEAR_SKY_ROCK) == fresh_clear_sky


def test_the_searchs_winner_holds_by_one_flight_and_not_when_the_rock_moves():
    searched = {combination: tallies[0] for combination, tallies in SEARCH.items()}
    assert max(landed for landed, _ in searched.values()) == 9
    assert min(hit for _, hit in searched.values()) == 1
    assert [c for c, row in searched.items() if row == (9, 1)] == SEARCH_WINNERS
    assert searched[SEARCH_DEFAULT] == (8, 2)
    assert all(SEARCH[winner][1] == (20, 0) for winner in SEARCH_WINNERS)
    assert SEARCH[SEARCH_DEFAULT][1] == (19, 1)
    clear_sky_hits = [tallies[2][1] for tallies in SEARCH.values()]
    assert (min(clear_sky_hits), max(clear_sky_hits)) == (10, 12)
    assert all(SEARCH[c][2][1] == 12 for c in [*SEARCH_WINNERS, SEARCH_DEFAULT])
