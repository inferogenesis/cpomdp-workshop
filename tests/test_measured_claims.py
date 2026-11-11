"""One test per value the deck or the notebook quotes.

A number in the prose has a test here, and the prose is patched to the test when the
two disagree. Every flight runs under the pinned cpomdp release, with the legacy
Mersenne Twister draw seeded once per flight. A re-pin re-runs this file first.
"""

import dataclasses
from importlib import metadata
from typing import NamedTuple

import jax.numpy as jnp
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
    CURIOUS_PRECISION,
    curious_rocket,
    model_with,
    one_node,
    pad_model,
    pad_preference,
    radar_preference,
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
from cpomdp_workshop.sim import fly, fly_pad
from cpomdp_workshop.solutions import (
    axis_certificate,
    curiosity_decided,
)
from cpomdp_workshop.world import (
    BEACON,
    LAUNCH,
    PAD,
    PAD_GUESS,
    beacon_sensor,
    broadcast_noise,
    default_params,
    pad_params,
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
    assert printed[-2] == "after 36 moves it is at (4.87, 0.04), 0.13 from the pad"


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


def test_two_moves_ahead_scores_eighty_one_plans_a_step():
    model = model_with(reference_sensor, default_params(0.08))
    planner = compass_planner(model, horizon=2)
    planner.select(model.prior, pad_preference())
    assert str(planner.certificate) == (
        "PROVED (set compass-9, |A|^H = 9^2 = 81, visited 81)"
    )


def test_the_coda_certificate_covers_every_seven_step_axis_plan():
    assert str(axis_certificate()) == (
        "PROVED (set axis-5, |A|^H = 5^7 = 78125, visited 78125)"
    )


def test_the_beacon_sensor_bends_the_path_to_the_beacon_and_lands(capsys):
    check_my_sensor(beacon_sensor, default_params(0.08))
    assert capsys.readouterr().out.strip() == (
        "The curious rocket bends 1.41 off the straight line and lands 0.23 from the "
        "pad. "
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


# --- Task 2: the same rocket flown with curiosity on and off ----------------------
#
# Every seed-3 claim below sits beside the ten-seed count it stands for, so a seed that
# stops behaving like the rest fails here rather than on a slide.


def a_beacon_model(beacon_gain, *, dynamics_noise=1e-3, prior_cov=4.0, **changes):
    # The attempt cell's model, with the bonus's one-line edits as arguments.
    params = {**default_params(beacon_gain), **changes}
    return LinearGaussianModel(
        I,
        control_matrix=I,
        dynamics_noise=dynamics_noise * I,
        prior=Belief(mean=[0.0, 0.0], cov=prior_cov * I),
        observation_model=CallableSensor(I, beacon_sensor, params),
    )


def fly_pair(beacon_gain, precision=CURIOUS_PRECISION, seed=3, **model_changes):
    preference = Preference(np.asarray(PAD), precision=precision * I)
    flights = []
    for curiosity in (True, False):
        model = a_beacon_model(beacon_gain, **model_changes)
        planner = compass_planner(model, curiosity=curiosity)
        flights.append(fly(Agent(model, preference, selector=planner), seed=seed))
    return flights


def lands(flight):
    return flight.distance_to(PAD) < 0.5


def nearest_beacon(flight):
    return float(np.linalg.norm(flight.path - np.asarray(BEACON), axis=1).min())


class Pairs(NamedTuple):
    parted: int  # seeds whose routes are ever 1 or more apart
    median_gap: float
    on_landed: int
    off_landed: int


def over_ten_seeds(beacon_gain, precision=CURIOUS_PRECISION, **model_changes):
    pairs = [
        fly_pair(beacon_gain, precision, seed, **model_changes) for seed in range(10)
    ]
    gaps = np.array([on.route_gap(off) for on, off in pairs])
    return pairs, Pairs(
        int((gaps >= 1).sum()),
        round(float(np.median(gaps)), 2),
        sum(lands(on) for on, _ in pairs),
        sum(lands(off) for _, off in pairs),
    )


def decided_on_each_seed(beacon_gain, precision):
    counts = []
    for seed in range(10):
        rocket = curious_rocket(beacon_gain, beacon_sensor, precision=precision)
        flight = fly(rocket, seed=seed)
        decided = curiosity_decided(flight, rocket.model, pad_preference(precision))
        counts.append(int(decided.sum()))
    return counts


def test_the_curious_rocket_prefers_the_pad_loosely():
    assert CURIOUS_PRECISION == 0.1
    assert np.allclose(pad_preference().precision, 0.1 * I)


@pytest.mark.parametrize("beacon_gain", [0.2, 0.4])
def test_with_curiosity_off_the_route_is_the_same_at_any_precision(beacon_gain):
    off = [fly_pair(beacon_gain, precision)[1].path for precision in (0.1, 1.0, 3.0)]
    assert all(np.array_equal(path, off[0]) for path in off)


def test_at_k_0_08_curiosity_flies_to_the_beacon_and_both_rockets_land():
    on, off = fly_pair(0.08)
    assert round(nearest_beacon(on), 2) == 0.09
    assert round(nearest_beacon(off), 2) == 1.40
    assert round(on.distance_to(PAD), 2) == 0.23
    assert round(off.distance_to(PAD), 2) == 0.29
    assert round(on.route_gap(off), 2) == 1.53


def test_at_the_starting_k_curiosity_decides_six_of_the_36_moves():
    rocket = curious_rocket(0.08, beacon_sensor)
    flight = fly(rocket)
    decided = curiosity_decided(flight, rocket.model, pad_preference())
    assert (int(decided.sum()), len(decided)) == (6, 36)


@pytest.mark.slow
def test_at_the_starting_k_the_routes_part_on_every_seed_and_every_rocket_lands():
    pairs, tally = over_ten_seeds(0.08)
    assert tally == Pairs(10, 1.53, 10, 10)
    assert round(float(np.median([nearest_beacon(on) for on, _ in pairs])), 2) == 0.31


def test_the_sensor_check_at_k_0_4_says_the_curious_rocket_stops_short(capsys):
    check_my_sensor(beacon_sensor, default_params(0.4))
    assert capsys.readouterr().out.strip() == (
        "The curious rocket bends 2.06 off the straight line and stops 2.79 short of "
        "the pad. It matches the workshop's beacon_sensor."
    )


def test_at_k_0_4_the_curious_rocket_stops_short_and_the_incurious_one_lands():
    on, off = fly_pair(0.4)
    assert round(on.distance_to(PAD), 2) == 2.79
    assert round(nearest_beacon(on), 2) == 0.08
    assert round(off.distance_to(PAD), 2) == 0.24
    assert lands(off)


@pytest.mark.slow
def test_at_k_0_4_no_curious_rocket_lands_and_every_incurious_one_does():
    pairs, tally = over_ten_seeds(0.4)
    assert (tally.on_landed, tally.off_landed) == (0, 10)
    short = np.median([on.distance_to(PAD) for on, _ in pairs])
    assert round(float(short), 2) == 1.67


@pytest.mark.parametrize(
    ("beacon_gain", "landed"), [(0.2, True), (0.3, True), (0.4, False)]
)
def test_on_seed_3_the_curious_rocket_lands_up_to_k_0_3(beacon_gain, landed):
    on, off = fly_pair(beacon_gain)
    assert lands(on) is landed
    assert lands(off)


@pytest.mark.slow
@pytest.mark.parametrize(("beacon_gain", "landed"), [(0.2, 10), (0.3, 8), (0.4, 0)])
def test_over_ten_seeds_the_curious_rocket_lands_less_as_k_rises(beacon_gain, landed):
    landings = 0
    for seed in range(10):
        rocket = curious_rocket(beacon_gain, beacon_sensor)
        landings += lands(fly(rocket, seed=seed))
    assert landings == landed


# Task 2 bonus 1: the preference turned up, on seed 3 and then over ten seeds.


@pytest.mark.parametrize(
    ("beacon_gain", "gaps"),
    [(0.08, (1.53, 0.77, 0.00, 0.00)), (0.2, (1.48, 1.15, 0.38, 0.00))],
)
def test_on_seed_3_turning_the_preference_up_closes_the_gap(beacon_gain, gaps):
    pairs = [fly_pair(beacon_gain, precision) for precision in (0.1, 0.3, 1.0, 3.0)]
    assert tuple(round(on.route_gap(off), 2) for on, off in pairs) == gaps
    assert all(lands(on) and lands(off) for on, off in pairs)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("beacon_gain", "precision", "expected"),
    [
        (0.08, 0.1, Pairs(10, 1.53, 10, 10)),
        (0.08, 0.3, Pairs(1, 0.38, 10, 10)),
        (0.08, 1.0, Pairs(0, 0.00, 10, 10)),
        (0.08, 3.0, Pairs(0, 0.00, 10, 10)),
        (0.2, 0.1, Pairs(10, 2.50, 10, 9)),
        (0.2, 0.3, Pairs(8, 1.12, 10, 9)),
        (0.2, 1.0, Pairs(0, 0.38, 9, 9)),
        (0.2, 3.0, Pairs(0, 0.00, 9, 9)),
    ],
)
def test_over_ten_seeds_turning_the_preference_up_closes_the_gap(
    beacon_gain, precision, expected
):
    assert over_ten_seeds(beacon_gain, precision)[1] == expected


# Slide 13: at precision 1 the route still bends and curiosity decides almost nothing.


@pytest.mark.slow
def test_at_precision_1_and_the_starting_k_curiosity_decides_no_move_on_any_seed():
    assert decided_on_each_seed(0.08, 1.0) == [0] * 10


@pytest.mark.slow
def test_at_k_0_2_curiosity_decides_one_move_at_most_at_precision_1_and_six_at_0_1():
    tight = decided_on_each_seed(0.2, 1.0)
    loose = decided_on_each_seed(0.2, CURIOUS_PRECISION)
    assert max(tight) == 1
    assert (min(loose), max(loose)) == (6, 9)
    assert (tight[3], loose[3]) == (1, 7)


def test_at_k_0_2_the_incurious_rocket_bends_towards_the_beacon_too():
    _, off = fly_pair(0.2)
    on_tight, _ = fly_pair(0.2, 1.0)
    on_loose, _ = fly_pair(0.2)
    straight = nearest_beacon(fly_pair(0.08, seed=3, beacon_cap=jnp.asarray(1.0))[1])
    assert round(straight, 2) == 1.40
    assert round(nearest_beacon(off), 2) == 1.06
    assert round(nearest_beacon(on_tight), 2) == 0.72
    assert round(nearest_beacon(on_loose), 2) == 0.09


@pytest.mark.slow
def test_at_k_0_4_and_precision_1_seed_3_is_not_the_typical_flight():
    # Why slide 13 compares the precisions at k = 0.2: at k = 0.4, curiosity decides
    # no move on seed 3 but parts the routes on six seeds of ten.
    assert decided_on_each_seed(0.4, 1.0) == [0, 0, 0, 0, 2, 2, 1, 3, 1, 6]
    assert over_ten_seeds(0.4, 1.0)[1] == Pairs(6, 1.20, 8, 10)


# Task 2 bonus 2: what curiosity needs, one edit to the attempt cell at a time.


def test_on_seed_3_a_flat_route_leaves_curiosity_nothing_to_choose():
    on, off = fly_pair(0.08, beacon_cap=jnp.asarray(1.0))
    assert np.array_equal(on.path, off.path)
    assert round(nearest_beacon(on), 2) == 1.40


@pytest.mark.slow
def test_over_ten_seeds_a_flat_route_leaves_curiosity_nothing_to_choose():
    pairs, tally = over_ten_seeds(0.08, beacon_cap=jnp.asarray(1.0))
    assert tally == Pairs(0, 0.00, 10, 10)
    # On seed 8 a noisy reading pushes the rocket within the beacon's reach, where
    # the noise varies again, and the routes split there.
    gaps = [round(on.route_gap(off), 2) for on, off in pairs]
    assert gaps == [0.0] * 8 + [0.77, 0.0]


def test_on_seed_3_more_drift_keeps_the_curious_rocket_from_landing():
    on, off = fly_pair(0.08, dynamics_noise=1e-2)
    assert round(on.distance_to(PAD), 2) == 2.67
    assert round(off.distance_to(PAD), 2) == 0.16


@pytest.mark.slow
def test_over_ten_seeds_more_drift_keeps_the_curious_rocket_from_landing():
    assert over_ten_seeds(0.08, dynamics_noise=1e-2)[1][2:] == (0, 8)


def test_on_seed_3_a_wide_prior_leaves_the_curious_route_unchanged():
    wide, _ = fly_pair(0.08, prior_cov=25.0)
    usual, _ = fly_pair(0.08)
    assert np.array_equal(wide.path, usual.path)


@pytest.mark.slow
def test_over_ten_seeds_a_wide_prior_leaves_the_curious_route_unchanged():
    for seed in range(10):
        wide, _ = fly_pair(0.08, seed=seed, prior_cov=25.0)
        usual, _ = fly_pair(0.08, seed=seed)
        assert np.array_equal(wide.path, usual.path), seed


# Curiosity off lands only up to k = 0.45. Past it, the pragmatic term alone stops the
# rocket short, because it pays for the reading's noise too.


@pytest.mark.parametrize(
    ("beacon_gain", "on_short", "off_short"),
    [
        (0.36, 0.19, 0.24),
        (0.37, 1.39, 0.36),
        (0.45, 1.52, 0.22),
        (0.46, 1.52, 1.42),
        (0.5, 1.87, 1.30),
    ],
)
def test_on_seed_3_the_rockets_part_only_for_k_from_0_37_to_0_45(
    beacon_gain, on_short, off_short
):
    on, off = fly_pair(beacon_gain)
    assert round(on.distance_to(PAD), 2) == on_short
    assert round(off.distance_to(PAD), 2) == off_short
    assert (lands(on), lands(off)) == (beacon_gain < 0.37, beacon_gain < 0.46)


@pytest.mark.slow
def test_at_k_0_5_neither_rocket_lands_on_any_seed():
    _, tally = over_ten_seeds(0.5)
    assert (tally.on_landed, tally.off_landed) == (0, 0)


# --- Task 3: the hidden pad, flown with curiosity on and off ----------------------


def capped_broadcast(x, params):
    # Task 2's field on the broadcast: k = 0.08 with a cap of 4.
    distance_squared = jnp.sum((x - params["beacon"]) ** 2)
    return params["noise_floor"] + 0.08 * jnp.minimum(distance_squared, 4.0)


def guessing(model, guess):
    # The model with the pad's two numbers in the prior's mean moved to the guess.
    mean = np.array([*LAUNCH, *guess])
    return dataclasses.replace(model, prior=Belief(mean=mean, cov=model.prior.cov))


def fly_pad_pair(
    seed=3,
    *,
    pad_spread=1.5,
    precision=CURIOUS_PRECISION,
    horizon=1,
    broadcast=broadcast_noise,
    guess=PAD_GUESS,
    **settings,
):
    params = {**pad_params(), **{k: jnp.asarray(v) for k, v in settings.items()}}
    flights = []
    for curiosity in (True, False):
        model = guessing(pad_model(broadcast, params, pad_spread), guess)
        planner = compass_planner(model, horizon=horizon, curiosity=curiosity)
        rocket = Agent(model, radar_preference(precision), selector=planner)
        flights.append(fly_pad(rocket, seed=seed))
    return flights


class PadTally(NamedTuple):
    on_landed: int
    on_median_miss: float
    off_landed: int
    off_median_miss: float


def pad_tally(**settings):
    pairs = [fly_pad_pair(seed, **settings) for seed in range(10)]
    on = np.array([flight.miss for flight, _ in pairs])
    off = np.array([flight.miss for _, flight in pairs])
    tally = PadTally(
        int((on < 0.5).sum()),
        round(float(np.median(on)), 2),
        int((off < 0.5).sum()),
        round(float(np.median(off)), 2),
    )
    return pairs, tally


def beacon_visits(flights):
    # The seeds, counted from zero, whose flight passes within 0.5 of the beacon.
    return [seed for seed, f in enumerate(flights) if nearest_beacon(f) < 0.5]


def test_on_seed_3_the_curious_rocket_finds_the_pad_and_the_other_misses():
    on, off = fly_pad_pair()
    assert (round(on.miss, 2), round(off.miss, 2)) == (0.17, 0.85)
    assert round(nearest_beacon(on), 2) == 0.17
    assert round(nearest_beacon(off), 2) == 1.40


@pytest.mark.slow
def test_gate_c2_curious_lands_on_8_or_more_seeds_and_incurious_on_4_or_fewer():
    pairs, tally = pad_tally()
    assert tally.on_landed >= 8
    assert tally.off_landed <= 4
    assert tally == PadTally(9, 0.25, 3, 0.61)
    visits = [nearest_beacon(on) for on, _ in pairs]
    assert round(float(np.median(visits)), 2) == 0.09
    # The incurious rocket nears the beacon on seed 8 alone, where the pad lies that
    # way.
    assert beacon_visits(off for _, off in pairs) == [8]


@pytest.mark.parametrize(
    ("pad_spread", "misses"), [(0.1, (2.76, 2.76)), (3.0, (0.17, 0.48))]
)
def test_on_seed_3_an_overconfident_rocket_never_checks_the_beacon(pad_spread, misses):
    on, off = fly_pad_pair(pad_spread=pad_spread)
    assert (round(on.miss, 2), round(off.miss, 2)) == misses
    checks = pad_spread > 0.1
    assert (nearest_beacon(on) < 0.5) is checks


WRONG_GUESS = (5.0, -2.0)
# Bonus 1's challenge: the rocket's guess moved 2 off the point the world draws round.


@pytest.mark.parametrize(
    ("pad_spread", "misses", "checks"),
    [(1.5, (0.23, 0.70), True), (0.1, (3.42, 3.78), False)],
)
def test_on_seed_3_only_a_rocket_unsure_of_a_wrong_guess_checks_the_beacon(
    pad_spread, misses, checks
):
    on, off = fly_pad_pair(pad_spread=pad_spread, guess=WRONG_GUESS)
    assert (round(on.miss, 2), round(off.miss, 2)) == misses
    assert (nearest_beacon(on) < 0.5) is checks


@pytest.mark.slow
@pytest.mark.parametrize(
    ("pad_spread", "expected", "visits"),
    [(1.5, PadTally(9, 0.24, 6, 0.43), 8), (0.1, PadTally(0, 2.59, 0, 2.59), 0)],
)
def test_over_ten_seeds_a_rocket_sure_of_a_wrong_guess_never_checks_or_lands(
    pad_spread, expected, visits
):
    pairs, tally = pad_tally(pad_spread=pad_spread, guess=WRONG_GUESS)
    assert tally == expected
    assert len(beacon_visits(on for on, _ in pairs)) == visits


@pytest.mark.slow
@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"pad_spread": 0.1}, PadTally(1, 1.42, 1, 1.42)),
        ({"pad_spread": 0.5}, PadTally(7, 0.30, 3, 0.70)),
        ({"pad_spread": 3.0}, PadTally(10, 0.25, 5, 0.55)),
        ({"broadcast_gain": 0.1}, PadTally(10, 0.18, 10, 0.17)),
        ({"broadcast_gain": 0.3}, PadTally(10, 0.19, 8, 0.24)),
        ({"broadcast_gain": 3.0}, PadTally(8, 0.27, 4, 0.58)),
        ({"broadcast": capped_broadcast}, PadTally(10, 0.25, 10, 0.14)),
        ({"radar_noise": 1.0}, PadTally(10, 0.23, 10, 0.17)),
        ({"precision": 0.3}, PadTally(7, 0.39, 3, 0.61)),
        ({"precision": 1.0}, PadTally(3, 0.61, 3, 0.61)),
        ({"horizon": 2}, PadTally(10, 0.22, 3, 0.61)),
    ],
    ids=lambda value: (
        ",".join(f"{k}={getattr(v, '__name__', v)}" for k, v in value.items())
        if isinstance(value, dict)
        else None
    ),
)
def test_over_ten_seeds_the_hidden_pads_balance_holds_only_in_a_window(
    settings, expected
):
    assert pad_tally(**settings)[1] == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    ("settings", "seeds"),
    [
        ({}, [0, 2, 3, 4, 5, 6, 7, 8, 9]),
        ({"precision": 0.3}, [2, 4, 5, 6, 8, 9]),
        ({"precision": 1.0}, [8]),
        ({"pad_spread": 0.1}, []),
    ],
    ids=["as built", "precision 0.3", "precision 1", "spread 0.1"],
)
def test_the_curious_rocket_checks_the_beacon_on_fewer_seeds_outside_the_window(
    settings, seeds
):
    pairs, _ = pad_tally(**settings)
    assert beacon_visits(on for on, _ in pairs) == seeds


@pytest.mark.slow
def test_two_moves_ahead_the_incurious_rocket_nears_the_beacon_on_seed_8_alone():
    pairs, _ = pad_tally(horizon=2)
    assert beacon_visits(off for _, off in pairs) == [8]


@pytest.mark.parametrize(
    ("pad_spread", "miss", "visits"),
    [(0.5, 0.31, True), (0.48, 0.31, True), (0.45, 1.69, False), (0.4, 1.57, False)],
)
def test_on_seed_3_the_curious_rocket_stops_finding_the_pad_below_a_spread_of_0_48(
    pad_spread, miss, visits
):
    on, _ = fly_pad_pair(pad_spread=pad_spread)
    assert round(on.miss, 2) == miss
    assert (nearest_beacon(on) < 0.5) is visits
    assert (on.miss < 0.5) is visits


@pytest.mark.slow
def test_over_ten_seeds_a_spread_of_0_4_lands_one_curious_rocket_fewer_than_0_5():
    assert pad_tally(pad_spread=0.4)[1].on_landed == 6
    assert pad_tally(pad_spread=0.5)[1].on_landed == 7


def test_on_seed_3_a_broadcast_that_carries_far_lets_both_rockets_land():
    on, off = fly_pad_pair(broadcast_gain=0.1)
    assert (round(on.miss, 2), round(off.miss, 2)) == (0.37, 0.06)


def test_on_seed_3_a_sharp_radar_lands_both_rockets_without_the_beacon():
    on, off = fly_pad_pair(radar_noise=1.0)
    assert (round(on.miss, 2), round(off.miss, 2)) == (0.38, 0.38)
    assert round(nearest_beacon(on), 2) == 1.40


@pytest.mark.slow
def test_with_a_sharp_radar_the_curious_rocket_still_detours_on_four_seeds():
    pairs, _ = pad_tally(radar_noise=1.0)
    assert beacon_visits(on for on, _ in pairs) == [2, 4, 6, 8]
    assert beacon_visits(off for _, off in pairs) == [8]


def test_on_seed_3_a_tight_preference_stops_the_curious_rocket_checking():
    # Seed 3 already stops checking at 0.3, where seven of ten seeds still land.
    for precision in (0.3, 1.0):
        on, _ = fly_pad_pair(precision=precision)
        assert round(nearest_beacon(on), 2) == 1.40
        assert on.miss > 0.5


def test_on_seed_3_two_moves_ahead_the_incurious_rocket_still_skips_the_beacon():
    on, off = fly_pad_pair(horizon=2)
    assert (round(on.miss, 2), round(off.miss, 2)) == (0.17, 0.85)
    assert round(nearest_beacon(off), 2) == 1.40
