import hashlib
import re
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

pytest.importorskip("pptx", reason="the deck extra builds the slides")

from cpomdp import (
    Agent,
    Belief,
    CouplingGraph,
    CouplingGraphBackend,
    FixedSensor,
    GaussianTransition,
    KalmanBackend,
    LinearGaussianModel,
    StateGoal,
)
from pptx import Presentation

import build_deck
import deck_layout
import lesson
from cpomdp_workshop import LAUNCH, PAD, model_with, pad_goal
from cpomdp_workshop.check import backend_differences
from deck_kit import without_imports

PROSE = ("title", "text", "card", "pill", "placeholder")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?:e-?\d+)?")
CLOCK = re.compile(r"\b\d+:\d\d\b")

MEASURED = {
    "0.000000": "test_no_move_of_the_1960_rocket_is_worth_more_than_any_other",
    "0.13": "test_the_1960_rocket_reports_where_it_lands",
    "1.41": "test_the_beacon_sensor_bends_the_path_to_the_beacon_and_lands",
    "0.24": "test_the_beacon_sensor_bends_the_path_to_the_beacon_and_lands",
    "1.49": "test_a_rocket_too_curious_to_land_stays_short_however_far_it_looks",
    "1.65": "test_a_rocket_too_curious_to_land_stays_short_however_far_it_looks",
    "1.86": "test_a_rocket_too_curious_to_land_stays_short_however_far_it_looks",
    "6": "test_the_1960_rocket_hits_the_rock_on_step_six_whatever_the_noise_does",
    "1.71": "test_the_default_slider_rocket_clears_the_rock_and_lands",
    "4.83": "test_the_default_slider_rocket_clears_the_rock_and_lands",
    "0.09": "test_the_default_slider_rocket_clears_the_rock_and_lands",
    "1.91": "test_the_k_point_two_rocket_clears_the_rock_and_lands",
    "5.18": "test_the_k_point_two_rocket_clears_the_rock_and_lands",
    "0.15": "test_the_k_point_two_rocket_clears_the_rock_and_lands",
    "5": "test_a_rock_in_clear_sky_stops_the_curious_rockets_on_step_five",
    "9": "test_the_search_is_decided_rather_than_sampled",
    "162": "test_a_second_step_of_lookahead_costs_eighteen_times_as_much",
    "78125": "test_the_coda_certificate_covers_every_seven_step_axis_plan",
    "1e-9": "test_the_kalman_filter_and_the_one_node_graph_agree",
    "0.5.0": "test_the_numbers_were_measured_on_the_pinned_release",
}
"""Each measured number in the prose or the notes, and the claim test asserting it."""

SET = {
    "0.08": "k, the slider's default",
    "0.20": "k, the second curious rocket",
    "0.4": "k, task 2's reference",
    "0.40": "k, task 2's reference",
    "0.6": "k, too curious to land",
    "0.5": "the compass step",
    "0": "the first seed",
    "1": "a task, poll, checkpoint, seed or answer number",
    "2": "a task, poll, checkpoint, seed or answer number",
    "3": "a task, poll, checkpoint, seed or answer number",
    "4": "an answer number, and a belief step drawn",
    "8": "a belief step drawn",
    "60": "a poll's length in seconds",
    "48": "the hours before the recording is posted",
    "1960": "the year of the controller",
    "2021": "the year of the reduction to LQR",
    "2026": "the year of the symposium",
    "3.11": "the oldest Python the workshop runs on",
}
"""Numbers the deck sets or names rather than measures."""


@pytest.fixture(scope="module")
def deck():
    return build_deck.build().presentation


def prose_and_notes(presentation):
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.name.split()[0] in PROSE:
                yield shape.text_frame.text
        yield slide.notes_slide.notes_text_frame.text


def pictures(slide):
    return [shape for shape in slide.shapes if shape.name.startswith("picture")]


@pytest.mark.slow
def test_the_deck_has_twenty_five_slides_each_with_notes(deck):
    assert len(deck.slides) == 25
    for slide in deck.slides:
        assert slide.notes_slide.notes_text_frame.text.strip()


@pytest.mark.slow
def test_no_shape_runs_off_a_slide_or_overlaps_another(deck):
    assert deck_layout.layout_problems(deck) == []


@pytest.mark.slow
def test_every_text_frame_holds_its_text(deck):
    measure = deck_layout.metrics()
    if measure is None:
        pytest.skip("no font files to measure with")
    assert deck_layout.fit_problems(deck, measure) == []


@pytest.mark.slow
def test_the_route_frames_build_up_in_one_place(deck):
    route = {9: "straight", 11: "beacon", 15: "blackout", 16: "rock"}
    places = set()
    for number, stage in route.items():
        (picture,) = pictures(deck.slides[number - 1])
        assert picture.name.endswith(f"_{stage}.png")
        places.add((picture.left, picture.top, picture.width, picture.height))
    assert len(places) == 1


@pytest.mark.slow
@pytest.mark.parametrize(
    ("number", "session"),
    [(8, lesson.SESSION_1), (10, lesson.SESSION_2), (13, lesson.SESSION_3)],
)
def test_a_task_slide_shows_the_lessons_own_code(deck, number, session):
    codes = [s for s in deck.slides[number - 1].shapes if s.name == "code"]
    shown = codes[0].text_frame.text.replace("\v", "\n").splitlines()
    assert [line.rstrip() for line in shown] == without_imports(
        session.attempt
    ).splitlines()


def test_the_pymdp_slide_code_builds_the_1960_rocket():
    namespace: dict[str, Any] = {
        "jnp": jnp,
        "Belief": Belief,
        "LinearGaussianModel": LinearGaussianModel,
        "StateGoal": StateGoal,
        "LAUNCH": LAUNCH,
        "PAD": PAD,
    }
    exec(build_deck.PYMDP_CODE, namespace)
    assert repr(Agent(namespace["model"], namespace["goal"])) == repr(
        Agent(model_with(), pad_goal())
    )


def test_the_graph_slide_code_flies_the_same_beliefs_as_the_kalman_filter():
    namespace: dict[str, Any] = {
        "jnp": jnp,
        "Agent": Agent,
        "Belief": Belief,
        "CouplingGraph": CouplingGraph,
        "CouplingGraphBackend": CouplingGraphBackend,
        "FixedSensor": FixedSensor,
        "GaussianTransition": GaussianTransition,
        "LAUNCH": LAUNCH,
        "goal": pad_goal(),
    }
    exec(build_deck.GRAPH_CODE, namespace)
    assert isinstance(namespace["rocket"], Agent)
    (difference,) = backend_differences(
        KalmanBackend(model_with()), namespace["backend"]
    )
    assert difference < 1e-9


@pytest.mark.slow
def test_every_number_in_the_prose_and_notes_is_measured_or_set(deck):
    numbers = {
        token
        for text in prose_and_notes(deck)
        for token in NUMBER.findall(CLOCK.sub("", text))
    }
    assert numbers - MEASURED.keys() - SET.keys() == set()


def test_every_measured_number_names_a_claim_test_that_exists():
    claims = (Path(__file__).parent / "test_measured_claims.py").read_text()
    for test_name in MEASURED.values():
        assert f"def {test_name}(" in claims, test_name


@pytest.mark.slow
def test_the_owners_placeholders_are_visible(deck):
    placeholders = [
        shape.text_frame.text
        for slide in deck.slides
        for shape in slide.shapes
        if shape.name == "placeholder"
    ]
    for owed in ("[photo]", "[bio]", "[email]", "[survey link]", "[Zenodo DOI"):
        assert any(text.startswith(owed) for text in placeholders), owed


@pytest.mark.slow
def test_claude_is_named_only_in_the_owners_disclosure(deck):
    naming = [text for text in prose_and_notes(deck) if "Claude" in text]
    assert naming[0] == build_deck.DISCLOSURE
    assert len(naming) == 2  # the caption and the Q&A answer in its notes
    assert naming[1].endswith("the answer is: some of it.")


def snapshot(presentation):
    slides = []
    for slide in presentation.slides:
        shapes = [
            (shape.name, shape.left, shape.top, shape.width, shape.height,
             shape.text_frame.text if shape.has_text_frame else
             hashlib.sha256(shape.image.blob).hexdigest())
            for shape in slide.shapes
        ]  # fmt: skip
        slides.append((shapes, slide.notes_slide.notes_text_frame.text))
    return slides


@pytest.mark.slow
def test_the_committed_deck_is_the_one_the_script_builds(deck):
    committed = Presentation(str(build_deck.DECK))
    assert snapshot(committed) == snapshot(deck)
