import inspect
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("pptx", reason="the deck extra builds the slides")
pytest.importorskip("segno", reason="the deck extra draws the QR codes")

import cpomdp
import segno
from PIL import Image
from pptx.opc.constants import RELATIONSHIP_TYPE
from pptx.shapes.picture import Movie
from pptx.util import Emu

import build_deck
import deck_layout
import lesson
import timer_clip
from cpomdp_workshop import (
    WORKSHOP_CPOMDP_VERSION,
    plot,
)

PROSE = ("title", "text", "card", "pill", "placeholder")
DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION = "http://schemas.openxmlformats.org/presentationml/2006/main"
NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)*(?:e-?\d+)?")
CLOCK = re.compile(r"\b\d+:\d\d\b")
NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "build_a_curious_rocket.ipynb"

MEASURED = {
    "0.000000": "test_no_move_of_the_1960_rocket_is_worth_more_than_any_other",
    "0.13": "test_the_1960_rocket_reports_where_it_lands",
    "1.41": "test_the_beacon_sensor_bends_the_path_to_the_beacon_and_lands",
    "0.23": "test_at_k_0_08_curiosity_flies_to_the_beacon_and_both_rockets_land",
    "0.29": "test_at_k_0_08_curiosity_flies_to_the_beacon_and_both_rockets_land",
    "36": "test_at_the_starting_k_curiosity_decides_six_of_the_36_moves",
    "0.24": "test_at_k_0_4_the_curious_rocket_stops_short_and_the_incurious_one_lands",
    "2.79": "test_at_k_0_4_the_curious_rocket_stops_short_and_the_incurious_one_lands",
    "1.67": "test_at_k_0_4_no_curious_rocket_lands_and_every_incurious_one_does",
    "0.09": "test_at_k_0_08_curiosity_flies_to_the_beacon_and_both_rockets_land",
    "0.17": "test_on_seed_3_the_curious_rocket_finds_the_pad_and_the_other_misses",
    "0.85": "test_on_seed_3_the_curious_rocket_finds_the_pad_and_the_other_misses",
    "0.25": "test_gate_c2_curious_lands_on_8_or_more_seeds_and_incurious_on_4_or_fewer",
    "0.61": "test_gate_c2_curious_lands_on_8_or_more_seeds_and_incurious_on_4_or_fewer",
    "6": "test_the_curious_rocket_checks_the_beacon_on_fewer_seeds_outside_the_window",
    "9": "test_the_search_is_decided_rather_than_sampled",
    "81": "test_two_moves_ahead_scores_eighty_one_plans_a_step",
    "78125": "test_the_coda_certificate_covers_every_seven_step_axis_plan",
    "1e-9": "test_the_kalman_filter_and_the_one_node_graph_agree",
    WORKSHOP_CPOMDP_VERSION: "test_the_numbers_were_measured_on_the_pinned_release",
}
"""Each measured number in the prose or the notes, and the claim test asserting it."""

SET = {
    "0.08": "k, the slider's default",
    "0.4": "k, task 2's reference, where its two rockets part",
    "0.1": "the curious rocket's preference precision",
    "0.2": "k, the precision comparison's",
    "0.48": "the pad spread at which seed 3's curious rocket still finds the pad",
    "1.5": "the hidden pad's spread, in the world and in the rocket's model",
    "5": "a timer's mark, and the minutes of a brief",
    "0.37": "k, where task 2's curious rocket first stops short on seed 3",
    "0.45": "task 2's last k at which the incurious rocket lands on seed 3, and "
    "the pad spread at which task 3's curious rocket stops finding the pad",
    "0.46": "k, where task 2's incurious rocket first stops short on seed 3",
    "0.40": "k, task 2's reference",
    "0.5": "the compass step",
    "0": "the first seed, and the start of a timer's bar",
    "1": "a task, checkpoint, seed, step or answer number",
    "2": "a task, checkpoint, seed, step or answer number",
    "3": "a task, checkpoint, seed, step or answer number",
    "4": "an answer number",
    "10": "the seeds a bonus flies, and a timer's mark",
    "15": "the minutes each task codes for, and a timer's mark",
    "20": "the minutes task 2's timer runs, with its break, and that timer's mark",
    "45": "the minutes of code in all",
    "48": "the hours before the recording is posted",
    "1960": "the year of the controller",
    "2007": "the year of the cited Franklin arm-stiffness study",
    "2021": "the year of the cited Koudahl result and the Mattingly chemotaxis paper",
    "2026": "the year of the symposium",
    "2.2": "a figure number in the Active Inference book",
    "2.3": "the related-work section of the cited Corva paper",
    "2.6": "an equation number in the Active Inference book",
    "3.11": "the oldest Python the workshop runs on",
    "2607.20306": "the arXiv number of presenter 1's paper",
    **dict.fromkeys(
        ("21", "28", "52", "196", "201", "203", "231", "232", "233"),
        "a cpomdp issue number on the roadmap slide",
    ),
    "10.3390": "the DOI registrant prefix of the cited Koudahl paper",
    "10.5281": "the Zenodo DOI prefix for the cited cpomdp release",
}
"""Numbers the deck sets or names rather than measures."""


@pytest.fixture(scope="module")
def deck():
    return build_deck.build().presentation


def slide_of(deck, slide):
    return deck.slides[build_deck.number_of(slide) - 1]


def prose_and_notes(presentation):
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.name.split()[0] in PROSE:
                yield shape.text_frame.text
            elif shape.has_table:
                yield from (cell.text for cell in shape.table.iter_cells())
        yield slide.notes_slide.notes_text_frame.text


def pictures(slide):
    return [shape for shape in slide.shapes if shape.name.startswith("picture")]


@pytest.mark.slow
def test_the_deck_has_twenty_three_slides_each_with_notes(deck):
    assert len(deck.slides) == 23
    for slide in deck.slides:
        assert slide.notes_slide.notes_text_frame.text.strip()


def minutes_of(kind):
    return sum(
        part.minutes
        for block in build_deck.AGENDA
        for part in block.parts
        if part.kind == kind
    )


def test_the_two_hours_are_65_minutes_of_talk_45_of_code_a_5_break_and_5_open():
    minutes = [minutes_of(kind) for kind in ("talk", "code", "break", "open")]
    assert minutes == [65, 45, 5, 5]


def test_every_part_of_the_agenda_is_a_whole_number_of_five_minutes():
    parts = [part for block in build_deck.AGENDA for part in block.parts]
    assert all(part.minutes % 5 == 0 for part in parts)
    assert sum(part.minutes for part in parts) == 120


def test_each_task_codes_for_as_long_as_its_timer_runs():
    tasks = [block for block in build_deck.AGENDA if block.label.startswith("Task")]
    assert [block.label for block in tasks] == ["Task 1", "Task 2", "Task 3"]
    for block in tasks:
        working = [part for part in block.parts if part.kind != "break"]
        assert [part.name for part in working] == ["brief", "you code", "discussion"]
        assert [part.kind for part in working] == ["talk", "code", "talk"]
        assert working[1].minutes == build_deck.TASK_MINUTES == 15


def test_the_one_break_is_five_minutes_between_task_2s_brief_and_its_coding():
    breaks = {
        block.label: [part.name for part in block.parts]
        for block in build_deck.AGENDA
        if any(part.kind == "break" for part in block.parts)
    }
    assert breaks == {"Task 2": ["brief", "break", "you code", "discussion"]}
    assert minutes_of("break") == 5


def test_a_timer_runs_for_the_coding_and_for_task_2s_break_before_it():
    minutes = {
        block.label: block.timer_minutes
        for block in build_deck.AGENDA
        if block.timer_minutes
    }
    assert minutes == {"Task 1": 15, "Task 2": 20, "Task 3": 15}
    assert build_deck.TIMER_MINUTES == (15, 20)
    task_2 = build_deck.block_called("Task 2")
    assert [part.name for part in task_2.timed] == ["break", "you code"]


def test_the_clock_reads_each_utc_start_off_the_agenda():
    assert build_deck.clock("Welcome") == "17:00"
    assert build_deck.clock("Task 1") == "17:10"
    assert build_deck.clock("Task 1", "you code") == "17:15"
    assert (
        build_deck.clock("Task 1", "you code", after=build_deck.TASK_MINUTES) == "17:30"
    )
    assert build_deck.clock("Task 1", "discussion") == "17:30"
    assert build_deck.clock("Task 2") == "17:40"
    assert build_deck.clock("Task 2", "break") == "17:45"
    assert build_deck.clock("Task 2", "you code") == "17:50"
    assert build_deck.clock("Task 2", "discussion") == "18:05"
    assert build_deck.clock("Task 3") == "18:15"
    assert build_deck.clock("Why it matters") == "18:45"
    assert build_deck.clock("Questions") == "18:55"
    assert build_deck.span(build_deck.AGENDA[-1]) == "18:55–19:00"
    with pytest.raises(ValueError, match="no block called 'Break'"):
        build_deck.clock("Break")
    with pytest.raises(ValueError, match="no part called 'break' in Task 1"):
        build_deck.clock("Task 1", "break")


@pytest.mark.slow
def test_the_agenda_names_every_part_with_a_tasks_minutes_and_each_rows_times(deck):
    slide = slide_of(deck, build_deck.agenda)
    blocks = [shape.text_frame.text for shape in slide.shapes if shape.name == "block"]
    expected = [
        f"{part.name}\n{part.minutes} minutes" if len(block.parts) > 1 else part.name
        for block in build_deck.AGENDA
        for part in block.parts
    ]
    assert blocks == expected
    assert blocks[1:4] == ["brief\n5 minutes", "you code\n15 minutes",
                           "discussion\n10 minutes"]  # fmt: skip
    assert blocks[4:8] == [
        "brief\n5 minutes",
        "break\n5 minutes",
        "you code\n15 minutes",
        "discussion\n10 minutes",
    ]
    texts = [shape.text_frame.text for shape in slide.shapes if shape.name == "text"]
    for block in build_deck.AGENDA:
        assert any(text.endswith(build_deck.span(block)) for text in texts)
    assert any(text.startswith("All times UTC.") for text in texts)


TIMERS = [
    (build_deck.task_1_timer, 15, "17:15", "17:30"),
    (build_deck.task_2_timer, 20, "17:45", "18:05"),
    (build_deck.task_3_timer, 15, "18:20", "18:35"),
]
"""Each timer slide, the minutes it runs and when it is planned to start and end."""


@pytest.mark.slow
@pytest.mark.parametrize(("timer", "minutes", "start", "end"), TIMERS)
def test_a_timer_fills_its_bar_over_its_minutes_with_no_click(
    deck, timer, minutes, start, end
):
    slide = slide_of(deck, timer)
    (bar,) = [shape for shape in slide.shapes if shape.name == "bar"]
    timing = slide._element.find(f"{{{PRESENTATION}}}timing")
    assert timing is not None
    wipe = timing.find(f".//{{{PRESENTATION}}}animEffect")
    assert wipe is not None
    assert wipe.get("filter") == "wipe(left)"
    behaviour = wipe.find(f"{{{PRESENTATION}}}cBhvr")
    assert behaviour is not None
    duration = behaviour.find(f"{{{PRESENTATION}}}cTn")
    target = behaviour.find(f".//{{{PRESENTATION}}}spTgt")
    assert duration is not None
    assert target is not None
    assert duration.get("dur") == str(minutes * 60_000)
    assert target.get("spid") == str(bar.shape_id)
    starts = timing.findall(f".//{{{PRESENTATION}}}cond[@evt='onBegin']")
    assert len(starts) == 1, "the effects start with the slide, not on a click"


def schedule(slide):
    """Each effect on a slide as (shape id, entr or exit, ms after the slide shows)."""
    timing = slide._element.find(f"{{{PRESENTATION}}}timing")
    assert timing is not None
    effects = []
    for node in timing.iter(f"{{{PRESENTATION}}}cTn"):
        kind = node.get("presetClass")
        if kind is None:
            continue
        cond = node.find(f"{{{PRESENTATION}}}stCondLst/{{{PRESENTATION}}}cond")
        target = node.find(f".//{{{PRESENTATION}}}spTgt")
        assert cond is not None
        assert target is not None
        effects.append((int(target.get("spid")), kind, int(cond.get("delay"))))
    return effects


@pytest.mark.slow
@pytest.mark.parametrize(("timer", "minutes", "start", "end"), TIMERS)
def test_a_timers_notes_say_when_it_starts_and_ends_and_how_long_it_stays_up(
    deck, timer, minutes, start, end
):
    said = slide_of(deck, timer).notes_slide.notes_text_frame.text
    assert f"Planned at {start}, until {end}.]" in said
    assert f"Leave this slide up for the full {minutes} minutes." in said


@pytest.mark.slow
@pytest.mark.parametrize(("timer", "minutes", "start", "end"), TIMERS)
def test_a_timer_plays_its_clock_above_its_bar_from_when_the_slide_shows(
    deck, timer, minutes, start, end
):
    slide = slide_of(deck, timer)
    (track,) = [shape for shape in slide.shapes if shape.name == "track"]
    (bar,) = [shape for shape in slide.shapes if shape.name == "bar"]
    (clock,) = [shape for shape in slide.shapes if shape.name == "clock"]
    (times_up,) = [
        shape
        for shape in slide.shapes
        if shape.has_text_frame and shape.text_frame.text.startswith("Time's up.")
    ]
    assert isinstance(clock, Movie)
    assert clock.top + clock.height <= track.top
    assert clock.left + clock.width == track.left + track.width
    assert len(slide._element.findall(f"{{{PRESENTATION}}}timing")) == 1
    effects = schedule(slide)
    assert (bar.shape_id, "entr", 0) in effects
    assert (clock.shape_id, "mediacall", 0) in effects
    assert (times_up.shape_id, "entr", minutes * 60_000) in effects
    media = slide._element.findall(
        f".//{{{PRESENTATION}}}video//{{{PRESENTATION}}}spTgt"
    )
    assert [target.get("spid") for target in media] == [str(clock.shape_id)]


@pytest.mark.slow
@pytest.mark.parametrize(("timer", "minutes", "start", "end"), TIMERS)
def test_a_timer_embeds_the_committed_clip_and_shows_its_full_time_until_it_plays(
    deck, timer, minutes, start, end
):
    slide = slide_of(deck, timer)
    (clock,) = [shape for shape in slide.shapes if shape.name == "clock"]
    assert isinstance(clock, Movie)
    poster = clock.poster_frame
    assert poster is not None
    assert poster.blob == timer_clip.poster(minutes).read_bytes()
    (video,) = clock._element.xpath("./p:nvPicPr/p:nvPr/a:videoFile/@r:link")
    assert slide.part.related_part(video).blob == timer_clip.clip(minutes).read_bytes()


def test_every_slide_is_on_the_light_palette(deck):
    for slide in deck.slides:
        assert str(slide.background.fill.fore_color.rgb) == plot.LIGHT.background[1:]


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
    route = {
        build_deck.task_1_result: "straight",
        build_deck.task_2_result: "beacon",
        build_deck.task_2_meaning: "too_curious",
        build_deck.task_3_result: "pad",
        build_deck.preview: "pad",
    }
    corners, sizes = set(), set()
    for slide, stage in route.items():
        (picture,) = pictures(slide_of(deck, slide))
        assert picture.name.endswith(f"_{stage}.png")
        corners.add((picture.left, picture.top, picture.width))
        sizes.add((stage == "pad", picture.height))
    assert len(corners) == 1
    assert len(sizes) == 2  # the pad's figure carries its readout, so it is shorter


def test_every_number_in_the_prose_and_notes_is_measured_or_set(deck):
    numbers = {
        token.replace(",", "")
        for text in prose_and_notes(deck)
        for token in NUMBER.findall(CLOCK.sub("", text))
    }
    assert numbers - MEASURED.keys() - SET.keys() == set()


def test_every_measured_number_names_a_claim_test_that_exists():
    claims = (Path(__file__).parent / "test_measured_claims.py").read_text()
    for test_name in MEASURED.values():
        assert f"def {test_name}(" in claims, test_name


def photo_frames(slide):
    return [shape for shape in slide.shapes if shape.name.startswith("photo ")]


def cards_of(slide):
    return [shape.text_frame for shape in slide.shapes if shape.name == "card"]


@pytest.mark.slow
def test_the_unused_features_follow_what_cpomdp_let_us_do_in_three_cards(deck):
    assert build_deck.number_of(build_deck.unused_features) == (
        build_deck.number_of(build_deck.where_cpomdp_sits) + 1
    )
    cards = cards_of(slide_of(deck, build_deck.unused_features))
    assert [card.paragraphs[0].text for card in cards] == [
        group.heading for group in build_deck.UNUSED_FEATURES
    ]
    lines = [[paragraph.text for paragraph in card.paragraphs[1:]] for card in cards]
    assert lines == [
        [feature.title for feature in group.features]
        for group in build_deck.UNUSED_FEATURES
    ]
    assert [len(card) for card in lines] == [3, 3, 3]


def links_of(slide):
    return {
        run.hyperlink.address
        for shape in slide.shapes
        if shape.has_text_frame
        for paragraph in shape.text_frame.paragraphs
        for run in paragraph.runs
    }


@pytest.mark.slow
def test_the_proved_slide_names_every_warrant_and_where_the_labels_come_from(deck):
    slide = slide_of(deck, build_deck.coda)
    (table,) = [shape.table for shape in slide.shapes if shape.has_table]
    labels = [row.cells[0].text for row in table.rows][1:]
    assert labels == [warrant.value for warrant in cpomdp.Warrant]
    assert build_deck.WARRANTLIB in links_of(slide)
    said = " ".join(
        [shape.text_frame.text for shape in slide.shapes if shape.has_text_frame]
        + [slide.notes_slide.notes_text_frame.text]
    )
    assert "warrantlib" in said
    # The ten-seed result carries no label in the notebook, so none is claimed for it.
    assert "At most" in table.rows[3].cells[2].text


@pytest.mark.slow
def test_what_cpomdp_let_us_do_is_three_cards_of_three_lines_and_a_link(deck):
    slide = slide_of(deck, build_deck.where_cpomdp_sits)
    cards = cards_of(slide)
    assert [len(card.paragraphs) - 1 for card in cards] == [3, 3, 3]
    assert build_deck.REPOSITORY in links_of(slide)


@pytest.mark.slow
@pytest.mark.parametrize(
    "listing", [build_deck.where_cpomdp_sits, build_deck.unused_features]
)
def test_each_listed_item_takes_one_line_of_its_card(deck, listing):
    measure = deck_layout.metrics()
    if measure is None:
        pytest.skip("no font files to measure with")
    slide = slide_of(deck, listing)
    for card in (shape for shape in slide.shapes if shape.name == "card"):
        frame = card.text_frame
        room = Emu(card.width - frame.margin_left - frame.margin_right).inches
        for paragraph in frame.paragraphs[1:]:
            size = paragraph.runs[0].font.size.pt
            width = measure.width(paragraph.text, "Calibri", size, bold=False)
            assert width <= room, paragraph.text


def test_every_unused_feature_is_public_in_the_pinned_cpomdp_and_not_in_the_notebook():
    notebook = NOTEBOOK.read_text()
    for group in build_deck.UNUSED_FEATURES:
        for feature in group.features:
            assert feature.names, feature.title
            assert set(feature.names) <= set(cpomdp.__all__), feature.names
            if feature.argument:
                # The room met the name, so the unused part is its argument.
                provider = getattr(cpomdp, feature.names[0])
                assert feature.argument in inspect.signature(provider).parameters
                unused = [rf"\b{feature.argument}\s*="]
            else:
                unused = [rf"\b{name}\b" for name in feature.names]
            for pattern in unused:
                assert not re.search(pattern, notebook), pattern


@pytest.mark.slow
def test_the_unused_features_notes_name_each_cpomdp_class_for_whoever_asks(deck):
    slide = slide_of(deck, build_deck.unused_features)
    notes = slide.notes_slide.notes_text_frame.text
    for group in build_deck.UNUSED_FEATURES:
        for feature in group.features:
            assert feature.lookup in notes, feature.lookup


@pytest.mark.slow
def test_the_roadmap_follows_the_unused_features_with_nine_issues_and_no_promises(
    deck,
):
    assert build_deck.number_of(build_deck.roadmap) == (
        build_deck.number_of(build_deck.unused_features) + 1
    )
    slide = slide_of(deck, build_deck.roadmap)
    cards = cards_of(slide)
    assert len(cards) == 3
    issues = [paragraph.text for card in cards for paragraph in card.paragraphs[1:]]
    assert len(issues) == 9
    assert all(re.match(r"#\d+\xa0+\S", line) for line in issues)
    # Each issue title sits on one line beside its number.
    assert max(len(line) for line in issues) <= 31
    words = " ".join(
        [shape.text_frame.text for shape in slide.shapes if shape.has_text_frame]
        + [slide.notes_slide.notes_text_frame.text]
    ).lower()
    # No release is promised, and no collaboration is implied.
    assert not re.search(r"\bv1\b|pymdp", words)
    assert {build_deck.CPOMDP_ISSUES, build_deck.CPOMDP_DISCUSSIONS} <= links_of(slide)


@pytest.mark.slow
def test_each_presenters_card_has_a_photo_frame_and_their_bio(deck):
    slide = slide_of(deck, build_deck.who_we_are)
    assert not pictures(slide), "the committed deck carries no photos"
    frames = photo_frames(slide)
    assert [frame.name for frame in frames] == [
        "photo presenter_1.jpg",
        "photo presenter_2.jpg",
    ]
    assert all(frame.width == frame.height for frame in frames)
    texts = [shape.text_frame.text for shape in slide.shapes if shape.name == "text"]
    assert {"Dan", "Kevin"} <= set(texts)
    for bio, opening in (
        (build_deck.PRESENTER_1_BIO, "Independent research software engineer, UK."),
        (build_deck.PRESENTER_2_BIO, "Kevin Backhouse is a PhD researcher"),
    ):
        assert bio.startswith(opening)
        assert bio in texts
    assert not [shape for shape in slide.shapes if shape.name == "placeholder"]


@pytest.mark.slow
def test_the_presenting_copy_puts_each_photo_in_its_frame(tmp_path):
    deck = build_deck.build()
    missing = re.escape("no presenter_1.jpg, presenter_2.jpg")
    with pytest.raises(FileNotFoundError, match=missing):
        deck.fill_photo_frames(tmp_path)
    slide = slide_of(deck.presentation, build_deck.who_we_are)
    assert not pictures(slide), "a missing photo stops the copy before any goes in"
    for name in ("presenter_1.jpg", "presenter_2.jpg"):
        Image.new("RGB", (8, 8), "grey").save(tmp_path / name)
    deck.fill_photo_frames(tmp_path)
    photos = pictures(slide)
    assert [photo.name for photo in photos] == [
        "picture presenter_1.jpg",
        "picture presenter_2.jpg",
    ]
    assert [(p.left, p.top, p.width, p.height) for p in photos] == [
        (f.left, f.top, f.width, f.height) for f in photo_frames(slide)
    ]


@pytest.mark.slow
def test_who_we_are_introduces_cpomdp_without_the_mark_or_warrantlib(deck):
    slide = slide_of(deck, build_deck.who_we_are)
    assert "mark" not in [shape.name for shape in slide.shapes]
    texts = [shape.text_frame.text for shape in slide.shapes if shape.has_text_frame]
    assert any(text.startswith("cpomdp\n") for text in texts)
    said = " ".join([*texts, slide.notes_slide.notes_text_frame.text])
    assert "warrantlib" not in said


@pytest.mark.slow
@pytest.mark.parametrize(
    "timer",
    [build_deck.task_1_timer, build_deck.task_2_timer, build_deck.task_3_timer],
)
def test_a_timer_slide_says_how_the_notebook_marks_the_lines_and_where_hints_are(
    deck, timer
):
    slide = slide_of(deck, timer)
    texts = [shape.text_frame.text for shape in slide.shapes if shape.name == "text"]
    assert any("Fill in the two lines marked ..." in text for text in texts)
    assert any("Open Hint 1, Hint 2 or the Reference below the cell." in text
               for text in texts)  # fmt: skip
    said = " ".join(texts)
    assert "amber" not in said
    assert not [s for s in slide.shapes if s.name == "code"]


@pytest.mark.slow
def test_the_setup_slide_gives_the_local_route_that_runs_and_the_question_form(deck):
    slide = slide_of(deck, build_deck.setup)
    code = " ".join(s.text_frame.text for s in slide.shapes if s.name == "code")
    assert "git clone" in code
    assert "python -m venv .venv" in code
    assert '"cpomdp-workshop[nb]==' in code
    links = [s.text_frame.text for s in slide.shapes if s.name == "link"]
    assert build_deck.QUESTIONS_FORM.removeprefix("https://") in links
    assert build_deck.QUESTIONS_FORM in lesson.TITLE


@pytest.mark.slow
def test_each_learner_opens_the_hints_and_nobody_pastes_them(deck):
    said = " ".join(prose_and_notes(deck))
    assert "on a timer" not in said
    assert "Paste the nudges" not in said
    slide = slide_of(deck, build_deck.how_we_work)
    cards = [shape.text_frame.text for shape in slide.shapes if shape.name == "card"]
    assert any("Click to open them whenever you like" in card for card in cards)
    assert any("bonus exercises" in card for card in cards)


def drawn_modules(shape):
    """The dark modules a QR shape draws, one rectangle per run along a row."""
    corners = [
        (int(point.get("x")), int(point.get("y")))
        for point in shape._element.iter(f"{{{DRAWING}}}pt")
    ]
    dark = set()
    for start in range(0, len(corners), 4):
        (left, row), (right, _) = corners[start : start + 2]
        dark.update((row, column) for column in range(left, right))
    return dark


@pytest.mark.slow
@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/in/kevin-backhouse-112554253",
        "https://inferogenesis.com/",
        "https://github.com/inferogenesis/cpomdp",
        "https://cpomdp.inferogenesis.com/",
        "https://cpomdp.inferogenesis.com/api/warrant/",
    ],
)
def test_the_thanks_slide_draws_each_qr_module_for_module(deck, url):
    slide = slide_of(deck, build_deck.thanks)
    (code,) = [shape for shape in slide.shapes if shape.name == f"qr {url}"]
    assert code.width == code.height
    matrix = segno.make_qr(url).matrix
    expected = {
        (row, column)
        for row, line in enumerate(matrix)
        for column, module in enumerate(line)
        if module
    }
    assert drawn_modules(code) == expected


@pytest.mark.slow
def test_the_thanks_slide_keeps_four_modules_clear_between_its_five_qr_codes(deck):
    slide = slide_of(deck, build_deck.thanks)
    codes = [shape for shape in slide.shapes if shape.name.startswith("qr ")]
    assert len(codes) == 5
    for code in codes:
        url = code.name.removeprefix("qr ")
        quiet = 4 * code.width / len(segno.make_qr(url).matrix)
        for other in codes:
            if other is code:
                continue
            apart = max(
                other.left - (code.left + code.width),
                code.left - (other.left + other.width),
                other.top - (code.top + code.height),
                code.top - (other.top + other.height),
            )
            assert apart >= quiet, (url, other.name)


def test_links_are_drawn_in_ink_not_the_templates_dark_blue(deck):
    theme = deck.slide_master.part.part_related_by(RELATIONSHIP_TYPE.THEME)
    scheme = ET.fromstring(theme.blob).find(f".//{{{DRAWING}}}clrScheme")
    assert scheme is not None
    for name in ("hlink", "folHlink"):
        colour = scheme.find(f"{{{DRAWING}}}{name}/{{{DRAWING}}}srgbClr")
        assert colour is not None
        assert colour.get("val") == plot.LIGHT.ink.lstrip("#")


@pytest.mark.slow
def test_the_committed_deck_is_the_one_the_script_builds(deck):
    # Compare package contents, including formatting and shape IDs. ZIP entry
    # timestamps record when PowerPoint was built and are outside the contents.
    buffer = BytesIO()
    deck.save(buffer)
    with ZipFile(build_deck.DECK) as committed, ZipFile(buffer) as rebuilt:
        assert set(committed.namelist()) == set(rebuilt.namelist())
        for name in rebuilt.namelist():
            assert committed.read(name) == rebuilt.read(name), name
