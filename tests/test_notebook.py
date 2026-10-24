import contextlib
import io
import re

import nbformat
import numpy as np
import pytest

import build_notebook
import lesson
from cpomdp_workshop import checkpoints
from cpomdp_workshop.world import default_params

HEAD = ["brief", "attempt", "symptoms", "hint", "hint", "reference"]
TAIL = ["checkpoint", "poll"]
HINTS = build_notebook.load_hints()


def first_tag(cell):
    return cell.metadata["tags"][0]


def rhythm(session):
    bonuses = []
    for bonus in session.bonuses:
        bonuses += ["bonus"] if bonus.code is None else ["bonus", "bonus-code"]
    return HEAD + bonuses + TAIL


def session_cells(notebook, number):
    briefs = [i for i, c in enumerate(notebook.cells) if first_tag(c) == "brief"]
    start = briefs[number - 1]
    return notebook.cells[start : start + len(rhythm(lesson.SESSIONS[number - 1]))]


def bonus_cells(notebook, number):
    return [c for c in session_cells(notebook, number) if first_tag(c) == "bonus"]


def run_quietly(source, namespace):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(source, namespace)


def test_the_hints_file_holds_six_hints_and_three_reference_lines():
    required = {
        (n, kind) for n in (1, 2, 3) for kind in ("hint 1", "hint 2", "reference")
    }
    assert required <= HINTS.keys()


def test_every_nudge_belongs_to_a_hint():
    nudges = {key for key in HINTS if key[1].startswith("nudge")}
    assert {(n, kind.replace("nudge", "hint")) for n, kind in nudges} <= HINTS.keys()


@pytest.mark.parametrize("key", [(n, f"hint {w}") for n in (1, 2, 3) for w in (1, 2)])
def test_every_hint_has_a_nudge(key):
    number, kind = key
    assert (number, kind.replace("hint", "nudge")) in HINTS


@pytest.mark.parametrize(
    "key", sorted(key for key in HINTS if key[1].startswith("nudge"))
)
def test_a_nudge_never_gives_away_its_hints_line(key):
    number, kind = key
    assert HINTS[(number, kind.replace("nudge", "hint"))] not in HINTS[key]


@pytest.mark.parametrize("number", [1, 2, 3])
@pytest.mark.parametrize("which", [1, 2])
def test_a_nudge_shows_above_its_hint_and_the_line_stays_behind_show_code(
    number, which
):
    notebook = build_notebook.build(HINTS)
    hints = [i for i, tag in enumerate(HEAD) if tag == "hint"]
    cell = session_cells(notebook, number)[hints[which - 1]]
    title, *nudge, line = cell.source.splitlines()
    assert title == f"# @title Hint {which}"
    expected = HINTS[(number, f"nudge {which}")].splitlines()
    assert nudge == [f"# @markdown {text}" for text in expected]
    assert line == HINTS[(number, f"hint {which}")]
    assert cell.metadata["cellView"] == "form"


@pytest.mark.parametrize(
    "session", lesson.SESSIONS, ids=lambda s: f"session {s.number}"
)
def test_every_attempt_leaves_exactly_two_lines_to_write(session):
    assert len(build_notebook.slots(session.attempt)) == 2


@pytest.mark.parametrize(
    "session", lesson.SESSIONS, ids=lambda s: f"session {s.number}"
)
@pytest.mark.parametrize("kind", ["hint 1", "hint 2"])
def test_every_hint_fills_one_slot_of_its_own_attempt(session, kind):
    filled = build_notebook.fill(session.attempt, HINTS[(session.number, kind)])
    assert len(build_notebook.slots(filled)) == 1


def test_task_2_sets_k_and_params_before_the_sensor_that_reads_them():
    lines = lesson.SESSION_2.attempt.splitlines()

    def first(prefix):
        return next(i for i, line in enumerate(lines) if line.startswith(prefix))

    assert first("beacon_gain =") < first("params =") < first("def my_sensor")


def shown(value):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return f"{float(value)}"
    return "(" + ", ".join(f"{float(entry)}" for entry in value) + ")"


def test_task_2_lists_every_setting_it_reads_with_its_true_value():
    attempt = lesson.SESSION_2.attempt
    listed = dict(re.findall(r'#\s+params\["(\w+)"\]\s+(.+)', attempt))
    hint = HINTS[(2, "hint 1")]
    read = set(re.findall(r'params\["(\w+)"\]', attempt + hint))
    assert read <= listed.keys()
    params = default_params(0.4)
    for key, description in listed.items():
        if key != "beacon_gain":  # k is the learner's own number
            assert description.split("  ")[0].strip() == shown(params[key]), key


@pytest.mark.parametrize("number", [1, 2, 3])
def test_every_task_has_two_numbered_bonuses(number):
    notebook = build_notebook.build(HINTS)
    headings = [c.source.splitlines()[0] for c in bonus_cells(notebook, number)]
    assert [heading.split(" · ")[0] for heading in headings] == [
        "### Bonus 1",
        "### Bonus 2",
    ]


@pytest.mark.parametrize("number", [1, 2, 3])
def test_every_bonus_shows_its_question_then_a_hint_then_an_answer(number):
    notebook = build_notebook.build(HINTS)
    for cell in bonus_cells(notebook, number):
        source = cell.source
        hint = source.index("<details><summary>Hint</summary>\n\n")
        answer = source.index("<details><summary>Answer</summary>\n\n")
        assert 0 < hint < answer
        assert source.count("</details>") == 2


@pytest.mark.parametrize("number", [1, 2])
def test_bonus_1_of_tasks_1_and_2_is_worked_in_the_attempt_cell(number):
    assert lesson.SESSIONS[number - 1].bonuses[0].code is None


def test_bonus_1_of_task_3_starts_from_a_seed_loop():
    code = lesson.SESSION_3.bonuses[0].code
    assert code is not None
    assert "for seed in seeds" in code


@pytest.mark.parametrize("number", [1, 2, 3])
def test_every_bonus_2_has_a_cell_to_work_in(number):
    assert lesson.SESSIONS[number - 1].bonuses[1].code is not None


def test_a_hint_that_names_no_slot_is_refused():
    with pytest.raises(ValueError, match="matches 0 slots"):
        build_notebook.fill(lesson.SESSION_1.attempt, "nonsense = 1")


@pytest.mark.parametrize("number", [1, 2, 3])
def test_every_session_keeps_the_rhythm(number):
    notebook = build_notebook.build(HINTS)
    session = lesson.SESSIONS[number - 1]
    assert [first_tag(c) for c in session_cells(notebook, number)] == rhythm(session)


@pytest.mark.parametrize("number", [1, 2, 3])
def test_the_reference_cell_is_the_attempt_with_the_hints_in(number):
    notebook = build_notebook.build(HINTS)
    reference = session_cells(notebook, number)[HEAD.index("reference")]
    expected = build_notebook.reference(lesson.SESSIONS[number - 1], HINTS)
    assert reference.source.split("\n", 1)[1] == expected


def test_the_attempts_and_hints_are_left_for_the_learner_to_run():
    notebook = build_notebook.build(HINTS)
    skipped = {
        first_tag(c)
        for c in notebook.cells
        if build_notebook.SKIP in c.metadata["tags"]
    }
    assert skipped == {"attempt", "hint", "bonus-code"}


def test_the_first_reference_cell_runs_in_a_fresh_namespace():
    run_quietly(build_notebook.reference(lesson.SESSION_1, HINTS), {})


@pytest.mark.slow
@pytest.mark.parametrize("number", [1, 2])
def test_a_checkpoint_restores_enough_to_run_the_next_reference_cell(number):
    namespace = dict(getattr(checkpoints, f"after_session_{number}")())
    next_session = lesson.SESSIONS[number]
    run_quietly(build_notebook.reference(next_session, HINTS), namespace)


@pytest.mark.slow
@pytest.mark.parametrize("number", [1, 2, 3])
def test_a_checkpoint_restores_enough_to_run_its_own_sessions_bonuses(number):
    for bonus in lesson.SESSIONS[number - 1].bonuses:
        if bonus.code is not None:
            namespace = dict(getattr(checkpoints, f"after_session_{number}")())
            run_quietly(bonus.code, namespace)


@pytest.mark.slow
def test_the_last_checkpoint_restores_enough_for_the_coda_and_the_tail():
    namespace = dict(checkpoints.after_session_3())
    run_quietly(lesson.CODA, namespace)
    for _, source in lesson.TAIL:
        run_quietly(source, namespace)


def test_a_rebuild_keeps_every_cell_id():
    first, again = build_notebook.build(HINTS), build_notebook.build(HINTS)
    assert [c.id for c in first.cells] == [c.id for c in again.cells]


def test_the_committed_notebook_is_the_built_one():
    # A failure after the gate ran means ruff rewrote a cell: write that cell of the
    # lesson in ruff's style, then rebuild.
    committed = nbformat.read(build_notebook.NOTEBOOK, as_version=4)
    built = build_notebook.build(HINTS)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in built.cells
    ]


@pytest.mark.slow
def test_the_notebook_runs_end_to_end_on_the_reference_path():
    notebook = build_notebook.run(build_notebook.build(HINTS))
    errors = [
        o
        for c in notebook.cells
        for o in c.get("outputs", [])
        if o.output_type == "error"
    ]
    assert errors == []
    printed = "".join(
        o.text
        for c in notebook.cells
        for o in c.get("outputs", [])
        if o.output_type == "stream"
    )
    assert "spread = 0.000000 nats" in printed
    assert "PROVED (set compass-9, |A|^H = 9^1 = 9, visited 9)" in printed
    assert "max |difference| = " in printed
