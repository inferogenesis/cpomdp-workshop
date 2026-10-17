import contextlib
import io

import nbformat
import pytest

import build_notebook
import lesson
from cpomdp_workshop import checkpoints

RHYTHM = [
    "brief",
    "attempt",
    "symptoms",
    "hint",
    "hint",
    "reference",
    "bonus",
    "checkpoint",
    "poll",
]
HINTS = build_notebook.load_hints()


def first_tag(cell):
    return cell.metadata["tags"][0]


def session_cells(notebook, number):
    briefs = [i for i, c in enumerate(notebook.cells) if first_tag(c) == "brief"]
    start = briefs[number - 1]
    return notebook.cells[start : start + len(RHYTHM)]


def run_quietly(source, namespace):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(source, namespace)


def test_the_hints_file_holds_six_hints_and_three_reference_lines():
    kinds = sorted(HINTS)
    assert kinds == sorted(
        (n, kind) for n in (1, 2, 3) for kind in ("hint 1", "hint 2", "reference")
    )


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


def test_a_hint_that_names_no_slot_is_refused():
    with pytest.raises(ValueError, match="matches 0 slots"):
        build_notebook.fill(lesson.SESSION_1.attempt, "nonsense = 1")


@pytest.mark.parametrize("number", [1, 2, 3])
def test_every_session_keeps_the_rhythm(number):
    notebook = build_notebook.build(HINTS)
    assert [first_tag(c) for c in session_cells(notebook, number)] == RHYTHM


@pytest.mark.parametrize("number", [1, 2, 3])
def test_the_reference_cell_is_the_attempt_with_the_hints_in(number):
    notebook = build_notebook.build(HINTS)
    reference = session_cells(notebook, number)[RHYTHM.index("reference")]
    expected = build_notebook.reference(lesson.SESSIONS[number - 1], HINTS)
    assert reference.source.split("\n", 1)[1] == expected


def test_the_attempts_and_hints_are_left_for_the_learner_to_run():
    notebook = build_notebook.build(HINTS)
    skipped = {
        first_tag(c)
        for c in notebook.cells
        if build_notebook.SKIP in c.metadata["tags"]
    }
    assert skipped == {"attempt", "hint"}


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
def test_a_checkpoint_restores_enough_to_run_its_own_sessions_bonus(number):
    namespace = dict(getattr(checkpoints, f"after_session_{number}")())
    run_quietly(lesson.SESSIONS[number - 1].bonus, namespace)


@pytest.mark.slow
def test_the_last_checkpoint_restores_enough_for_the_coda_and_the_tail():
    namespace = dict(checkpoints.after_session_3())
    run_quietly(lesson.CODA, namespace)
    for _, source in lesson.TAIL:
        run_quietly(source, namespace)


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
