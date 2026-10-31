"""Build the workshop deck from the package, the lesson and the figures.

    uv run --no-sync python scripts/render_assets.py   # the figures, first
    uv run --no-sync python scripts/build_deck.py

The two hours run as an introduction, three tasks and a closing section on why cpomdp
matters, then time left open for questions. Each task is a brief, the code, a slide
that stays up with a fifteen-minute bar while the room works, and a discussion of
what the room built. ``AGENDA`` holds the plan, and every clock time in the notes is
read from it.

The slides carry the route figures, the code the room writes and what the cells
print. Code comes from ``lesson.py``, printed output from running the package on the
pinned release, figures from ``assets/``. Every number in the prose and the notes is
asserted in ``tests/test_measured_claims.py`` or named in the deck test as set. The
build refuses to write a deck whose shapes run off a slide, overlap or overflow.

The presenters' photos are never committed. The committed deck shows an empty frame
for each. When ``deck/photos/`` holds them, the build also writes a presenting copy
with each photo in its frame, and git ignores both the photos and that copy.
"""

import contextlib
import io
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from cpomdp import KalmanBackend
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

import lesson
from cpomdp_workshop import (
    WORKSHOP_CPOMDP_VERSION,
    after_session_3,
    beacon_sensor,
    check_backends,
    check_my_sensor,
    default_params,
    model_with,
    one_node,
)
from deck_kit import (
    AMBER,
    AMBER_MARK,
    BLUE,
    CODE_FONT,
    CONTENT_TOP,
    GREEN,
    INK,
    MARGIN,
    MATH_FONT,
    MUTED,
    SLIDE_WIDTH,
    Deck,
    Page,
    Paragraph,
    Run,
    code_height,
    code_size,
    without_imports,
)
from deck_layout import fit_problems, layout_problems, metrics

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DECK = ROOT / "deck" / "cpomdp_symposium_2026.pptx"
PHOTOS = ROOT / "deck" / "photos"
PRESENTING = ROOT / "deck" / "cpomdp_symposium_2026_presenting.pptx"

REPOSITORY = "https://github.com/inferogenesis/cpomdp-workshop"
COLAB = (
    "https://colab.research.google.com/github/inferogenesis/cpomdp-workshop"
    "/blob/main/notebooks/build_a_curious_rocket.ipynb"
)
PYPI = "https://pypi.org/project/cpomdp/"
SITE = "https://inferogenesis.com/"
CPOMDP_DOCS = "https://cpomdp.inferogenesis.com/"
PRESENTER_2_LINKEDIN = "https://www.linkedin.com/in/kevin-backhouse-112554253"

RIGHT = SLIDE_WIDTH - MARGIN  # the right edge of the content
FULL_WIDTH = RIGHT - MARGIN
# Every route slide puts its figure here, so only the route changes between them.
ROUTE_FIGURE = (MARGIN, 1.4, 8.4)  # x, y, width
ROUTE_COLUMN = 9.3  # where the callouts beside a route frame start
TASK_CODE_BOTTOM = 6.0  # a task's code ends above the strip of instructions
TASK_STRIP = 6.2  # where that strip starts
UNDER_CODE = 0.45  # what a task's code leaves for the line under it
PHOTO_SIDE = 1.6  # a presenter's photo frame is square
CARD_TEXT = (PHOTO_SIDE + 0.25, 4.0)  # a presenter's name and bio: x from the card, w


@dataclass(frozen=True)
class Part:
    """One stretch of a block.

    Attributes:
        name: what the agenda slide writes on it.
        kind: ``"talk"`` when we present, ``"code"`` when the room works and
            ``"open"`` for the time left for questions.
        minutes: how long it runs.
    """

    name: str
    kind: str
    minutes: int


@dataclass(frozen=True)
class Block:
    """One stretch of the two hours, in parts.

    Attributes:
        label: what the agenda slide calls it.
        parts: what happens in it, in order.
        subject: what a task builds, or empty.
    """

    label: str
    parts: tuple[Part, ...]
    subject: str = ""

    @property
    def minutes(self) -> int:
        """How long the block runs."""
        return sum(part.minutes for part in self.parts)


TASK_MINUTES = 15
"""How long the room codes on each task, and the length of each timer's bar."""

TASK_PARTS = (
    Part("brief", "talk", 4),
    Part("you code", "code", TASK_MINUTES),
    Part("discussion", "talk", 8),
)
"""Every task's parts."""

AGENDA = (
    Block("Welcome", (Part("introductions and setup", "talk", 12),)),
    Block("Task 1", TASK_PARTS, "the 1960 rocket"),
    Block("Task 2", TASK_PARTS, "the beacon"),
    Block("Task 3", TASK_PARTS, "the blackout"),
    Block("Why it matters", (Part("what cpomdp adds, and what's next", "talk", 12),)),
    Block("Questions", (Part("questions, or leave early", "open", 15),)),
)
"""The two hours: an hour of talk, 45 minutes of code and 15 left open."""


def clock(label: str, part: int = 0) -> str:
    """When part ``part`` of the block called ``label`` starts, as ``h:mm``."""
    elapsed = 0
    for block in AGENDA:
        if block.label == label:
            elapsed += sum(earlier.minutes for earlier in block.parts[:part])
            return f"{elapsed // 60}:{elapsed % 60:02d}"
        elapsed += block.minutes
    raise ValueError(f"no block called {label!r} in the agenda")


GRAPH_CODE = """\
sensor = FixedSensor(jnp.eye(2), observation_noise=0.05 * jnp.eye(2))
graph = CouplingGraph(dims=[2], couplings=[], observation_models={0: sensor})
backend = CouplingGraphBackend(
    graph,
    [GaussianTransition(jnp.eye(2), dynamics_noise=1e-3 * jnp.eye(2))],
    prior=Belief(mean=LAUNCH, cov=4 * jnp.eye(2)),
    control_matrix=jnp.eye(2),
)
rocket = Agent.from_backend(backend, goal)"""
"""What ``one_node`` builds from task 1's model. The deck test runs it."""

PYMDP_MAP = (
    ("A", "the observation likelihood", ("observation_matrix,", "observation_noise")),
    ("B", "the transition", ("dynamics_matrix,", "control_matrix")),
    ("C", "the preference", ("Preference", "from Task 2")),
    ("D", "the prior", ("prior",)),
)
"""pymdp's letters and the cpomdp fields that hold the same thing."""

DISCLOSURE = (
    f"today's notebook pins {WORKSHOP_CPOMDP_VERSION}. Some of its src was written by "
    "Claude Code under the review described."
)
"""The disclosure slide's caption, the owner's words."""


@dataclass(frozen=True)
class Presenter:
    """One card on the who-we-are slide.

    Attributes:
        name: the card's heading.
        photo: the photo's file name in ``deck/photos/``, which only the presenting
            copy reads.
        bio: the presenter's own words.
    """

    name: str
    photo: str
    bio: str


PRESENTER_1_BIO = (
    "Independent research software engineer, UK. Builds cpomdp, a JAX-native "
    "continuous-state active inference toolbox and the continuous sibling to pymdp, "
    "under Inferogenesis. Background in numerical general relativity and event-driven "
    "orchestration of RL models at scale. Author of State-Dependent Observation Noise "
    "Reintroduces Epistemic Value in Linear-Gaussian Active Inference "
    "(arXiv:2607.20306). Co-leads the Certifiable Active Inference programme."
)
"""Presenter 1's bio, the owner's words."""

PRESENTER_2_BIO = (
    "Kevin Backhouse is a PhD researcher in Cognitive Neuroscience at Durham "
    "University. His research examines how reward and expectation shape awareness, "
    "attention and memory. He also collaborates on CPOMDP, developing and evaluating "
    "artificial agents within the active inference framework."
)
"""Presenter 2's bio, as the owner gave it. Kevin on a slide, Kev in the notes."""

PRESENTERS = (
    Presenter("Presenter 1 (keyboard)", "presenter_1.jpg", PRESENTER_1_BIO),
    Presenter("Kevin (concepts)", "presenter_2.jpg", PRESENTER_2_BIO),
)


def captured(function: Callable[..., object], *args: object) -> str:
    """What ``function(*args)`` prints, without the final newline."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        function(*args)
    return buffer.getvalue().rstrip("\n")


def run_cell(source: str, namespace: dict[str, object]) -> str:
    """What a notebook cell prints when run in ``namespace``."""
    return captured(exec, source, namespace)


def plain(markdown: str) -> str:
    """A lesson poll as plain text for the notes pane."""
    text = markdown.replace("<details><summary>Answer</summary>", "Answer.")
    text = text.replace("</details>", "")
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("`", "").replace("*", "")
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines: list[str] = []
        for line in block.splitlines():
            if lines and not re.match(r"\d+\. ", line.strip()):
                lines[-1] += f" {line.strip()}"
            else:
                lines.append(line.strip())
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


def notes(*paragraphs: str) -> str:
    """Speaker notes, one paragraph per argument."""
    return "\n\n".join(paragraphs)


def linked(url: str, *, size: float = 14) -> Run:
    """A URL as a run that opens it, in the code face, without its scheme."""
    return Run(url.removeprefix("https://"), font=CODE_FONT, size=size, link=url)


def paste_card(
    page: Page, x: float, y: float, w: float, h: float, paste: str, *, stacked: bool
) -> None:
    """The number the room pastes in chat when a task closes.

    Stacked puts the number under its label, for a narrow column.
    """
    value = Run(paste, font=CODE_FONT)
    if stacked:
        paragraphs = [[Run("Paste in chat", colour=MUTED)], [value]]
    else:
        paragraphs = [[Run("Paste in chat  ", colour=MUTED), value]]
    page.card(
        x, y, w, h, paragraphs,
        size=17, line=AMBER_MARK, anchor=MSO_ANCHOR.MIDDLE, space_after=2,
    )  # fmt: skip


def shown(attempt: str, through: str | None = None) -> str:
    """The part of an attempt cell a task slide shows.

    The imports are cut. With ``through``, so is everything after the first line that
    starts with it, so a long cell keeps a size the room can read.
    """
    lines = without_imports(attempt).splitlines()
    if through is not None:
        last = next(i for i, line in enumerate(lines) if line.startswith(through))
        lines = lines[: last + 1]
    return "\n".join(lines)


def task_page(
    deck: Deck, title: str, segment: str, source: str, paste: str, *, below: str
) -> Page:
    """A task's code, a line under it, and a strip of what to do.

    The lines to write are amber. The code takes the room the line under it leaves.
    """
    page = deck.page(title, segment)
    bottom = TASK_CODE_BOTTOM - UNDER_CODE
    size = code_size(source, FULL_WIDTH, bottom - CONTENT_TOP)
    height = code_height(source, size)
    page.code(MARGIN, CONTENT_TOP, FULL_WIDTH, height, source, size=size)
    top = CONTENT_TOP + height + 0.15
    page.text(MARGIN, top, FULL_WIDTH, TASK_STRIP - top - 0.05, [below], colour=MUTED)
    left, write, hints = MARGIN, 3.5, 3.7
    page.card(
        left, TASK_STRIP, write, 0.75,
        [[Run("Write the two amber lines", colour=AMBER, bold=True)]],
        size=17, anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip
    left += write + 0.3
    page.card(
        left, TASK_STRIP, hints, 0.75, ["Hints are below the cell"], size=17,
        anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip
    left += hints + 0.3
    paste_card(page, left, TASK_STRIP, RIGHT - left, 0.75, paste, stacked=False)
    return page


def timer_page(deck: Deck, number: int, steps: tuple[str, ...]) -> Page:
    """The slide that stays up while the room works: what to do, and the bar."""
    page = deck.page(f"Task {number} · Over to you", f"task {number}")
    page.text(
        MARGIN, CONTENT_TOP + 0.1, FULL_WIDTH, 2.5,
        [
            [Run(f"{index}   ", colour=AMBER, bold=True), Run(step)]
            for index, step in enumerate(steps, start=1)
        ],
        size=24, space_after=14,
    )  # fmt: skip
    page.countdown(
        MARGIN, 4.3, FULL_WIDTH, TASK_MINUTES,
        then=[
            [
                Run("Time's up. ", colour=AMBER),
                Run(f"Run checkpoint {number}, even if everything worked."),
            ]
        ],
    )  # fmt: skip
    page.text(
        MARGIN, 6.55, FULL_WIDTH, 0.45,
        [
            [
                Run("Stuck on an error?  "),
                Run("workshop.symptoms()", font=CODE_FONT),
                Run("  lists the five most common, with their fixes."),
            ]
        ],
        size=16, colour=MUTED,
    )  # fmt: skip
    return page


def timer_notes(number: int, *paragraphs: str) -> str:
    """A timer slide's notes: when it starts, how the bar runs, then ``paragraphs``."""
    return notes(
        f"Planned at {clock(f'Task {number}', 1)}. Keyboard. Leave this slide up "
        f"for the whole {TASK_MINUTES} minutes. The bar starts when the slide "
        "appears and needs no click. It starts again if you leave the slide and "
        "come back, so share the notebook from another window if you need to.",
        "Nobody pastes hints. Each person opens the collapsed hint and reference "
        "cells when they like. Whoever isn't at the keyboard watches the chat and the "
        "anonymous form. We don't debug anyone's code from the front. Point people "
        "at the hints, the reference cell and workshop.symptoms().",
        *paragraphs,
        f"When the bar fills, the line under it appears. Read it out. Checkpoint "
        f"{number} restores the reference setup the next part needs.",
    )


def route_page(deck: Deck, title: str, segment: str, frame: str) -> Page:
    """A route frame, in the one place every route slide puts it."""
    page = deck.page(title, segment)
    x, y, width = ROUTE_FIGURE
    page.picture(ASSETS / frame, x, y, width)
    return page


def route_strip(page: Page, *, output: str = "", caption: str = "") -> None:
    """Under a route frame: what the cell printed, or a caption."""
    x, _, width = ROUTE_FIGURE
    if output:
        page.printed(x, 5.98, width, 0.97, output, size=11, wrap=True)
    else:
        page.text(x, 6.05, width, 0.85, [caption], size=15, colour=MUTED)


def callout(page: Page, label: str, value: Run, gloss: str) -> None:
    """The number a route slide is about, large, beside the frame."""
    page.card(
        ROUTE_COLUMN, CONTENT_TOP, RIGHT - ROUTE_COLUMN, 2.95,
        [[Run(label, colour=MUTED)], [value], [Run(gloss)]],
        size=17, space_after=6, anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip


def task_close(page: Page, paste: str, poll: int) -> None:
    """The close of a task: the number to paste, then the poll."""
    width = RIGHT - ROUTE_COLUMN
    paste_card(page, ROUTE_COLUMN, 4.65, width, 1.05, paste, stacked=True)
    page.pill(ROUTE_COLUMN, 5.98, width, 0.55, f"Poll {poll} · 60 s", size=16)


def headed(heading: str, *lines: Paragraph, colour: str = INK) -> list[Paragraph]:
    """A card's paragraphs: a bold heading, then its lines."""
    return [[Run(heading, colour=colour, bold=True)], *lines]


def card_row(
    page: Page,
    top: float,
    height: float,
    cards: tuple[list[Paragraph], ...],
    *,
    size: float = 18,
) -> None:
    """Cards side by side across the content width, the same size."""
    gap = 0.3
    width = (FULL_WIDTH - gap * (len(cards) - 1)) / len(cards)
    for index, paragraphs in enumerate(cards):
        page.card(
            MARGIN + index * (width + gap), top, width, height, paragraphs,
            size=size, space_after=8,
        )  # fmt: skip


def question(page: Page, top: float, text: str) -> None:
    """A question left open for the room, under the slide's content."""
    page.text(
        MARGIN, top, FULL_WIDTH, 0.95, [[Run(text, colour=AMBER, bold=True)]], size=22
    )


# --- welcome ----------------------------------------------------------------------


def title(deck: Deck) -> None:
    """The title, the pin and the three tasks."""
    page = deck.page()
    page.mark(MARGIN, 0.55, 0.85)
    page.text(
        MARGIN + 0.6, 0.72, 5.0, 0.5, ["Inferogenesis"], size=22, bold=True,
        colour=MUTED,
    )  # fmt: skip
    page.text(MARGIN, 1.7, 11.5, 1.2, ["Build a curious rocket"], size=60, bold=True)
    page.text(
        MARGIN, 3.05, 11.5, 0.6,
        ["cpomdp · continuous active inference in Python, hands-on"],
        size=26, colour=MUTED,
    )  # fmt: skip
    page.text(
        MARGIN, 3.7, 11.5, 0.5, ["6th Applied Active Inference Symposium · 2026"],
        size=20, colour=MUTED,
    )  # fmt: skip
    page.pill(
        MARGIN, 4.5, 2.6, 0.55, f"cpomdp {WORKSHOP_CPOMDP_VERSION}", size=18,
        font=CODE_FONT,
    )  # fmt: skip
    for index, (label, colour) in enumerate(
        (
            ("1  the 1960 rocket", INK),
            ("2  the beacon", AMBER),
            ("3  the blackout", BLUE),
        )
    ):
        page.pill(MARGIN + index * 4.1, 5.9, 3.8, 0.7, label, colour=colour, size=20)
    page.notes(
        notes(
            f"Planned at {clock('Welcome')}. Keyboard shares the screen with this "
            "slide up and pastes the Colab link in the chat. Kev opens.",
            "Welcome, everyone. Over the next two hours, we'll build three rockets and "
            "watch how each one decides where to fly. Do open the notebook now, even "
            "while we're talking. The next few slides say how, and the installs can "
            "run while we introduce ourselves.",
        )
    )


def presenter_card(page: Page, left: float, presenter: Presenter) -> None:
    """A presenter's photo frame, then their name over their bio."""
    page.photo_frame(left, CONTENT_TOP, PHOTO_SIDE, presenter.photo)
    offset, width = CARD_TEXT
    page.text(
        left + offset, CONTENT_TOP, width, 0.5, [presenter.name], size=22, bold=True
    )
    page.text(left + offset, CONTENT_TOP + 0.6, width, 2.4, [presenter.bio], size=14)


def who_we_are(deck: Deck) -> None:
    """The two presenters, and the library the room uses."""
    page = deck.page("Who we are", "welcome")
    for index, presenter in enumerate(PRESENTERS):
        presenter_card(page, MARGIN + index * 6.2, presenter)
    page.text(
        MARGIN, 4.7, FULL_WIDTH, 2.2,
        [
            [Run("cpomdp", bold=True, size=24, font=CODE_FONT)],
            "The open-source library we'll use today, for active inference with "
            "continuous states.",
            "It's written in Python on JAX, and it's the continuous-state sibling of "
            "pymdp.",
            [linked(CPOMDP_DOCS, size=16)],
        ],
        size=20, space_after=8,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev introduces us both while the installs run.",
            "cpomdp is the library we'll use today. It does active inference with "
            "continuous states, which makes it the continuous-state sibling of pymdp. "
            "If you've used pymdp, its agent runs the same loop of perceiving and then "
            "acting. We'll come back to what it adds at the end.",
        )
    )


def disclosure(deck: Deck) -> None:
    """The AI-use panel, the owner's to write, and its caption."""
    page = deck.page("Disclosure", "welcome")
    page.placeholder(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 3.9,
        "[owner: the AI-use panel and the review it describes]", size=20,
    )  # fmt: skip
    page.text(MARGIN, 5.75, FULL_WIDTH, 0.9, [DISCLOSURE], size=20, colour=MUTED)
    page.notes(
        notes(
            "TODO owner: the panel.",
            "Q&A. Asked whether Claude wrote the release the room is running, the "
            "answer is: some of it.",
        )
    )


def agenda(deck: Deck) -> None:
    """The two hours, one row per block, each part named and as wide as its minutes."""
    page = deck.page("Two hours, three tasks", "welcome")
    bars = MARGIN + 3.4  # where every row's parts start
    inch = (RIGHT - bars) / max(block.minutes for block in AGENDA)  # per minute
    pitch, height, gap = 0.78, 0.6, 0.04
    for index, block in enumerate(AGENDA):
        top = CONTENT_TOP + 0.15 + index * pitch
        heading = [Run(block.label, bold=True)]
        if block.subject:
            heading.append(Run(f"  {block.subject}", colour=MUTED, size=16))
        page.text(
            MARGIN, top, bars - MARGIN - 0.2, height, [heading], size=20,
            anchor=MSO_ANCHOR.MIDDLE, space_after=0,
        )  # fmt: skip
        left = bars
        for part in block.parts:
            page.block(
                left + gap / 2, top, part.minutes * inch - gap, height, part.kind,
                label=part.name, size=16,
            )  # fmt: skip
            left += part.minutes * inch
    page.text(
        MARGIN, CONTENT_TOP + 0.15 + len(AGENDA) * pitch + 0.15, FULL_WIDTH, 0.5,
        [f"While you code, a {TASK_MINUTES}-minute timer stays on the screen."],
        size=19, colour=MUTED, space_after=0,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. We present for an hour in all, and you code for 45 minutes, "
            f"{TASK_MINUTES} on each task. The last {AGENDA[-1].minutes} minutes are "
            "left open for questions and discussion, and you're welcome to leave "
            "early.",
            f"Planned: Task 1 at {clock('Task 1')}, Task 2 at {clock('Task 2')}, "
            f"Task 3 at {clock('Task 3')}, why it matters at "
            f"{clock('Why it matters')}, questions from {clock('Questions')}.",
            "There's no break in the plan. If the room needs one, take it from the "
            "open time at the end.",
        )
    )


def how_we_work(deck: Deck) -> None:
    """The ways of working: two lines, hints, checkpoints, one number."""
    page = deck.page("How we will work", "welcome")
    width, height, gap = (FULL_WIDTH - 0.3) / 2, 1.95, 0.3
    cards = (
        headed(
            "Two lines to complete",
            [
                Run("In each task, you will fill in two lines marked "),
                Run("...", font=CODE_FONT, colour=AMBER, bold=True),
                Run(
                    ". We have written the rest so you can focus on the ideas. Some "
                    "bonus exercises give you a chance to write more of your own code."
                ),
            ],
            colour=AMBER,
        ),
        headed(
            "Help whenever you need it",
            "You will find two hints and a worked solution below each task’s code "
            "cell. Click to open them whenever you like. They are there to help, "
            "so use them freely.",
        ),
        headed(
            "Run the checkpoint",
            "At the end of each task, run the checkpoint cell, even if your code "
            "worked. This gives everyone the same starting point for the next task.",
        ),
        headed(
            "Share your result",
            "Paste the number requested at the end of the task into the chat. "
            "We will then take a one-minute poll.",
        ),
    )
    for index, paragraphs in enumerate(cards):
        row, column = divmod(index, 2)
        page.card(
            MARGIN + column * (width + gap), CONTENT_TOP + row * (height + gap), width,
            height, paragraphs, size=18, space_after=8,
        )  # fmt: skip
    page.text(
        MARGIN, CONTENT_TOP + 2 * (height + gap) + 0.1, FULL_WIDTH, 0.9,
        [
            "The focus is on understanding the ideas behind the code. Questions "
            "are welcome throughout, in the chat or through the anonymous form."
        ],
        size=18, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Keyboard. Each task has two lines to write, and we have written the rest "
            "of the code, so the Python stays out of the way. Some of the bonus "
            "exercises ask you to write your own.",
            "Each task has two hints and a reference solution, collapsed below the "
            "cell. A hint shows its nudge first, and the line itself is behind Show "
            "code. Open them whenever you like, and using them is not failing.",
            "At the end of every task, run the checkpoint cell, even if everything "
            "worked. It puts the reference names back, so the next task starts from "
            "the same place for everyone.",
            "Each task closes with one number in the chat. It lets us see how the "
            "room is doing without anyone writing a sentence. Then there is a "
            "one-minute poll.",
        )
    )


def setup(deck: Deck) -> None:
    """Google Colab, or the room's own Python."""
    page = deck.page("Get set up", "welcome")
    column = (FULL_WIDTH - 0.6) / 2
    version = metadata.version("cpomdp-workshop")
    page.text(
        MARGIN, CONTENT_TOP, column, 2.6,
        [
            [Run("Google Colab", bold=True, size=26, colour=AMBER)],
            [Run("Recommended", colour=MUTED)],
            "It runs in your browser, so there's nothing to install.",
            "Open the notebook link below, then run cell 1. It installs the "
            "workshop and checks it's ready.",
        ],
        size=19, space_after=8,
    )  # fmt: skip
    right = MARGIN + column + 0.6
    page.text(
        right, CONTENT_TOP, column, 1.85,
        [
            [Run("Your own Python", bold=True, size=26, colour=BLUE)],
            [Run("VS Code or Jupyter", colour=MUTED)],
            "Python 3.11 or newer, then install the workshop:",
        ],
        size=19, space_after=8,
    )  # fmt: skip
    page.code(
        right, CONTENT_TOP + 1.95, column, 0.55,
        f"pip install cpomdp-workshop=={version}", size=15,
    )  # fmt: skip
    page.text(
        right, CONTENT_TOP + 2.65, column, 0.9,
        ["Open the notebook from the repository below and run cell 1."],
        size=19,
    )  # fmt: skip
    top = 5.1
    for label, url in (("notebook in Colab", COLAB), ("repository", REPOSITORY)):
        page.text(MARGIN, top, 4.0, 0.35, [label], size=15, colour=MUTED)
        page.link(MARGIN, top + 0.36, FULL_WIDTH, 0.35, url, size=12)
        top += 0.85
    page.notes(
        notes(
            "Keyboard. We recommend Colab. It runs the notebook in your browser, so "
            "nothing needs installing. Open the link from the chat, then run cell 1, "
            "which installs the workshop from PyPI.",
            "If you'd rather work locally, that's fine too. You'll need Python 3.11 "
            "or newer and the pip line, or uv as the README shows. Then open "
            "notebooks/build_a_curious_rocket.ipynb from the repository.",
            f"Either way, cell 1 stops if cpomdp isn't {WORKSHOP_CPOMDP_VERSION} or "
            "Python is older than 3.11, and it prints what it found.",
        )
    )


def preview(deck: Deck) -> None:
    """Cell 2's picture, the whole game first."""
    page = route_page(deck, "What we will build", "welcome", "route_4_rock.png")
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, RIGHT - ROUTE_COLUMN, 4.7,
        [
            "By the end of Task 3, we will have built the models that produce these "
            "three rocket paths.",
            "The dashed line shows the 1960 rocket. The coloured lines show two "
            "versions of the curious rocket.",
            [
                Run("Run the second code cell to see the preview: "),
                Run("workshop.preview()", font=CODE_FONT),
            ],
        ],
        size=19, space_after=14,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 6.3, RIGHT - ROUTE_COLUMN, 0.5, ["Seed 3."], size=15, colour=MUTED
    )
    page.notes(
        notes(
            "Kev. Introduce the preview before beginning the tasks. Run the second "
            "code cell, then ask: What makes these rockets take different routes?",
            "The dashed line is the 1960 rocket. It hits the rock on step 6. The two "
            "solid lines are curious rockets, and on seed 3 both fly round the rock.",
            "Do not worry if you cannot explain it yet. Neither rocket knows the "
            "rock is there, and working out what is going on is what the three "
            "tasks are for.",
        )
    )


# --- task 1 -----------------------------------------------------------------------


def task_1_brief(deck: Deck) -> None:
    """What the 1960 rocket is, and pymdp's letters in cpomdp's names."""
    page = deck.page("Task 1 · The 1960 rocket", "task 1")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 1.65,
        [
            [
                Run("A "),
                Run("Kalman filter", bold=True),
                Run(" estimates the rocket’s position from noisy sensor readings."),
            ],
            [
                Run("A "),
                Run("linear quadratic regulator (LQR)", bold=True),
                Run(" steers the rocket towards the landing pad."),
            ],
            "The sensor has the same level of noise wherever the rocket flies.",
        ],
        size=21, space_after=12,
    )  # fmt: skip
    page.text(
        MARGIN, 3.2, FULL_WIDTH, 0.55,
        [[Run("Are all possible moves equally informative?", colour=AMBER, bold=True)]],
        size=22,
    )  # fmt: skip
    page.text(
        MARGIN, 3.9, FULL_WIDTH, 0.3,
        ["pymdp notation and the corresponding cpomdp fields"],
        size=15, colour=MUTED,
    )  # fmt: skip
    width = (FULL_WIDTH - 0.9) / 4
    for index, (letter, meaning, names) in enumerate(PYMDP_MAP):
        page.card(
            MARGIN + index * (width + 0.3), 4.3, width, 2.3,
            [
                [Run(letter, colour=AMBER, bold=True, size=32)],
                [Run(meaning, colour=MUTED)],
                *([Run(name, font=CODE_FONT)] for name in names),
            ],
            size=15, space_after=2,
        )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 1')}. Kev.",
            "Our first rocket is the classic one. A Kalman filter estimates its "
            "position from noisy readings, and a linear quadratic regulator steers it "
            "to the pad. Its sensor has the same level of noise everywhere. "
            "Are all possible moves equally informative? Hold that question.",
            "If you know pymdp, the cards map its letters onto cpomdp's fields. A is "
            "the observation likelihood. In cpomdp that is the observation_matrix "
            "together with the observation_noise, R, because the noise is part of "
            "what a reading says about the state. B is the transition, so "
            "dynamics_matrix, with control_matrix for where a move enters. C is the "
            "preference, which we meet in Task 2 as Preference. D is the prior.",
            "The 1960 rocket's StateGoal has no pymdp letter. It is a goal over states "
            "with a thruster cost, which is what makes it an LQR controller.",
        )
    )


def task_1_code(deck: Deck) -> None:
    """Task 1's code."""
    source = shown(lesson.SESSION_1.attempt)
    page = task_page(
        deck, "Task 1 · Two lines to write", "task 1", source, "spread",
        below="Then the bonuses. Can you make the spread move?",
    )  # fmt: skip
    page.notes(
        notes(
            "Keyboard. Two lines to write: what the sensor observes and how noisy "
            "its readings are. The rest is written, the thruster cost included. The "
            "imports sit at the top of the cell.",
            "Run it, and check_agent prints the rocket and where it lands. probe_model "
            "then evaluates how much information the rocket could expect to gain "
            "from each of the nine moves. The spread is the difference between the "
            "most and the least. That's the number for the chat.",
        )
    )


def task_1_timer(deck: Deck) -> None:
    """Task 1's fifteen minutes."""
    page = timer_page(
        deck, 1,
        (
            "Write the two amber lines and run the cell.",
            "Paste your spread in the chat.",
            "Finished early? Try the two bonuses.",
        ),
    )  # fmt: skip
    page.notes(timer_notes(1))


def task_1_result(deck: Deck) -> None:
    """The reference route and spread, with prompts for the task 1 discussion."""
    page = route_page(
        deck,
        "Task 1 · Discussion",
        "task 1",
        "route_1_straight.png",
    )
    width = RIGHT - ROUTE_COLUMN
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, width, 0.35, ["Reference result"],
        size=15, colour=MUTED,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 1.88, width, 0.5, ["spread = 0.000000 nats"],
        size=18, font=CODE_FONT, bold=True,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 2.5, width, 1.15,
        ["The difference between the highest and lowest expected information "
         "gain across the possible moves."],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 3.9, width, 0.75,
        ["What does a spread of zero tell us?"], size=20, bold=True,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 4.85, width, 1.4,
        ["If you tried the bonus exercises, what changed and what stayed the same?"],
        size=19, bold=True,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 6.65, width, 0.35, ["Poll 1 · 60 seconds"],
        size=16, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 1', 2)}. Kev invites participants to compare "
            "their results with the reference route and spread. The rocket is LQR, "
            "and lands 0.13 from the pad.",
            "Ask what a spread of zero tells us. Then ask those who tried the bonus "
            "exercises what changed and what stayed the same. Give participants "
            "time to discuss before explaining the result.",
            "The probe compares the information each of the nine compass moves is "
            "expected to give. spread = 0.000000 nats, so no move gives more than any "
            "other. Zero spread does not mean zero information gain. The bonus "
            "changes can alter the amount of information gained, while the spread "
            "remains zero.",
            "Then poll 1 for 60 seconds. Kev reads the split. Use the following "
            "slide to introduce how position-dependent sensor noise can make "
            "expected information gain differ between moves.",
            "After the poll, connect the result to what we call the Koudahl "
            "collapse. Koudahl, Kouw and de Vries (2021) show that, for the linear "
            "Gaussian models and full expected free energy (EFE) objective they "
            "analyse, EFE reduces to the KL-control objective plus a constant "
            "determined by observation noise. Minimising EFE therefore supplies "
            "no purposeful exploratory (epistemic) drive in that setting.",
            "Here, with fixed sensor noise and additive controls, every candidate "
            "move has the same expected information gain. The epistemic "
            "contribution cannot favour one move over another, which is why spread "
            "is zero. Zero spread does not mean zero information gain.",
            "Reference: Koudahl, M. T., Kouw, W. M. and de Vries, B. (2021). "
            "On Epistemics in Expected Free Energy for Linear Gaussian State "
            "Space Models. Entropy. https://doi.org/10.3390/e23121565",
            plain(lesson.SESSION_1.poll),
        )
    )


def task_1_meaning(deck: Deck) -> None:
    """How position-dependent sensor noise connects movement to learning."""
    page = deck.page("Making information gain depend on the action", "task 1")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.7,
        ["In cpomdp, sensor noise can depend on the rocket’s position: R(x)."],
        size=22,
    )  # fmt: skip
    for index, label in enumerate(
        (
            "Where the rocket moves",
            "How reliable its next reading is",
            "How much it expects to learn",
        )
    ):
        top = 2.5 + 1.2 * index
        page.text(
            MARGIN, top, FULL_WIDTH, 0.5, [label],
            size=24, bold=True, align=PP_ALIGN.CENTER,
        )  # fmt: skip
        if index < 2:
            page.text(
                MARGIN, top + 0.6, FULL_WIDTH, 0.4, ["↓"],
                size=24, colour=MUTED, align=PP_ALIGN.CENTER,
            )  # fmt: skip
    page.text(
        MARGIN, 6.35, FULL_WIDTH, 0.55,
        ["Some moves can now offer more information than others."],
        size=22, colour=AMBER, bold=True,
    )  # fmt: skip
    page.notes(
        notes(
            "R describes the noise in the sensor readings. Writing R(x) means that "
            "this noise depends on the rocket’s position.",
            "Near the beacon, readings are clearer. By moving towards it, the "
            "rocket can obtain a more informative measurement and reduce its "
            "uncertainty about where it is. Movement can therefore affect both "
            "progress towards the goal and what the rocket learns.",
            "This expected reduction in uncertainty is called epistemic value. "
            "With our sensor model, some moves can offer more information than "
            "others.",
            "State-dependent sensory precision has a history in active inference. "
            "Dan’s contribution establishes conditions under which it makes "
            "epistemic value depend on action and rules out an equivalent fixed "
            "linear-Gaussian filter.",
            "The mean dynamics and observation map remain linear, and the "
            "conditional noise distributions remain Gaussian. However, the joint "
            "model is no longer fully Gaussian.",
            "Tasks 2 and 3 explore this mechanism. Whether it changes the chosen "
            "route also depends on the objective and planning horizon.",
            "R(x) is one route to action-dependent information gain. Another "
            "modelling choice is multiplicative control, where an action selects "
            "the dynamics governing how the state evolves. Koudahl, Kouw and de "
            "Vries show that this can produce exploration when the epistemic term "
            "is optimised on its own. Their full EFE objective still reduces to "
            "KL control plus a constant, so these results concern different "
            "objective choices.",
            "A biological illustration of actions changing dynamics is muscle "
            "co-contraction. Activating opposing muscles together can increase "
            "joint stiffness with little change in net torque. The limb then "
            "responds differently to the same disturbance. In a simplified model, "
            "muscle activation changes the stiffness multiplying the displacement, "
            "which can be represented using multiplicative control. This illustrates "
            "a change in dynamics; it does not by itself demonstrate "
            "information-seeking behaviour. We use R(x) in the workshop because "
            "the beacon and blackout change the reliability of observations "
            "across locations.",
            "Reference: Corva, D. (2026). State-Dependent Observation Noise "
            "Reintroduces Epistemic Value in Linear-Gaussian Active Inference. "
            "Sections 2.3 and 3, Theorem 1. https://arxiv.org/abs/2607.20306",
            "Further references: Koudahl, M. T., Kouw, W. M. and de Vries, B. "
            "(2021). On Epistemics in Expected Free Energy for Linear Gaussian "
            "State Space Models. https://doi.org/10.3390/e23121565",
            "Franklin, D. W. et al. (2007). Endpoint Stiffness of the Arm Is "
            "Directionally Tuned to Instability in the Environment. "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6672883/",
        )
    )


# --- task 2 -----------------------------------------------------------------------


def task_2_brief(deck: Deck) -> None:
    """What changes when the sensor's noise depends on where the rocket is."""
    page = deck.page("What changes when observations depend on location?", "task 2")
    card_row(
        page, CONTENT_TOP, 2.4,
        (
            headed(
                "Task 1: fixed sensor noise",
                "Sensor reliability is the same wherever the rocket travels.",
            ),
            headed(
                "Task 2: location-dependent noise",
                "Observations are more reliable near the beacon and noisier further "
                "away.",
                colour=AMBER,
            ),
            headed(
                "A change in the objective",
                "The rocket now has a preference over observations. It wants a "
                "reading that says it's on the landing pad.",
            ),
        ),
    )  # fmt: skip
    question(
        page, 4.2,
        "Will the rocket change its route when some places give more reliable "
        "observations?",
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 2')}. Kev. In Task 1, the sensor was equally "
            "reliable everywhere. Now we'll introduce a navigation beacon. Readings "
            "are more reliable near it and noisier further away. k, beacon_gain in "
            "the code, sets how quickly the noise grows with distance, up to a cap.",
            "The goal changes too. This rocket prefers a reading, the one that says "
            "it's on the pad, where the 1960 rocket preferred a position. At each "
            "step its planner scores the nine compass moves by expected free energy.",
            "Leave the question with the room. Don't answer it yet.",
        )
    )


def task_2_code(deck: Deck) -> None:
    """Task 2's code."""
    source = shown(lesson.SESSION_2.attempt, through="check_my_sensor(")
    page = task_page(
        deck, "Task 2 · Two lines to write", "task 2", source, "your k",
        below="Below this, the cell builds a curious rocket on your sensor, flies it "
        "and plots its route.",
    )  # fmt: skip
    page.notes(
        notes(
            "Keyboard. Two lines again: k itself, and the beacon's extra noise inside "
            "my_sensor. params holds the beacon's settings, and the comments say how "
            "to read each one by name.",
            "Start at k = 0.08, then raise it. Can you find a k that changes the "
            "route and still lands?",
        )
    )


def task_2_timer(deck: Deck) -> None:
    """Task 2's fifteen minutes."""
    page = timer_page(
        deck, 2,
        (
            "Write the two amber lines. Start with k = 0.08.",
            "Raise k until the route changes. Paste your k in the chat.",
            "Finished early? Try the two bonuses.",
        ),
    )  # fmt: skip
    page.notes(timer_notes(2, "Hint 1 gives k = 0.4."))


def task_2_result(deck: Deck) -> None:
    """The second route frame. The beacon bends the route."""
    page = route_page(
        deck, "The beacon bends the route", "task 2", "route_2_beacon.png"
    )
    route_strip(
        page, output=captured(check_my_sensor, beacon_sensor, default_params(0.4))
    )
    callout(
        page, "k = 0.40", Run("1.41", colour=AMBER, bold=True, size=60),
        "off the straight line. It lands 0.24 from the pad.",
    )  # fmt: skip
    task_close(page, "your k", 2)
    page.notes(
        notes(
            f"Planned at {clock('Task 2', 2)}. Keyboard shows check_my_sensor's line "
            "for k = 0.4. The rocket bends 1.41 off the straight line and lands 0.24 "
            "from the pad.",
            "It's the same map as the last route slide. Only the field and the amber "
            "line are new.",
            "Ask for k in the chat, then poll 2 for 60 s. Kev reads the split.",
            plain(lesson.SESSION_2.poll),
        )
    )


def task_2_meaning(deck: Deck) -> None:
    """What changes the route: the reading's noise inside the expected miss."""
    page = deck.page("What steers the route", "task 2")
    page.card(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 1.8,
        [
            [Run("the expected miss from “on the pad”", colour=MUTED, font="Calibri")],
            [Run("½ (o⁺ − o*)ᵀ Λ (o⁺ − o*)  +  ½ tr(Λ S)", size=24)],
            [
                Run("S = C Σ⁺ Cᵀ + "),
                Run("R(μ⁺)", colour=AMBER, bold=True),
                Run("    the sensor's own noise, where the move is expected to land",
                    colour=MUTED, font="Calibri", size=17),
            ],
        ],
        size=20, font=MATH_FONT, space_after=6, anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip
    page.text(
        MARGIN, 3.55, FULL_WIDTH, 2.4,
        [
            "At each step the rocket scores the nine moves. The first part of each "
            "score is how far its reading is expected to land from “on the pad”.",
            "That expectation includes the sensor's own noise. Where the noise is "
            "large, even a reading taken on the pad lands far from it on average, so "
            "moves towards the beacon score better.",
            "It takes a sensor whose noise depends on position. The 1960 rocket's "
            "never does.",
        ],
        size=19, space_after=12,
    )  # fmt: skip
    page.text(
        MARGIN, 6.1, FULL_WIDTH, 0.75,
        [
            "At k = 0.6 the rocket stops short of the pad, however many moves ahead "
            "it looks."
        ],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. The answer to poll 2.",
            "Each step, the rocket scores the nine moves by expected free energy. Its "
            "first term is the expected miss from the preferred reading, on the pad. "
            "That expectation carries S, and S carries R where the move is expected "
            "to land. In the book's terms, this noise sets the ambiguity in expected "
            "free energy (Active Inference, eq. 2.6), and now it depends on where the "
            "rocket is.",
            "That term changes between moves only when R depends on position. For the "
            "1960 rocket it's the same for every move, and LQR is what's left.",
            "At k = 0.6 the rocket stops 1.49 short of the pad looking one move "
            "ahead, 1.65 looking two and 1.86 looking three. Looking further doesn't "
            "help, because the pad itself reads badly at that k.",
            "Bonus 2 varied the preference's precision. A stronger preference for a "
            "reading isn't guaranteed to give a better landing, because a higher "
            "precision scales the whole expected miss, the reading's own noise "
            "included. It can't make a noisy pad read any better.",
        )
    )


# --- task 3 -----------------------------------------------------------------------


def task_3_brief(deck: Deck) -> None:
    """The blackout, and the rock none of the rockets models."""
    page = deck.page("What happens when observations become unreliable?", "task 3")
    card_row(
        page, CONTENT_TOP, 2.4,
        (
            headed(
                "The beacon",
                "Observations become more reliable as the rocket approaches the "
                "beacon.",
            ),
            headed(
                "The blackout",
                "Observations become increasingly noisy towards the centre of the "
                "blackout.",
                colour=BLUE,
            ),
            headed(
                "A rock",
                "There's a rock inside the blackout, but none of the rockets has any "
                "representation of it in its model.",
            ),
        ),
    )  # fmt: skip
    question(
        page, 4.2,
        "How might the rockets respond, and what can their routes tell us about "
        "their behaviour?",
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 3')}. Kev. In Task 2, the beacon made "
            "observations more reliable nearby. Now we'll add a blackout on the "
            "straight line to the pad, where the noise rises sharply towards its "
            "centre.",
            "There's also a rock in the blackout, and none of the rockets knows it's "
            "there. In the book's terms, the rock is in the generative process, the "
            "world, but not in any rocket's generative model (Active Inference, "
            "Figure 2.2).",
            "We'll fly three rockets past it: the 1960 rocket, yours with the k on "
            "the slider, and one at k = 0.20.",
        )
    )


def task_3_code(deck: Deck) -> None:
    """Task 3's code."""
    source = shown(lesson.SESSION_3.attempt)
    page = task_page(
        deck, "Task 3 · Two lines to write", "task 3", source, "max |difference|",
        below="There's a rock in the blackout. None of the rockets knows.",
    )  # fmt: skip
    page.notes(
        notes(
            "Keyboard. Two lines again: the blackout's extra noise in my_world, and "
            "the 1960 rocket's model as a graph of one node. The slider sets k for "
            "your rocket.",
            "plot_world flies three rockets past the rock. check_backends flies the "
            "1960 rocket on the Kalman filter and on the graph, and prints the "
            "largest difference between them. That's the number for the chat.",
        )
    )


def task_3_timer(deck: Deck) -> None:
    """Task 3's fifteen minutes."""
    page = timer_page(
        deck, 3,
        (
            "Write the two amber lines and run the cell.",
            "Move the k slider and watch the routes. Paste max |difference| in the "
            "chat.",
            "Finished early? Try the two bonuses.",
        ),
    )  # fmt: skip
    page.notes(timer_notes(3))


def task_3_blackout(deck: Deck) -> None:
    """The third route frame, the blackout before the rock."""
    page = route_page(
        deck, "The blackout, before the rock", "task 3", "route_3_blackout.png"
    )
    route_strip(page, caption="Seed 3. No rock yet.")
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, RIGHT - ROUTE_COLUMN, 5.0,
        [
            "Near the blackout's centre the sensor noise is very large.",
            "The 1960 rocket flies straight through. Its model has one noise level.",
            "The curious rockets swing wide.",
        ],
        size=20, space_after=18,
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 3', 2)}. Kev. It's the same map again, and the "
            "field now has the blackout.",
            "The 1960 rocket flies the line it flew in Task 1. Its model has one noise "
            "level, and it flies in the world that model describes. The curious "
            "rockets swing wide of the noisy patch.",
            "There's no rock yet. The next slide adds it and nothing else.",
        )
    )


def task_3_rock(deck: Deck) -> None:
    """The last route frame. The rock, and the 1960 rocket's impact."""
    page = route_page(
        deck, "The 1960 rocket hits the rock", "task 3", "route_4_rock.png"
    )
    route_strip(
        page,
        caption="Seed 3. Both curious rockets clear the rock. The 1960 rocket hits it "
        "on every seed from 0 to 9.",
    )
    callout(
        page, "1960 rocket", Run("step 6", bold=True, size=54),
        "None of the rockets models the rock.",
    )  # fmt: skip
    task_close(page, "max |difference|", 3)
    page.notes(
        notes(
            "Kev. One thing added, the rock in the blackout. None of the rockets "
            "models it.",
            "The 1960 rocket hits it on step 6, on every seed from 0 to 9. The figure "
            "is seed 3, and both curious rockets clear it there. k = 0.08 peaks at "
            "y = 1.71 and ends at (4.83, 0.09). k = 0.20 peaks at 1.91 and ends at "
            "(5.18, 0.15).",
            "Ask for max |difference| in the chat. Below 1e-9 means the two engines "
            "agree. Then poll 3 for 60 s.",
            plain(lesson.SESSION_3.poll),
        )
    )


def task_3_meaning(deck: Deck) -> None:
    """Bonus 1's result: a rock never changes a route, only ends it."""
    page = deck.page("A rock never changes a route", "task 3")
    hits = (
        (
            "Rock in the blackout",
            (
                ("1960 rocket", "10 of 10"),
                ("k = 0.08", "2 of 10"),
                ("k = 0.20", "3 of 10"),
            ),
        ),
        (
            "Rock in clear sky, at (4.2, −0.7)",
            (
                ("1960 rocket", "0 of 10"),
                ("k = 0.08", "6 of 10"),
                ("k = 0.20", "5 of 10"),
            ),
        ),
    )  # fmt: skip
    card_row(
        page, CONTENT_TOP, 2.55,
        tuple(
            headed(
                where,
                [Run("seeds where it hits the rock", colour=MUTED, size=16)],
                *(
                    [Run(f"{name}    "), Run(count, font=CODE_FONT)]
                    for name, count in rows
                ),
            )
            for where, rows in hits
        ),
        size=19,
    )  # fmt: skip
    page.text(
        MARGIN, 4.35, FULL_WIDTH, 2.4,
        [
            "With or without the rock, a rocket flies exactly the same path until "
            "the moment it collides.",
            "The curious rockets steer away from noisy readings, and the rock sits "
            "where the readings are noisiest.",
            "Missing the rock isn't evidence that a rocket represented it or meant "
            "to avoid it.",
        ],
        size=19, space_after=12,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. The answer to poll 3, and bonus 1's result.",
            "The rockets aren't avoiding the rock at all. A rock never changes a "
            "route. Inside the blackout, over seeds 0 to 9, the k = 0.08 rocket hits "
            "it on 2 seeds and the k = 0.20 rocket on 3, while the 1960 rocket hits "
            "it every time. Moved into clear sky on their routes, it's hit on 6 and 5 "
            "of the 10 seeds by the curious rockets, and never by the 1960 rocket.",
            "So here the blackout makes the curious rockets hit the rock less often, "
            "although they know nothing about it. Seed 3 is one seed, and the picture "
            "earlier shows its less common route.",
            "Bonus 2 tuned two of the rocket's settings to land more and hit less. "
            "The best setting led the default by one flight, and the seeds mattered "
            "more than the settings. Tuning against one rock in one place can carry "
            "our own knowledge of the rock into the rocket.",
        )
    )


def backends(deck: Deck) -> None:
    """The 1960 rocket on two engines."""
    page = deck.page("One rocket, two engines", "task 3")
    size = code_size(GRAPH_CODE, 8.9, 3.45)
    height = code_height(GRAPH_CODE, size)
    page.code(MARGIN, CONTENT_TOP, 8.9, height, GRAPH_CODE, size=size)
    page.text(
        9.8, CONTENT_TOP, RIGHT - 9.8, height,
        [
            [Run("KalmanBackend", font=CODE_FONT)],
            [Run("the Kalman filter", colour=MUTED)],
            [Run("CouplingGraphBackend", font=CODE_FONT)],
            [Run("the same model as a graph of one node", colour=MUTED)],
        ],
        size=16, space_after=6,
    )  # fmt: skip
    model = model_with()
    beliefs = captured(check_backends, KalmanBackend(model), one_node(model))
    top = CONTENT_TOP + height + 0.3
    page.printed(
        MARGIN, top, FULL_WIDTH, 0.75, "\n".join(beliefs.splitlines()[:2]), size=12
    )
    page.text(
        MARGIN, top + 1.0, FULL_WIDTH, 0.75,
        [
            "The two agree below 1e-9. The graph engine also runs models one Kalman "
            "filter can't hold."
        ],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Keyboard. The 1960 rocket on two engines. The code is what one_node "
            "builds. Agent.from_backend puts a rocket on either backend.",
            "The two print the same belief to six places, and max |difference| sits "
            "below 1e-9. A graph of one node is a Kalman filter, so this is the check "
            "that the graph engine is right before we trust it with shapes a Kalman "
            "filter isn't. The self-guided cells show one.",
        )
    )


# --- why it matters ---------------------------------------------------------------


def coda(deck: Deck) -> None:
    """The certificates, as the coda cell prints them."""
    page = deck.page("What a printed number entitles us to say", "why it matters")
    height = code_height(lesson.CODA, 16)
    page.code(MARGIN, CONTENT_TOP, FULL_WIDTH, height, lesson.CODA, size=16)
    output = run_cell(lesson.CODA, after_session_3())
    top = CONTENT_TOP + height + 0.25
    page.printed(MARGIN, top, FULL_WIDTH, 1.1, output, size=15, wrap=True)
    page.text(
        MARGIN, top + 1.4, FULL_WIDTH, 6.9 - top - 1.4,
        [
            "PROVED means the search scored every plan in a declared, versioned set. "
            "The certificate counts them.",
            "It says what the computation did. It doesn't say the model is right.",
        ],
        size=20, space_after=14,
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Why it matters')}. Kev. Run the coda cell. What "
            "does a printed number entitle us to say?",
            "The certificate names the set by its version, compass-9, and counts "
            "every plan the search scored, 9 a step. The axis set is five moves, and "
            "seven moves deep it's 78125 plans, every one scored.",
            "That supports a specific claim: this computation was exhaustive over "
            "this set. The same goes for the backend agreement in Task 3. Neither is "
            "a general guarantee that the model is correct or that its behaviour "
            "generalises.",
        )
    )


def beginning(deck: Deck) -> None:
    """What the session showed, and what it hasn't established."""
    page = deck.page("A working model is the beginning, not the end", "why it matters")
    card_row(
        page, CONTENT_TOP, 2.5,
        (
            headed(
                "What have we demonstrated?",
                "How changing the observation model and the preferences can "
                "influence an agent's behaviour.",
                colour=GREEN,
            ),
            headed(
                "What remains to be established?",
                "Whether those behaviours generalise, how robust they are, and which "
                "parts of the computational model have been independently validated.",
                colour=AMBER,
            ),
        ),
        size=21,
    )  # fmt: skip
    page.text(
        MARGIN, 4.3, FULL_WIDTH, 1.6,
        [
            "The rest of this section is about cpomdp itself: what it adds, and what "
            "has been built on it."
        ],
        size=20, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. Four things to keep from today.",
            "Fixed observation noise can give the same information gain for every "
            "action, even if the overall uncertainty changes.",
            "A preference for precise observations doesn't, by itself, establish an "
            "epistemic motive. In Task 2, the expected observations and their noise "
            "shaped how the moves were scored.",
            "A route that happens to miss an unmodelled rock isn't evidence that the "
            "rocket represented it or avoided it on purpose.",
            "A certificate, a backend agreement or a working demo supports a specific "
            "claim. None of them is a general guarantee.",
        )
    )


def where_cpomdp_sits(deck: Deck) -> None:
    """Cpomdp beside its nearest neighbours, and what it adds."""
    page = deck.page("What makes cpomdp different", "why it matters")
    card_row(
        page, CONTENT_TOP, 1.55,
        (
            headed("pymdp", "Discrete states, in Python.", colour=MUTED),
            headed(
                "cpomdp",
                "Continuous states, in Python on JAX. pymdp's continuous-state "
                "sibling.",
                colour=AMBER,
            ),
            headed(
                "RxInfer",
                "Message passing on factor graphs, in Julia.",
                colour=MUTED,
            ),
        ),
        size=17,
    )  # fmt: skip
    card_row(
        page, 3.3, 2.75,
        (
            headed(
                "R(x)",
                "Noise that depends on the state makes the expected information gain "
                "differ between actions, where fixed noise makes it the same for all "
                "of them.",
                colour=AMBER,
            ),
            headed(
                "Factor graphs",
                "CouplingGraph runs models one Kalman filter can't hold, and refuses "
                "to flatten them into one.",
            ),
            headed(
                "Certificates",
                "An exhaustive search over a declared set of plans returns a "
                "certificate that counts every plan it scored.",
            ),
            headed(
                "Warrants",
                "warrantlib labels each check by how well it's warranted, apart from "
                "whether it held.",
            ),
        ),
        size=16,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. Where cpomdp sits. pymdp is the leading active inference package, "
            "and it works with discrete states. cpomdp is its continuous-state "
            "sibling, and pymdp's A, B, C and D map onto its fields. RxInfer does "
            "Bayesian inference by message passing on factor graphs, in Julia, and "
            "cpomdp checks its own filter against it.",
            "What cpomdp adds is the four cards. State-dependent noise is what made "
            "today's rockets differ from the 1960 one. The factor graph engine takes "
            "models a single filter can't hold. Certificates and warrants say how far "
            "each printed number can be trusted.",
            "This comparison is the project's own description. We haven't run the "
            "three packages against each other.",
        )
    )


def research(deck: Deck) -> None:
    """Work already built on cpomdp: biology, factor graphs, R(x) and industry."""
    page = deck.page("Already built on cpomdp", "why it matters")
    width, height, gap = (FULL_WIDTH - 0.3) / 2, 2.45, 0.3
    cards = (
        headed(
            "Biology",
            "A bacillus hunts for food it can't see, and a beacon gives the food's "
            "position away. E. coli chemotaxis runs as a factor graph, its hidden "
            "CheA hub inferred from the cell's readouts.",
            [linked(f"{CPOMDP_DOCS}examples/", size=10.5)],
            colour=GREEN,
        ),
        headed(
            "Factor graphs",
            "With R(x) on a branch, a model can't be flattened into one Kalman filter, "
            "and cpomdp refuses to try before it sees any data. With R(x), an agent "
            "reads a cue and crosses to the reward. With fixed noise, it stays put.",
            [linked(f"{CPOMDP_DOCS}examples-ffg/", size=10.5)],
        ),
        headed(
            "State-dependent observation noise",
            "Coupling R(x) to a mean the agent controls makes the posterior "
            "covariance, and so the Kalman gain, depend on the action. A preprint.",
            [linked(f"{SITE}research/state-dependent-observation-noise/", size=10.5)],
            colour=AMBER,
        ),
        headed(
            "Industry: a compressor's surge margin",
            "Two flow meters of equal accuracy, with noise that moves opposite ways "
            "with flow, settle the same agent on opposite sides of the surge line.",
            [linked(f"{SITE}projects/cpomdp/applied/anti-surge/", size=10.5)],
            colour=BLUE,
        ),
    )
    for index, paragraphs in enumerate(cards):
        row, column = divmod(index, 2)
        page.card(
            MARGIN + column * (width + gap), CONTENT_TOP + row * (height + gap), width,
            height, paragraphs, size=16, space_after=6,
        )  # fmt: skip
    page.notes(
        notes(
            "Kev. Pointers, not results. Each card links to the write-up.",
            "Biology. In the bacillus example, four bacilli differ only in how "
            "precise their goal is. Some go straight for the food, one detours to the "
            "beacon first, one parks at the beacon. The chemotaxis model is the "
            "shape of the signalling network, not its biophysics.",
            "Factor graphs. The two-node models are the shapes a single Kalman filter "
            "isn't. The cue example is a statement about control, not biology.",
            "State-dependent noise. The preprint is the theory behind today's Task 2 "
            "and Task 3: once R depends on a mean the agent controls, no fixed "
            "linear-Gaussian filter reproduces it. The README adds a caution. "
            "Non-constant isn't the same as decision-changing.",
            "Industry. An agent holds a compressor's surge margin. A "
            "differential-pressure meter keeps it on the safe side. A Coriolis meter "
            "of the same accuracy settles it past the surge line. The write-up's "
            "conclusion is that the information term isn't a safety factor, and surge "
            "protection belongs in the setpoint or a hard limit.",
        )
    )


def programmes(deck: Deck) -> None:
    """The research programmes proposed on top of cpomdp."""
    page = deck.page("Where the research goes next", "why it matters")
    card_row(
        page, CONTENT_TOP, 4.1,
        (
            headed(
                "p*",
                [Run("certifiable active inference", colour=MUTED)],
                "Hold the agent's model and the world apart. How much free energy "
                "does an agent pay for being wrong about the world, and how much for "
                "being wrong about what its own model implies?",
                [Run("Active", colour=GREEN, bold=True)],
                colour=AMBER,
            ),
            headed(
                "Seam Networks",
                [Run("agents that meet through a world", colour=MUTED)],
                "Active inference agents that meet only through a shared world. Does "
                "that structure make sense to explore, and can cpomdp tell?",
                [Run("Planned. Nothing has run yet.", colour=AMBER, bold=True)],
            ),
            headed("The Individuating Membrane"),
        ),
        size=17,
    )  # fmt: skip
    third = MARGIN + 2 * (FULL_WIDTH + 0.3) / 3
    page.placeholder(
        third + 0.25, CONTENT_TOP + 0.75, (FULL_WIDTH - 0.6) / 3 - 0.5, 1.6,
        "[owner: one line on The Individuating Membrane, and its status]", size=15,
    )  # fmt: skip
    page.text(
        MARGIN, 6.05, FULL_WIDTH, 0.7,
        [[Run("p* and its gates  "), linked(f"{SITE}programmes/p-star/", size=15)]],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. Three programmes, at different stages.",
            "p* is active. The standard derivation lets one distribution play both "
            "the agent's model and the world, so a wrong model has nowhere to show "
            "up. p* holds the two apart and asks how tightly each cost can be "
            "certified. Its gates are public, passes and failures both.",
            "Seam Networks is a string of thought experiments that grew out of p* and "
            "cpomdp's harness. Each cell is a whole active inference agent, and cells "
            "meet only through a world object, the seam. It's planned, and nothing "
            "has run.",
            "TODO owner: The Individuating Membrane, a line for the card and what to "
            "say here.",
            "TODO owner: the lost deck's 'What comes next' slide carried the status "
            "'scoped, starts after p*'. Put it on the programme it belonged to.",
        )
    )


# --- close ------------------------------------------------------------------------


def keep_going(deck: Deck) -> None:
    """The self-guided cells, and where everything lives."""
    page = deck.page("Keep going", "questions")
    blurbs = {
        "Two moves ahead": "The world again, two moves ahead. Each step scores 81 "
        "plans in place of 9.",
        "A model one filter cannot hold": "Two nodes that cpomdp refuses to flatten "
        "into one Kalman filter, with its reason.",
    }
    card_row(
        page, CONTENT_TOP, 1.75,
        tuple(headed(name, blurbs[name]) for name, _ in lesson.TAIL),
        size=17,
    )  # fmt: skip
    top = 3.45
    for label, url in (
        ("repository", REPOSITORY),
        ("cpomdp on PyPI", PYPI),
        ("Inferogenesis", SITE),
    ):
        page.text(MARGIN, top, 4.0, 0.35, [label], size=15, colour=MUTED)
        page.link(MARGIN, top + 0.36, FULL_WIDTH, 0.35, url, size=12)
        top += 0.78
    page.text(MARGIN, top, 4.0, 0.35, ["Discord"], size=15, colour=MUTED)
    page.text(
        MARGIN, top + 0.36, 7.0, 0.45, ["the cpomdp channel on the AII Discord"],
        size=17,
    )  # fmt: skip
    page.placeholder(
        8.0, top, RIGHT - 8.0, 0.8, f"[Zenodo DOI of cpomdp {WORKSHOP_CPOMDP_VERSION}]"
    )
    page.notes(
        notes(
            f"Planned at {clock('Questions')}. The open time starts here. The "
            "self-guided cells sit below the last checkpoint, and each stands on "
            "checkpoint 3. Nothing there runs live.",
            "Recording, notebook, package, slides and notes go to the Discord channel "
            "and the README within 48 hours.",
            "TODO owner: the Zenodo DOI once minted.",
        )
    )


def thanks(deck: Deck) -> None:
    """Thanks, questions, the survey while the room is still here, and two QR codes."""
    page = deck.page()
    page.mark(MARGIN, 0.55, 0.85)
    side, gap, top = 1.9, 0.4, 1.75  # each QR code, the clear space between, its top
    codes = RIGHT - 2 * side - gap  # where the QR codes start
    column = codes - gap - MARGIN  # the width the words take, left of the codes
    page.text(MARGIN, 1.6, column, 1.2, ["Thank you"], size=60, bold=True)
    page.text(
        MARGIN, 2.95, column, 1.2,
        ["Questions now, or on the anonymous form.", "One up, one down before you go."],
        size=24, colour=MUTED, space_after=10,
    )  # fmt: skip
    width = (column - 0.4) / 2
    page.placeholder(MARGIN, 4.45, width, 0.8, "[survey link]")
    page.placeholder(MARGIN + width + 0.4, 4.45, width, 0.8, "[email]")
    page.text(
        MARGIN, 5.9, column, 0.6,
        [f"Everything shown runs on cpomdp {WORKSHOP_CPOMDP_VERSION} from PyPI."],
        size=18, colour=MUTED,
    )  # fmt: skip
    for index, (url, caption) in enumerate(
        ((PRESENTER_2_LINKEDIN, "Kevin on LinkedIn"), (SITE, "inferogenesis.com"))
    ):
        left = codes + index * (side + gap)
        page.qr(url, left, top, side)
        page.text(
            left, top + side + 0.15, side, 0.4, [caption], size=15, colour=MUTED,
            align=PP_ALIGN.CENTER, space_after=0,
        )  # fmt: skip
    page.notes(
        notes(
            "Leave this up through the questions. The survey link goes in the chat as "
            "well, while the room is still here. One up, one down: one thing that "
            "worked and one that didn't.",
            f"Everything shown runs on cpomdp {WORKSHOP_CPOMDP_VERSION} from PyPI.",
            "The QR codes open Kev's LinkedIn and inferogenesis.com.",
            "TODO owner: the survey link and the email.",
        )
    )


SLIDES: tuple[Callable[[Deck], None], ...] = (
    title,
    who_we_are,
    disclosure,
    agenda,
    how_we_work,
    setup,
    preview,
    task_1_brief,
    task_1_code,
    task_1_timer,
    task_1_result,
    task_1_meaning,
    task_2_brief,
    task_2_code,
    task_2_timer,
    task_2_result,
    task_2_meaning,
    task_3_brief,
    task_3_code,
    task_3_timer,
    task_3_blackout,
    task_3_rock,
    task_3_meaning,
    backends,
    coda,
    beginning,
    where_cpomdp_sits,
    research,
    programmes,
    keep_going,
    thanks,
)


def number_of(slide: Callable[[Deck], None]) -> int:
    """Where ``slide`` falls in the deck, counting from one."""
    return SLIDES.index(slide) + 1


def build() -> Deck:
    """Every slide, in order."""
    deck = Deck("Build a curious rocket", author="Inferogenesis")
    for slide in SLIDES:
        slide(deck)
    return deck


def problems(deck: Deck) -> list[str]:
    """What the layout check finds, with a line if the text fit went unchecked."""
    found = layout_problems(deck.presentation)
    measure = metrics()
    if measure is None:
        print("No Calibri, Courier New or Cambria files found. Text fit not checked.")
        return found
    return found + fit_problems(deck.presentation, measure)


def main() -> None:
    """Build the deck, check its layout and write it, then the presenting copy."""
    deck = build()
    found = problems(deck)
    if found:
        print("\n".join(found))
        sys.exit(1)
    deck.save(DECK)
    print(f"wrote {DECK.relative_to(ROOT)}, {len(SLIDES)} slides")
    if not PHOTOS.is_dir():
        print(f"no {PHOTOS.relative_to(ROOT)}/, so no presenting copy")
        return
    deck.fill_photo_frames(PHOTOS)
    deck.save(PRESENTING)
    print(f"wrote {PRESENTING.relative_to(ROOT)}, with the presenters' photos")


if __name__ == "__main__":
    main()
