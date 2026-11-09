"""Build the workshop deck from the package, the lesson and the figures.

    uv run --no-sync python scripts/render_assets.py   # the figures, first
    uv run --no-sync python scripts/build_deck.py

The two hours run as an introduction, three tasks and a closing section on why cpomdp
matters, then time left open for questions. Each task is a brief, a slide that stays
up with a fifteen-minute bar while the room works, and a discussion of what the room
built. The room reads the code on its own screen, so no slide repeats it. ``AGENDA``
holds the plan on 5-minute steps from ``START``, and every clock time on the agenda
slide and in the notes is read from it, in UTC.

The slides carry the route figures and what the cells print. Printed output comes from
running the package on the pinned release, figures from ``assets/``. Every number in
the prose and the notes is asserted in ``tests/test_measured_claims.py`` or named in
the deck test as set. The build refuses to write a deck whose shapes run off a slide,
overlap or overflow.

The presenters' photos are never committed. The committed deck shows an empty frame
for each. When ``deck/photos/`` holds them, the build also writes a presenting copy
with each photo in its frame, and git ignores both the photos and that copy.
"""

import contextlib
import io
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from cpomdp import KalmanBackend
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

from cpomdp_workshop import (
    WORKSHOP_CPOMDP_VERSION,
    after_session_3,
    axis_certificate,
    beacon_sensor,
    check_backends,
    default_params,
    grade_sensor,
    model_with,
    one_node,
    probe,
    show_probe,
)
from cpomdp_workshop._seams import RecedingHorizonSelector
from deck_kit import (
    AMBER,
    BLUE,
    CODE_FONT,
    CONTENT_TOP,
    GREEN,
    INK,
    MARGIN,
    MUTED,
    SLIDE_WIDTH,
    Deck,
    Page,
    Paragraph,
    Run,
    code_height,
)
from deck_layout import fit_problems, layout_problems, metrics
from timer_clip import CLIP, POSTER

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
KOUDAHL_REFERENCE = (
    "Koudahl, M. T., Kouw, W. M. and de Vries, B. (2021). "
    "On Epistemics in Expected Free Energy for Linear Gaussian State Space Models. "
    "Entropy. https://doi.org/10.3390/e23121565"
)

RIGHT = SLIDE_WIDTH - MARGIN  # the right edge of the content
FULL_WIDTH = RIGHT - MARGIN
# Every route slide puts its figure here, so only the route changes between them.
ROUTE_FIGURE = (MARGIN, 1.4, 8.4)  # x, y, width
ROUTE_COLUMN = 9.3  # where the callouts beside a route frame start
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
    Part("brief", "talk", 5),
    Part("you code", "code", TASK_MINUTES),
    Part("discussion", "talk", 10),
)
"""Every task's parts."""

AGENDA = (
    Block("Welcome", (Part("introductions and setup", "talk", 10),)),
    Block("Task 1", TASK_PARTS, "the 1960 rocket"),
    Block("Task 2", TASK_PARTS, "the beacon"),
    Block("Task 3", TASK_PARTS, "the blackout"),
    Block("Why it matters", (Part("what cpomdp adds, and what's next", "talk", 10),)),
    Block("Questions", (Part("questions, or leave early", "open", 10),)),
)
"""The two hours: 65 minutes of talk, 45 of code and 10 left open, on 5-minute steps."""

START = 17 * 60
"""When the session starts, in minutes after midnight UTC."""

QUESTIONS_FORM = "https://docs.google.com/forms/d/e/1FAIpQLSf7mgwxYEFpnDAkqaYolxWzP1sesG6L5Rf_5pVswE3UwtkcwA/viewform"
"""The question form. Anonymous unless the asker leaves an email for a reply."""


def clock(label: str, part: int = 0, *, after: int = 0) -> str:
    """When part ``part`` of the block called ``label`` starts, as UTC ``hh:mm``.

    ``after`` adds minutes, so ``after=TASK_MINUTES`` on a coding part is its end.
    """
    elapsed = START + after
    for block in AGENDA:
        if block.label == label:
            elapsed += sum(earlier.minutes for earlier in block.parts[:part])
            return f"{elapsed // 60}:{elapsed % 60:02d}"
        elapsed += block.minutes
    raise ValueError(f"no block called {label!r} in the agenda")


def span(block: Block) -> str:
    """When ``block`` runs, as ``hh:mm–hh:mm``."""
    return f"{clock(block.label)}–{clock(block.label, after=block.minutes)}"


PYMDP_MAP = (
    ("A", "the observation likelihood", ("observation_matrix,", "observation_noise")),
    ("B", "the transition", ("dynamics_matrix,", "control_matrix")),
    ("C", "the preference", ("Preference", "from Task 2")),
    ("D", "the prior", ("prior",)),
)
"""pymdp's letters and the cpomdp fields that hold the same thing."""


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


def notes(*paragraphs: str) -> str:
    """Speaker notes, one paragraph per argument."""
    return "\n\n".join(paragraphs)


def linked(url: str, *, size: float = 14) -> Run:
    """A URL as a run that opens it, in the code face, without its scheme."""
    return Run(url.removeprefix("https://"), font=CODE_FONT, size=size, link=url)


FILL_IN = "Fill in the two lines marked "
"""How a timer slide's first step opens. The notebook marks the lines with ``...``."""


def timer_page(deck: Deck, number: int, steps: tuple[str, ...]) -> Page:
    """The slide that stays up while the room works: what to do, and the bar.

    The room has the code on its own screen, so the slide carries the steps and the
    bar. A step that opens with ``FILL_IN`` gets the notebook's ``...`` in amber.
    """
    page = deck.page(f"Task {number} · Over to you", f"task {number}")
    paragraphs: list[Paragraph] = []
    for index, step in enumerate(steps, start=1):
        runs = [Run(f"{index}   ", colour=AMBER, bold=True)]
        if step.startswith(FILL_IN):
            runs += [
                Run(FILL_IN),
                Run("...", font=CODE_FONT, colour=AMBER, bold=True),
                Run(step.removeprefix(FILL_IN)),
            ]
        else:
            runs.append(Run(step))
        paragraphs.append(runs)
    page.text(
        MARGIN, CONTENT_TOP + 0.1, FULL_WIDTH, 1.9, paragraphs, size=22, space_after=12
    )
    page.countdown(
        MARGIN, 4.3, FULL_WIDTH, TASK_MINUTES,
        then=[
            [
                Run("Time's up. ", colour=AMBER),
                Run(f"Run checkpoint {number}, even if everything worked."),
            ]
        ],
        clip=CLIP, poster=POSTER,
    )  # fmt: skip
    page.text(
        MARGIN, 6.45, FULL_WIDTH, 0.75,
        [
            [
                Run("Stuck? Open "),
                Run("Hint 1", bold=True),
                Run(", "),
                Run("Hint 2", bold=True),
                Run(" or the "),
                Run("Reference", bold=True),
                Run(" below the cell."),
            ],
            [
                Run("An error?  "),
                Run("workshop.symptoms()", font=CODE_FONT),
                Run("  lists the five most common."),
            ],
        ],
        size=16, colour=MUTED, space_after=2,
    )  # fmt: skip
    return page


def timer_notes(number: int, *paragraphs: str) -> str:
    """A timer slide's notes: when it runs, ``paragraphs``, then how the bar runs."""
    task = f"Task {number}"
    return notes(
        f"Planned at {clock(task, 1)}, until {clock(task, 1, after=TASK_MINUTES)}. "
        "Keyboard.",
        *paragraphs,
        f"Leave this slide up for the whole {TASK_MINUTES} minutes. The bar starts "
        "when the slide appears and needs no click, and the clock above it counts "
        "down the time left to the second. Both start again if you leave the "
        "slide and come back, so share the notebook from another window if you need "
        "to.",
        "Nobody pastes hints. Each person opens the collapsed hint and reference "
        "cells when they like. Paste the question form link in the chat again as the "
        f"bar starts: {QUESTIONS_FORM}. Whoever isn't at the keyboard watches the "
        "chat and the form's responses. We don't debug anyone's code from the front. "
        "Point people at the hints, the reference cell and workshop.symptoms().",
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


def question(page: Page, top: float, text: str, *, height: float = 0.95) -> None:
    """A question left open for the room, under the slide's content."""
    page.text(
        MARGIN, top, FULL_WIDTH, height, [[Run(text, colour=AMBER, bold=True)]], size=22
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
            f"Planned at {clock('Welcome')} UTC. Keyboard shares the screen with this "
            "slide up and pastes the Colab link and the question form in the chat. "
            "Kev opens.",
            "Kev - Welcome, everyone. Over the next two hours, we'll build three "
            "rockets and watch how each one decides where to fly. If you could open "
            "the Google Colab notebook now, even while we're talking. The next few "
            "slides say how, and the installs can run while we introduce ourselves. "
            "We will have a break around the hour mark.",
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
            "Dan\vI was introduced through Active Inference via\nWe built some things "
            "in pymdp blah\nI have a personal interest in micro-bio and Abiogenesis "
            "which led to the question, could we show evidence in silico that "
            "microorganisms are “doing” active inference. \vI came across a series of "
            "papers by a researcher called Mattingly (fill out) that looks at the "
            "information rate of chemotaxis and run & tumble actions. The limit here "
            "is the discretisation artifacts and errors that the binning process "
            "would introduce, that would likely swamp the very specific and tiny "
            "information rates we’d look to replicate.",
            "That led to cpomdp which started its life as a python serving RxInfer "
            "over a juliabridge. The issue here was that the latency introduced via "
            "Juliabridge undermined the speed and accuracy of the RxInfer message "
            "passing that makes it so great. I then took a page out of pymdp and "
            "transitioned to Python with JAX, making cpomdp public at the same time. "
            "Since then it has grown to a state that I am happy sharing with you all "
            "today.",
            "Dan (aloud)\nSome of the source code in "
            f"cpomdp {WORKSHOP_CPOMDP_VERSION} was written by Claude Code and "
            "reviewed before release.",
        )
    )


def agenda(deck: Deck) -> None:
    """The two hours, one row per block, each part named and as wide as its minutes.

    Each row's heading carries its UTC times. A part in a block of several carries its
    minutes under its name, so a task shows its brief, code and discussion lengths.
    """
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
            MARGIN, top, bars - MARGIN - 0.2, height,
            [heading, [Run(span(block), colour=MUTED, size=14)]],
            size=20, anchor=MSO_ANCHOR.MIDDLE, space_after=0,
        )  # fmt: skip
        left = bars
        for part in block.parts:
            lines: list[Paragraph] = [part.name]
            if len(block.parts) > 1:
                lines.append([Run(f"{part.minutes} minutes", size=13)])
            page.block(
                left + gap / 2, top, part.minutes * inch - gap, height, part.kind,
                label=lines, size=16,
            )  # fmt: skip
            left += part.minutes * inch
    page.text(
        MARGIN, CONTENT_TOP + 0.15 + len(AGENDA) * pitch + 0.15, FULL_WIDTH, 0.5,
        [
            f"All times UTC. While you code, a {TASK_MINUTES}-minute timer stays on "
            "the screen."
        ],
        size=19, colour=MUTED, space_after=0,
    )  # fmt: skip
    page.notes(
        notes(
            "Dan continues\nWe’ll keep moving and avoid too much talking, we want to "
            "get people hands on keyboards by quarter minutes past.\nBriefly, you "
            f"will have {TASK_MINUTES} minutes to code, attempt the main task and any "
            "bonuses, followed by a brief discussion of what you’ve just wrote and "
            "why.\nThere are 3 tasks.\nThen we will discuss cpomdp and why you should "
            "a little more afterwards, followed by questions.",
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
                    " in the code. We have written the rest so you can focus on the "
                    "ideas. Some bonus exercises give you a chance to write more of "
                    "your own code."
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
            "Each task closes with a question in the notebook to think about.",
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
            "Kev. Each task has two lines to write, and we have written the rest of "
            "the code, so the Python stays out of the way. Some of the bonus "
            "exercises ask you to write your own.",
            "Each task has two hints and a worked solution, collapsed below the cell. "
            "A hint shows a nudge first, and the line itself is behind Show code. "
            "Open them whenever you like, and using them is absolutely fine - this is "
            "not a coding test. We encourage you not to use an AI agent, we feel you "
            "would get more from the workshop doing it by hand, but feel free to use "
            "an agent if needed",
            "At the end of every task, run the checkpoint cell, even if everything "
            "worked. It puts the reference names back, so the next task starts from "
            "the same place for everyone.",
            "Each task closes with one number in the chat. It lets us see how the "
            "room is doing without anyone writing a sentence or alternatively just "
            "gives a thumbs up. If you do not finish the task do not worry, you can "
            "save and come back to in your own time, which Dan will discuss in the "
            "next slide.",
        )
    )


LOCAL_STEPS = (
    ("Get the notebook.", f"git clone {REPOSITORY}\ncd cpomdp-workshop"),
    (
        "Make an environment, then install the workshop with Jupyter.",
        "python -m venv .venv\n"
        "source .venv/bin/activate  # Windows: .venv\\Scripts\\activate\n"
        'pip install "cpomdp-workshop[nb]=={version}"',
    ),
    (
        "Open notebooks/build_a_curious_rocket.ipynb in VS Code, or run jupyter lab. "
        "Pick .venv as the kernel and run cell 1.",
        "",
    ),
)
"""The local route, step by step. The wheel carries no notebook and no Jupyter."""


def setup(deck: Deck) -> None:
    """Google Colab, or the room's own Python in three steps, then the two links."""
    page = deck.page("Get set up", "welcome")
    colab = 3.6
    page.text(
        MARGIN, CONTENT_TOP, colab, 3.8,
        [
            [Run("Google Colab", bold=True, size=26, colour=AMBER)],
            [Run("Recommended", colour=MUTED)],
            "It runs in your browser, so there's nothing to install.",
            "Open the notebook link below, then run cell 1. It installs the "
            "workshop and checks it's ready.",
            [
                Run("If you would like to keep your code for later, in Colab click "),
                Run("File -> Save a Copy in Drive", bold=True),
            ],
        ],
        size=18, space_after=8,
    )  # fmt: skip
    right = MARGIN + colab + 0.5
    width = RIGHT - right
    page.text(
        right, CONTENT_TOP, width, 0.5,
        [
            [
                Run("Your own Python", bold=True, size=26, colour=BLUE),
                Run("   Python 3.11 or newer", colour=MUTED),
            ]
        ],
        size=18, space_after=0,
    )  # fmt: skip
    version = metadata.version("cpomdp-workshop")
    top = CONTENT_TOP + 0.6
    for index, (step, commands) in enumerate(LOCAL_STEPS, start=1):
        lines = 2 if len(step) > 70 else 1
        page.text(
            right, top, width, 0.33 * lines,
            [[Run(f"{index}  ", colour=BLUE, bold=True), Run(step)]],
            size=17, space_after=0,
        )  # fmt: skip
        top += 0.33 * lines + 0.06
        if commands:
            source = commands.format(version=version)
            height = code_height(source, 13)
            page.code(right, top, width, height, source, size=13)
            top += height + 0.1
    top = 5.55
    for label, url in (
        ("notebook in Colab", COLAB),
        ("questions, anonymous unless you leave an email", QUESTIONS_FORM),
    ):
        page.text(MARGIN, top, 6.0, 0.33, [label], size=15, colour=MUTED)
        page.link(MARGIN, top + 0.34, FULL_WIDTH, 0.33, url, size=12)
        top += 0.75
    page.notes(
        notes(
            "Dan\nWe’ve kept the workshop content as low-friction as possible for any "
            "programming level.\nHere are the two links, they are already posted in "
            "the Zoom chat.\nIf you’d like to use your own environment hopefully "
            "you’ve set it up before now, if not I would encourage the default to be "
            "Colab. I would also encourage you save a copy in case your tab closes or "
            "if you’d like to continue any of the tasks after the session. Each "
            "Colab instance is private to yourself, no one else can see your code so "
            "don’t worry about making mistakes.",
            "The first time you run a cell, Colab warns the notebook wasn't authored "
            "by Google: click Run anyway. Nothing in the workshop asks for access to "
            "anything on your drive.",
            "The second link is an anonymous questions form, we know it can be quite "
            "intimidating to ask questions in a room full of people so using that "
            "form only hands us the question and nothing else. If your question "
            "requires a longer form response leave an email and we’ll get back to you "
            "after the session.",
            f"The two links to paste: {COLAB} and {QUESTIONS_FORM}",
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
            "Kev - If you could run the first and second code cell, that would be "
            "great",
            "This is just a check to make sure everything is working for you, and if "
            "you can see the graph that is on the slide - give us a thumbs up",
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
    question(
        page, 3.2, "Are all possible moves equally informative?", height=0.55,
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
            "Kev - NASA used a Kalman filter on Apollo because every measurement had "
            "some uncertainty. The filter combined where the spacecraft was predicted "
            "to be with what the astronauts actually observed. The more reliable the "
            "observation, the more weight it got in the new estimate.",
            "In Karl Friston’s account of active inference, we update our beliefs by "
            "minimising free energy, and in a linear Gaussian model like ours, that "
            "gives us exactly the same update as the Kalman filter.\v\vThe linear "
            "quadratic regulator then uses that estimate to decide how much thrust to "
            "apply, balancing the cost of being off target against the cost of firing "
            "the thruster harder.",
        )
    )


def task_1_timer(deck: Deck) -> None:
    """Task 1's fifteen minutes."""
    page = timer_page(
        deck, 1,
        (
            f"{FILL_IN} and run the cell.",
            "Paste your spread in the chat.",
            "Finished early? Try the two bonuses below the cell.",
        ),
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 1', 1)}, until "
            f"{clock('Task 1', 1, after=TASK_MINUTES)}. Kev.",
            "Two lines to write: what the sensor observes and how noisy its readings "
            "are.",
            "Once you run the cell, you will be given an information spread value. "
            "Paste that in the chat. This is the variation in epistemic values across "
            "the action selections.",
            "Have a go at the bonus at your own leisure.",
        )
    )


def task_1_result(deck: Deck) -> None:
    """The reference route and spread, with prompts for the task 1 discussion."""
    spread = captured(show_probe, probe(model_with())).split("   ")[-1]
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
        ROUTE_COLUMN, 1.88, width, 0.5, [spread],
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
    page.notes(
        notes(
            f"Planned at {clock('Task 1', 2)}. Dan",
            "Rhetorically ask what a spread of zero tells us. The probe function "
            "compares the information each of the nine compass moves is expected to "
            "give. spread = 0.000000 nats basically means each action gives the exact "
            "same information. To be clear zero spread does not mean zero information "
            "gain and if you completed the bonus, you would have seen they alter the "
            "amount of information gained, while the spread still remains zero.",
            "My usual analogy for this is walking into an optician's appointment and "
            "being asked to read the letters on that strange board they have and if "
            "you're like me, completely blind, you can't see anything. If you were to "
            "then get up and press your nose against the board, but the letters were "
            "still equally blurry, you have no incentive to move closer to read the "
            "letters, or in more technical terms, you have no drive to act as the "
            "information gain is the same regardless of where you stand.",
            "This result is formalised in Koudahl and colleagues - On Epistemics in "
            "Expected Free Energy for Linear Gaussian State Space Models. Entropy. "
            "https://doi.org/10.3390/e23121565 where they show that there is no "
            "information spread for linear-gaussian regimes, essentially meaning "
            "active inference epistemic drive is absent. Not so useful when your "
            "entire goal is to make a toolbox that builds linear-Gaussian active "
            "inference agents. You are now at the roadblock I hit about a month into "
            "development. Welcome.",
        )
    )


# --- task 2 -----------------------------------------------------------------------


def task_2_brief(deck: Deck) -> None:
    """Why position-dependent sensor noise can make some moves more informative."""
    page = deck.page("Making information gain depend on the action", "task 2")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.85,
        [
            [Run("Like your eyes, most sensors blur with range.", bold=True)],
            [
                Run("R", bold=True),
                Run(" is the noise in the sensor’s readings. "),
                Run("R(x)", bold=True),
                Run(" means that noise depends on the state x"),
            ],
        ],
        size=20, space_after=4,
    )  # fmt: skip
    card_row(
        page, 2.75, 1.4,
        (
            headed(
                "Fixed R · Task 1",
                "Equally noisy wherever the rocket goes. Every move teaches it the "
                "same.",
            ),
            headed(
                "R(x) · Task 2",
                "Readings are sharper near the beacon. Moving closer means the rocket "
                "expects to learn more.",
                colour=AMBER,
            ),
        ),
    )  # fmt: skip
    page.text(
        MARGIN, 4.45, FULL_WIDTH, 0.5,
        [
            "where it moves  →  how reliable its next reading is  →  how much it "
            "expects to learn"
        ],
        size=20, colour=MUTED, align=PP_ALIGN.CENTER,
    )  # fmt: skip
    question(
        page, 5.2,
        "Task 2: does R(x) give a linear-Gaussian rocket an epistemic drive?",
        height=0.55,
    )  # fmt: skip
    page.text(
        MARGIN, 6.1, FULL_WIDTH, 0.6,
        [
            "Another route is multiplicative control (Koudahl, Kouw and de Vries "
            "2021). It’s planned for cpomdp and not built yet."
        ],
        size=15, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 2')}.",
            "Now imagine the precision of your eyes varies with where you stand in "
            "that optician’s room, much like real life. Your vision gets blurrier "
            "with range, and so do almost all our sensors.",
            "We write the observation noise matrix as R for short. R describes the "
            "noise in the sensor readings. Writing R(x) means this noise depends on "
            "the rocket’s position, where x is the state. In this 2D example the "
            "state is just position.",
            "With a fixed R, the sensor is equally noisy wherever the rocket goes. "
            "With R(x), the noise depends on the rocket’s position. Readings are more "
            "reliable near the beacon, so by moving closer the rocket can expect to "
            "learn more.",
            "Task 2 tests whether this returns epistemic drive to a linear-Gaussian "
            "agent.",
            "A quick caveat. There are other ways to restore epistemics. Koudahl and "
            "colleagues use multiplicative control. It isn’t in cpomdp yet, and it’s "
            "planned as part of future development.",
            "What you’ll need this time, is to set a beacon_gain value denoted as k "
            "in the code cell. This sets how quickly the noise grows with distance "
            "from the beacon. The goal type also changes: this rocket prefers a "
            "reading, the one that says it’s on the pad, where the 1960 rocket "
            "preferred a position. At each step its planner scores the nine compass "
            "moves by expected free energy.",
            "Reference: Corva, D. (2026). State-Dependent Observation Noise "
            "Reintroduces Epistemic Value in Linear-Gaussian Active Inference. "
            "https://arxiv.org/abs/2607.20306",
            f"Further references: {KOUDAHL_REFERENCE}",
            "Franklin, D. W. et al. (2007). Endpoint Stiffness of the Arm Is "
            "Directionally Tuned to Instability in the Environment. "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6672883/",
        )
    )


def task_2_timer(deck: Deck) -> None:
    """Task 2's fifteen minutes."""
    page = timer_page(
        deck, 2,
        (
            f"{FILL_IN} and start with k = 0.08.",
            "Raise k until the route bends and still lands. Paste your k in the chat.",
            "Finished early? Try the two bonuses below the cell.",
        ),
    )  # fmt: skip
    page.notes(
        notes(
            "Kev - For Task 2, you’ve got two lines to fill in again. First, set k to "
            "0.08. Then, in my sensor, fill in the line for the beacon’s extra noise.",
            "You’ll find the beacon settings in params, and the comments show you how "
            "to look them up by name.",
            "Run the cell to see the curious rocket fly with your sensor and plot its "
            "route.",
            "Then try increasing k. Can you get the rocket to take a different route "
            "and still land on the pad? If you’re stuck, Hint 1 suggests trying k = "
            "0.4.",
        )
    )


def task_2_result(deck: Deck) -> None:
    """Compare the reference flight with the routes the room found."""
    k = 0.4
    result = grade_sensor(beacon_sensor, default_params(k))
    if result.bend is None or result.distance is None:
        raise ValueError(f"No reference flight for the Task 2 slide: {result.message}")
    page = route_page(deck, "Task 2 · Discussion", "task 2", "route_2_beacon.png")
    left, _, plot_width = ROUTE_FIGURE
    page.text(
        left, 5.99, plot_width, 0.35, [f"Reference run with k = {k:.2f}"],
        size=16, bold=True, space_after=0,
    )  # fmt: skip
    page.text(
        left, 6.40, plot_width, 0.75,
        [f"At its furthest point from the straight route, the rocket was "
         f"{result.bend:.2f} units off course. It finished {result.distance:.2f} "
         "units from the pad."],
        size=17, colour=MUTED,
    )  # fmt: skip
    width = RIGHT - ROUTE_COLUMN
    page.text(
        ROUTE_COLUMN, 1.85, width, 1.45,
        ["How did the route change as you increased k?"],
        size=21, bold=True,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 3.65, width, 2.50,
        ["Which values of k gave you a route that curved towards the beacon "
         "and still reached the pad?"],
        size=20, bold=True,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 6.65, width, 0.35, ["Poll 2 · 60 seconds"],
        size=16, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            "Welcome back, everyone. You’ve just had a chance to fly the curious "
            "rocket yourselves.",
            "You started at k = 0.08 and turned it up to see whether the rocket would "
            "take a detour towards the beacon and still reach the pad.",
            "Let’s look at our run. The dashed line shows the 1960 rocket heading "
            "straight for the pad.",
            f"The amber line shows the curious rocket at k = {k:.2f}. It curves "
            "towards the beacon, where its readings are more reliable, then heads "
            f"back towards the pad and finishes just {result.distance:.2f} units away.",
            "When we turn k up to 0.6, the rocket stops short of the pad.",
        )
    )


def task_2_meaning(deck: Deck) -> None:
    """Explain the poll result by comparing readings at the pad and near the beacon."""
    page = deck.page("Why does the rocket stop short of the pad?", "task 2")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.7,
        ["The rocket prefers a sensor reading that places it on the landing pad."],
        size=22,
    )  # fmt: skip
    page.text(
        MARGIN, 2.60, FULL_WIDTH, 1.05,
        [[Run("At the pad, ", bold=True),
          Run("readings are centred on the target, but high sensor noise means "
              "they can fall a long way from it.")]],
        size=22,
    )  # fmt: skip
    page.text(
        MARGIN, 3.95, FULL_WIDTH, 1.05,
        [[Run("Nearer the beacon, ", bold=True),
          Run("readings are centred short of the target, but lower sensor noise "
              "keeps them closer together.")]],
        size=22,
    )  # fmt: skip
    page.text(
        MARGIN, 5.85, FULL_WIDTH, 1.05,
        ["At k = 0.6, moving closer to the pad can increase the expected error "
         "in the next reading. In the reference run, the rocket stops short."],
        size=21, bold=True, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev - The rocket weighs up two things when it chooses a move. How much "
            "it expects to learn about its position is the epistemic value.",
            "How close it expects its next reading to be to the pad is the pragmatic "
            "value.",
            "In this run, both can favour moving towards the beacon.",
            "Even if the rocket lands right on the pad, the sensor is noisy there, so "
            "a reading could say it missed by a long way.",
            "A little short of the pad, the readings are steadier and tend to show "
            "only a small miss. At k = 0.6, the rocket expects a smaller error in its "
            "next reading if it stays there.",
            "That’s why even the pragmatic goal favours stopping short.",
        )
    )


# --- task 3 -----------------------------------------------------------------------


def task_3_brief(deck: Deck) -> None:
    """The blackout, and the rock none of the rockets models."""
    page = deck.page("Task 3 · The blackout", "task 3")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.65,
        ["The direct route to the landing pad now crosses a blackout region."],
        size=22,
    )  # fmt: skip
    card_row(
        page, 2.45, 2.45,
        (
            headed(
                "The beacon",
                [Run("Near the beacon, sensor noise is "), Run("lower", bold=True),
                 Run(" and readings are "), Run("more reliable", bold=True), Run(".")],
            ),
            headed(
                "The blackout",
                [Run("Near the centre of the blackout, sensor noise is "),
                 Run("higher", bold=True), Run(" and readings are "),
                 Run("less reliable", bold=True), Run(".")],
                colour=BLUE,
            ),
            headed(
                "The rock",
                "A rock sits inside the blackout. All three rockets plan their "
                "moves without knowing it is there.",
            ),
        ),
    )  # fmt: skip
    question(
        page, 5.45,
        "Will all three rockets take the same route through the blackout?",
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 3')}. Kev. In Task 2, the beacon made "
            "observations more reliable nearby. Now we add a blackout on the "
            "straight line to the pad, where the noise rises sharply towards its "
            "centre.",
            "The rock sits inside the blackout. In the language of Active Inference "
            "(Figure 2.2), it belongs to the generative process. It is part of the "
            "world the rockets fly through. All three plan their moves without "
            "knowing it is there.",
            "We will compare the 1960 rocket with two curious rockets. The slider "
            "sets k for yours, and the other uses k = 0.20. Leave the route "
            "question with participants as they start Task 3.",
        )
    )


def task_3_timer(deck: Deck) -> None:
    """Task 3's fifteen minutes."""
    page = timer_page(
        deck, 3,
        (
            f"{FILL_IN} and run the cell.",
            "Paste max |difference| in the chat.",
            "Then move the k slider and watch the routes. Finished early? Try the "
            "bonuses.",
        ),
    )  # fmt: skip
    page.notes(
        timer_notes(
            3,
            "Two lines again: the blackout's extra noise in my_world, and the 1960 "
            "rocket's model as a graph of one node. The slider sets k for your "
            "rocket. There's a rock in the blackout, and none of the rockets knows.",
            "plot_world flies three rockets past the rock. check_backends flies the "
            "1960 rocket on the Kalman filter and on the graph, and prints the "
            "largest difference between them. That's the number for the chat.",
        )
    )


def task_3_blackout(deck: Deck) -> None:
    """The third route frame, the blackout before the rock."""
    page = route_page(
        deck,
        "Why do the curious rockets detour before the rock?",
        "task 3",
        "route_3_blackout.png",
    )
    route_strip(
        page,
        caption="Seed 3. Darker green shows higher sensor noise "
        "for the rocket with k = 0.08.",
    )
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, RIGHT - ROUTE_COLUMN, 5.0,
        [
            "The 1960 rocket flies with the same sensor noise everywhere. Its "
            "dashed path runs almost straight to the pad.",
            "For the two curious rockets, readings are more reliable near the "
            "beacon. Their planners take that reliability into account, and both "
            "routes bend towards the beacon on the way to the pad.",
        ],
        size=20, space_after=18,
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Task 3', 2)}. Kev. Return to the prediction from "
            "the briefing. Trace the dashed line first. The 1960 rocket takes the "
            "direct route in a world where its sensor has one noise level.",
            "Now trace the two coloured lines. Both pass near the beacon in this "
            "run. As in Task 2, each planner scores the readings it expects "
            "after a move. Readings near the beacon are more reliable, and the "
            "centre of the blackout is particularly noisy. That helps explain "
            "the detours we see here.",
            "Now let us put the rock in the blackout and see what happens to "
            "those same flights.",
        )
    )


def task_3_rock(deck: Deck) -> None:
    """Show the same flights with the rock."""
    page = route_page(
        deck, "What changes when we add the rock?", "task 3", "route_4_rock.png"
    )
    route_strip(page, caption="Reference flights, seed 3")
    width = RIGHT - ROUTE_COLUMN
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, width, 4.8,
        [
            "The 1960 rocket follows the direct route until it hits the rock "
            "on step 6.",
            "In this run, both curious rockets follow the same curves as before "
            "and clear the rock.",
        ],
        size=20, space_after=18,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. Follow the dashed line to the star. The 1960 rocket hits the rock "
            "on step 6. Trace the two coloured routes past the rock to the pad. "
            "These are the reference flights for seed 3.",
            "Look back at the previous plot. Both rockets had "
            "already taken the detour before we added the rock. Those routes "
            "happened to carry them clear of it on this run.",
            "Next, compare ten flights with the rock in two different places.",
        )
    )


def task_3_meaning(deck: Deck) -> None:
    """Compare the collision counts for the same ten seeds in two rock locations."""
    page = deck.page("What happens if we move the rock into clear sky?", "task 3")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.75,
        ["We repeated the same ten flights for each rocket, changing only the "
         "position of the rock."],
        size=21,
    )  # fmt: skip
    page.table(
        MARGIN,
        2.5,
        (4.55, 3.8, FULL_WIDTH - 8.35),
        2.8,
        (
            ("Collisions in 10 flights", "Rock in the blackout", "Rock in clear sky"),
            ("1960 rocket", "10", "0"),
            ("Curious rocket, k = 0.08", "2", "6"),
            ("Curious rocket, k = 0.20", "3", "5"),
        ),
    )
    page.text(
        MARGIN, 5.85, FULL_WIDTH, 1.0,
        ["Each rocket followed its original path until a collision ended its "
         "flight. Moving the rock changed which paths crossed it."],
        size=21,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. Follow the rocket with k = 0.08. It hit the rock on two flights "
            "in the blackout and six when we moved it into clear sky. We used the "
            "same sensor noise map both times. Moving the rock placed it in the "
            "path of more of those flights."
        )
    )


def backends(deck: Deck) -> None:
    """Show the difference participants measured with the two inference methods."""
    page = deck.page("Do the two methods agree about where the rocket is?", "task 3")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.9,
        ["We ran the 1960 rocket model using a Kalman filter, then ran the same "
         "model as a graph with one node."],
        size=22,
    )  # fmt: skip
    page.text(
        MARGIN, 2.65, FULL_WIDTH, 0.7,
        ["At every step, we compared the estimated position and its uncertainty."],
        size=22,
    )  # fmt: skip
    model = model_with()
    report = captured(check_backends, KalmanBackend(model), one_node(model))
    page.text(
        MARGIN, 3.75, FULL_WIDTH, 0.4, ["Reference run"],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.printed(MARGIN, 4.25, FULL_WIDTH, 0.85, report.splitlines()[-1], size=24)
    page.text(
        MARGIN, 5.8, FULL_WIDTH, 0.95,
        ["The two methods agreed throughout the flight to within numerical rounding."],
        size=22, colour=AMBER, bold=True,
    )  # fmt: skip
    page.notes(
        notes(
            "Keyboard. This is the number you shared in Task 3. With one node, "
            "the graph should reproduce the familiar Kalman filter result. That "
            "gives us a case we can check before we use the graph to connect "
            "several variables."
        )
    )


# --- why it matters ---------------------------------------------------------------


def coda(deck: Deck) -> None:
    """Compare the expected and visited counts from the notebook's certificates."""
    planner = after_session_3()["planner"]
    if not isinstance(planner, RecedingHorizonSelector) or planner.certificate is None:
        raise ValueError("The workshop flight did not produce a search certificate.")
    searches = (
        ("One step ahead, with nine possible moves", planner.certificate),
        (
            "Seven steps ahead, with five possible moves at each step",
            axis_certificate(),
        ),
    )
    rows = [("Search", "Possible plans", "Plans scored")]
    for label, certificate in searches:
        if certificate.warrant.value != "PROVED":
            raise ValueError(
                f"The search did not prove complete coverage: {certificate}"
            )
        rows.append((label, f"{certificate.expected:,}", f"{certificate.visited:,}"))

    page = deck.page("What does PROVED mean here?", "why it matters")
    page.table(MARGIN, 1.8, (6.65, 2.75, FULL_WIDTH - 9.4), 3.3, rows)
    page.text(
        MARGIN, 5.75, FULL_WIDTH, 1.0,
        ["In both cases, the planner scored every plan in the set it was given."],
        size=22, colour=AMBER, bold=True,
    )  # fmt: skip
    page.notes(
        notes(
            f"Planned at {clock('Why it matters')}. Kev. Run the final notebook cell.",
            "A moment ago, we checked whether two ways of running the same model "
            "gave the same estimates. Here we checked whether the planner "
            "considered every option we gave it. To judge how well the model "
            "describes the world, we would need to compare its predictions with "
            "observations.",
        )
    )


def where_cpomdp_sits(deck: Deck) -> None:
    """Three uses of cpomdp that participants saw in the workshop."""
    page = deck.page("What did cpomdp let us do?", "why it matters")
    certificate = axis_certificate()
    card_row(
        page, 1.95, 3.95,
        (
            headed(
                "Change what the rocket could learn",
                "We made sensor noise depend on position. Moving towards the "
                "beacon could give the rocket a more informative reading.",
                colour=AMBER,
            ),
            headed(
                "Check the position estimates",
                "We ran the same rocket model as a Kalman filter and as a graph "
                "with one node. The two methods agreed on the rocket’s position "
                "and its uncertainty.",
                colour=GREEN,
            ),
            headed(
                "Check which plans the rocket considered",
                "In the seven-step search, the certificate showed that the planner "
                f"scored all {certificate.visited:,} possible plans in its set.",
                colour=BLUE,
            ),
        ),
        size=19,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. If you know pymdp, it models discrete states in Python. "
            "We used cpomdp for continuous states, such as the position of the "
            "rocket. It runs on JAX in Python. RxInfer uses message passing on "
            "factor graphs for Bayesian inference in Julia.",
            "The certificate gives us the counts. The label PROVED tells us why "
            "we can say the search was complete. The planner scored every plan "
            "in the set. Next, let us look at what people have built with these "
            "tools.",
        )
    )


def research(deck: Deck) -> None:
    """Three published modelling examples beyond the rocket."""
    page = deck.page("What else can we model with cpomdp?", "why it matters")
    card_row(
        page, 1.95, 4.15,
        (
            headed(
                "Finding a reward in a maze",
                "The agent detours to read a cue, then heads for the reward. "
                "With fixed sensor noise, it stays in the wrong arm of the maze.",
                colour=AMBER,
            ),
            headed(
                "Inferring a hidden signal in a cell",
                [Run("In a model of "), Run("E. coli", italic=True),
                 Run(", we can estimate the activity of a protein called CheA "
                     "from the signals it sends through the cell.")],
                colour=GREEN,
            ),
            headed(
                "Choosing a flow meter for a compressor",
                "In a simulation, changing only the flow meter led the controller "
                "to settle on opposite sides of the surge line, the boundary where "
                "flow becomes unstable.",
                colour=BLUE,
            ),
        ),
        size=19,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. The two flow meters were equally accurate at the design flow. "
            "As flow fell, one gave clearer readings and the other gave noisier "
            "ones. That changed where the controller settled. It is the same "
            "idea we saw with the rocket. Dan’s preprint explains when an action "
            "can change how much the agent expects to learn from its next reading."
        ),
        sources=(
            (
                "Maze and cell models",
                "https://github.com/inferogenesis/cpomdp/blob/main/examples/ffg/README.md",
            ),
            (
                "Compressor simulation",
                "https://github.com/inferogenesis/cpomdp/blob/main/docs/guides/surge_margin.md",
            ),
            ("Dan’s preprint", "https://arxiv.org/abs/2607.20306"),
        ),
    )


def programmes(deck: Deck) -> None:
    """The research programmes proposed on top of cpomdp."""
    page = deck.page("Where the research goes next", "why it matters")
    card_row(
        page, CONTENT_TOP, 4.1,
        (
            headed(
                "p* · Certifiable active inference",
                [Run("Active", colour=GREEN, italic=True)],
                "The rocket might assume its sensor is more accurate than it really "
                "is. Or it might model the sensor correctly but use a shortcut to "
                "estimate its position. Can we measure the free energy cost of each "
                "source of error separately?",
                colour=AMBER,
            ),
            headed(
                "Seam Networks",
                [Run("Planned", colour=AMBER, italic=True)],
                "Imagine two active inference agents searching the same patch for "
                "food. When one eats, it changes what the other finds on its next "
                "visit. How does their behaviour change over time when they can "
                "influence each other only through the world they share?",
            ),
        ),
        size=19,
    )  # fmt: skip
    page.text(
        MARGIN, 6.05, FULL_WIDTH, 0.7,
        [[Run("p* and its gates  "), linked(f"{SITE}programmes/p-star/", size=15)]],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. With p-star, we want to go a step beyond estimating those two "
            "costs and put bounds on how large each one can be."
        )
    )


# --- close ------------------------------------------------------------------------


def keep_going(deck: Deck) -> None:
    """Two optional experiments in Colab and their follow-up resources."""
    page = deck.page("Two more experiments to try in Colab", "questions")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.65,
        [[Run("Open the "), Run("Self-guided", bold=True),
          Run(" section of the notebook, below Checkpoint 3.")]],
        size=22,
    )  # fmt: skip
    card_row(
        page, 2.5, 3.65,
        (
            headed(
                "Let the rocket look two moves ahead",
                "Fly the same rockets again. This time, each rocket considers 81 "
                "pairs of moves before choosing its next move. Compare the flight "
                "paths with the ones you saw in Task 3.",
                colour=AMBER,
            ),
            headed(
                "Give the rocket a probe to tow",
                "The probe takes readings as it follows the rocket. How reliable "
                "those readings are depends on where the probe is. Run the example "
                "to see why a single fixed Kalman filter cannot reproduce this model.",
                colour=GREEN,
            ),
        ),
        size=19,
    )  # fmt: skip
    page.notes(
        "You can try either exercise in your own time. Within 48 hours, we will "
        "share the recording and workshop materials in the repository and in "
        "the cpomdp channel on the Active Inference Institute Discord.",
        sources=(
            ("Open the notebook in Colab", COLAB),
            ("Workshop materials", REPOSITORY),
            ("cpomdp on PyPI", PYPI),
            ("Inferogenesis", SITE),
            (
                "Inferogenesis (2026). cpomdp (v0.5.1) [Software]. Zenodo. "
                "10.5281/zenodo.23823455",
                "https://doi.org/10.5281/zenodo.23823455",
            ),
        ),
    )


def thanks(deck: Deck) -> None:
    """Open questions, with both presenters' emails and the two QR codes."""
    page = deck.page()
    page.mark(MARGIN, 0.55, 0.85)
    side, gap, top = 1.9, 0.4, 1.75  # each QR code, the clear space between, its top
    codes = RIGHT - 2 * side - gap  # where the QR codes start
    column = codes - gap - MARGIN  # the width the words take, left of the codes
    page.text(MARGIN, 1.6, column, 1.2, ["Any questions?"], size=60, bold=True)
    page.text(
        MARGIN, 2.95, column, 0.95,
        ["Type your question in the chat, or just ask out loud."],
        size=24, colour=MUTED,
    )  # fmt: skip
    emails = ("dan@inferogenesis.com", "kevin@kevinbackhouse.com")
    page.text(
        MARGIN, 4.45, column, 1.4,
        [[Run(email, link=f"mailto:{email}")] for email in emails],
        size=22, space_after=8,
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
    page.notes("Thank everyone for joining and leave this slide up during questions.")


SLIDES: tuple[Callable[[Deck], None], ...] = (
    title,
    who_we_are,
    agenda,
    how_we_work,
    setup,
    preview,
    task_1_brief,
    task_1_timer,
    task_1_result,
    task_2_brief,
    task_2_timer,
    task_2_result,
    task_2_meaning,
    task_3_brief,
    task_3_timer,
    task_3_blackout,
    task_3_rock,
    task_3_meaning,
    backends,
    coda,
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
