"""Build the workshop deck from the package, the lesson and the figures.

    uv run --no-sync python scripts/render_assets.py   # the figures, first
    uv run --no-sync python scripts/build_deck.py

The two hours run as an introduction, three tasks and a closing section on why cpomdp
matters, then time left open for questions. Each task is a brief, a slide that stays
up with a timer's bar while the room works, and a discussion of what the room built.
The bar runs the fifteen minutes of coding. Task 2 has a five-minute break between its
brief and its coding, so its bar runs twenty minutes and covers both. The room reads
the code on its own screen, so no slide repeats it. ``AGENDA`` holds the plan on
5-minute steps from ``START``, and every clock time on the agenda slide and in the
notes is read from it, in UTC.

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

import numpy as np
from cpomdp import KalmanBackend
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

import timer_clip
from cpomdp_workshop import (
    BEACON,
    PAD,
    WORKSHOP_CPOMDP_VERSION,
    after_session_3,
    axis_certificate,
    check_backends,
    model_with,
    one_node,
    probe,
    show_probe,
)
from cpomdp_workshop._seams import RecedingHorizonSelector
from cpomdp_workshop.plot import (
    BALANCE_GAIN,
    CURIOSITY_OFF,
    CURIOSITY_ON,
    curiosity_pair,
    pad_pair,
)
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
STARTING_GAIN = 0.08  # k, where task 2 starts
PARTING_GAIN = 0.4  # k, where task 2's two rockets part

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
        kind: ``"talk"`` when we present, ``"code"`` when the room works,
            ``"break"`` for a pause and ``"open"`` for the time left for questions.
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

    def minutes_before(self, part: str) -> int:
        """How far into the block the part named ``part`` starts."""
        names = [each.name for each in self.parts]
        if part not in names:
            raise ValueError(f"no part called {part!r} in {self.label}")
        return sum(earlier.minutes for earlier in self.parts[: names.index(part)])

    @property
    def timed(self) -> tuple[Part, ...]:
        """The parts its timer slide stays up for: the coding, and a break before it."""
        return tuple(part for part in self.parts if part.kind in ("break", "code"))

    @property
    def timer_minutes(self) -> int:
        """How long its timer runs, which is zero for a block with nothing to time."""
        return sum(part.minutes for part in self.timed)


TASK_MINUTES = 15
"""How long the room codes on each task."""

BRIEF = Part("brief", "talk", 5)
CODING = Part("you code", "code", TASK_MINUTES)
DISCUSSION = Part("discussion", "talk", 10)
BREAK = Part("break", "break", 5)

TASK_PARTS = (BRIEF, CODING, DISCUSSION)
"""Every task's parts. Task 2 adds the break."""

AGENDA = (
    Block("Welcome", (Part("introductions and setup", "talk", 10),)),
    Block("Task 1", TASK_PARTS, "the 1960 rocket"),
    Block("Task 2", (BRIEF, BREAK, CODING, DISCUSSION), "the beacon"),
    Block("Task 3", TASK_PARTS, "the hidden pad"),
    Block("Why it matters", (Part("what cpomdp adds, and what's next", "talk", 10),)),
    Block("Questions", (Part("questions", "open", 5),)),
)
"""The two hours on 5-minute steps: 65 minutes of talk, 45 of code, a 5-minute break
and 5 left open."""

TIMER_MINUTES = tuple(
    sorted({block.timer_minutes for block in AGENDA if block.timer_minutes})
)
"""Each length a timer runs for, shortest first, with a clock clip in ``assets/``."""

START = 17 * 60
"""When the session starts, in minutes after midnight UTC."""

QUESTIONS_FORM = "https://docs.google.com/forms/d/e/1FAIpQLSf7mgwxYEFpnDAkqaYolxWzP1sesG6L5Rf_5pVswE3UwtkcwA/viewform"
"""The question form. Anonymous unless the asker leaves an email for a reply."""


def block_called(label: str) -> Block:
    """The agenda's block called ``label``."""
    for block in AGENDA:
        if block.label == label:
            return block
    raise ValueError(f"no block called {label!r} in the agenda")


def clock(label: str, part: str = "", *, after: int = 0) -> str:
    """When the block called ``label`` starts, as UTC ``hh:mm``.

    ``part`` names one of the block's parts, and the time is then that part's start.
    ``after`` adds minutes, so ``after=TASK_MINUTES`` on a coding part is its end.
    """
    elapsed = START + after
    for block in AGENDA:
        if block.label == label:
            if part:
                elapsed += block.minutes_before(part)
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
    Presenter("Dan", "presenter_1.jpg", PRESENTER_1_BIO),
    Presenter("Kevin", "presenter_2.jpg", PRESENTER_2_BIO),
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
    minutes = block_called(f"Task {number}").timer_minutes
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
        MARGIN, 4.3, FULL_WIDTH, minutes,
        then=[
            [
                Run("Time's up. ", colour=AMBER),
                Run(f"Run checkpoint {number}, even if everything worked."),
            ]
        ],
        clip=timer_clip.clip(minutes), poster=timer_clip.poster(minutes),
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


def timer_notes(number: int, speaker: str, *paragraphs: str) -> str:
    """A timer slide's notes: who and when, ``paragraphs``, then how the bar runs.

    Spoken lines are plain paragraphs. Stage directions sit in square brackets.
    """
    task = block_called(f"Task {number}")
    first, minutes = task.timed[0].name, task.timer_minutes
    return notes(
        f"[{speaker}. Planned at {clock(task.label, first)}, until "
        f"{clock(task.label, first, after=minutes)}.]",
        *paragraphs,
        f"[Leave this slide up for the full {minutes} minutes. The bar and the "
        "clock start on their own, and both restart if you leave the slide, so show "
        "the notebook from another window if you need to.]",
        f"[Paste the question form in the chat again: {QUESTIONS_FORM}. Whoever "
        "isn't presenting watches the chat and the form. We don't debug anyone's "
        "code from the front. Point people to the hints, the reference cell and "
        "workshop.symptoms().]",
        f"[When the bar fills, read out the line under it. Checkpoint {number} puts "
        "everyone back on the same starting point.]",
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
            ("3  the hidden pad", BLUE),
        )
    ):
        page.pill(MARGIN + index * 4.1, 5.9, 3.8, 0.7, label, colour=colour, size=20)
    page.notes(
        notes(
            f"[Kev. Planned at {clock('Welcome')} UTC. Dan shares the screen with this "
            "slide up and pastes the Colab link and the question form into the chat.]",
            "Welcome, everyone. Over the next two hours we're going to build three "
            "rockets and watch how each one decides where to fly.",
            "If you can, open the Colab notebook now. The link is in the chat, and "
            "the next few slides walk you through it. The install takes a minute or "
            "two, so it's worth starting it while we introduce ourselves.",
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
            "[Dan, then Kev introduces himself.]",
            "Hi all, a quick bit of background. I'm a research software engineer by "
            "trade. Most of my work is large-scale architecture and tool "
            "orchestration for machine learning models.",
            "I first came across active inference in late-night conversations with "
            "Kev here, and the two of us started playing around with pymdp. I'm "
            "interested in all things science, and two of my favourites are "
            "microbiology and abiogenesis, the origin-of-life stuff.",
            "Then I came across a paper by Henry Mattingly and colleagues, called "
            "Escherichia (esh-uh-RIK-ee-uh) coli chemotaxis is information limited. "
            "They measured how much information a bacterium gets from its "
            "surroundings as it swims, and it's a tiny amount. I began to wonder "
            "whether an active inference model could reproduce those numbers.",
            "The problem was that pymdp works in discrete states, so you'd have to "
            "bin a continuous world, and the binning errors would more than likely "
            "dwarf the information rates. It ruins any hope of getting those values. "
            "So I started work on cpomdp, so that we, the community, could "
            "investigate continuous-state problems, like most of biology, in Python, "
            "arguably the most popular research language.",
            "It started life as Python calling RxInfer in Julia through a bridge. "
            "The bridge introduced too much latency, which practically undid what "
            "makes RxInfer's message passing so good. So I took a page out of "
            "pymdp's book, rewrote it in Python on JAX, and made it public at the "
            "same time. It's still in very early development, but it's now at a "
            "point where I'm happy to share it with you and to take feedback, "
            "feature requests, that sort of thing.",
            "Dan (aloud)\nSome of the source code in "
            f"cpomdp {WORKSHOP_CPOMDP_VERSION} was written by Claude Code and "
            "reviewed before release.",
            "[Reference: Mattingly, H. H., Kamino, K., Machta, B. B. and Emonet, T. "
            "(2021). Escherichia coli chemotaxis is information limited. Nature "
            "Physics.]",
        )
    )


def agenda(deck: Deck) -> None:
    """The two hours, one row per block, each part named and as wide as its minutes.

    Each row's heading carries its UTC times. A part in a block of several carries its
    minutes under its name, so a task shows its brief, code and discussion lengths,
    and Task 2 shows its break.
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
            "All times UTC. While you code, a timer stays on the screen. Task 2's "
            "timer includes the break."
        ],
        size=19, colour=MUTED, space_after=0,
    )  # fmt: skip
    page.notes(
        notes(
            "[Dan.]",
            "Here's the plan. Every task runs the same way. We talk for about five "
            f"minutes, you get {TASK_MINUTES} minutes to code, and then we spend ten "
            "minutes on what you found.",
            "There are three tasks. The first is the classic 1960 controller, the "
            "second adds a beacon, and in the third the rocket has to find a landing "
            "pad it can't see. After that we'll take ten minutes on what cpomdp adds "
            "and where it's going, and leave the last five for questions.",
            "We want your hands on the keyboard by quarter past, so we'll keep the "
            "introductions short. There is one five-minute break, at "
            f"{clock('Task 2', BREAK.name)}, after the Task 2 brief and before you "
            "start coding. Task 2's timer runs for twenty minutes so that it "
            "includes the break. If you need another one, the coding time is a good "
            "moment, because the checkpoint cells mean you can always catch up.",
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
            "[Kev.]",
            "In each task you'll fill in two lines, marked with three dots. We've "
            "written everything else, so the Python stays out of the way of the "
            "ideas. A few of the bonus exercises ask you to write a bit more "
            "yourself.",
            "Below each code cell there are two hints and a worked solution, folded "
            "away. A hint gives you a nudge first, and the line itself is behind Show "
            "code. Open them whenever you like. This isn't a coding test. We'd "
            "suggest not handing the task to an AI assistant, because you'll get "
            "more out of working the lines out yourself, but it's your call.",
            "When a task ends, run its checkpoint cell, even if everything worked. It "
            "resets the names the next task relies on, so we all start the next task "
            "from the same place.",
            "Each task ends with one number to paste into the chat. It gives us a "
            "quick read of how the room is doing, and a thumbs up is fine too. If you "
            "don't finish, don't worry. You can save a copy and come back to it "
            "later, and Dan will show you how on the next slide.",
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
            "[Dan.]",
            "We've tried to make this work whatever your level of programming. Both "
            "links on the slide are already in the Zoom chat.",
            "If you've set up your own Python, great. If not, use Colab. Either way "
            "I'd suggest saving a copy, with File --> Save a copy in Drive. That way "
            "you won't "
            "lose your work if the tab closes, and you can carry on after the "
            "session. Your copy is private to you. Nobody else can see your code, so "
            "don't worry about making mistakes.",
            "The first time you run a cell, Colab warns you that the notebook wasn't "
            "written by Google. Click Run anyway. Nothing in the workshop asks for "
            "access to your Drive.",
            "The second link is an anonymous question form. Asking a question in "
            "front of a full room can be daunting, and the form sends us the "
            "question and nothing else. If it needs a longer answer, leave an email "
            "and we'll get back to you after the session.",
            f"[The two links to paste: {COLAB} and {QUESTIONS_FORM}]",
        )
    )


def preview(deck: Deck) -> None:
    """Cell 2's picture, the whole game first."""
    page = route_page(deck, "What we will build", "welcome", "route_3_pad.png")
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, RIGHT - ROUTE_COLUMN, 4.7,
        [
            "By the end of Task 3, we will have built two rockets that differ only "
            "in curiosity.",
            "Neither knows where the landing pad is. One of them finds it.",
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
            "[Kev.]",
            "Please run the first two code cells now. The first installs the "
            "workshop, and the second draws this picture.",
            "It's where we'll end up by the end of the session: two rockets that "
            "differ only in curiosity, looking for a landing pad neither of them can "
            "see. One finds it, one doesn't. By Task 3 you'll know why.",
            "If the picture on your screen matches this one, give us a thumbs up in "
            "the chat. If you get an error instead, tell us in the chat and we'll "
            "help you get set up.",
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
        page, 3.2, "Can you spot the problem?", height=0.55,
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
            f"[Kev. Planned at {clock('Task 1')}.]",
            "NASA used a Kalman filter on Apollo, because every measurement had some "
            "uncertainty in it. The filter blends where the spacecraft should be, "
            "going by its last estimate, with what the sensors actually read. The "
            "more reliable the reading, the more weight it gets.",
            "In Karl Friston's account of active inference, we update our beliefs by "
            "minimising free energy. In a linear-Gaussian model like this one, that "
            "gives exactly the same update as the Kalman filter.",
            "The linear quadratic regulator, LQR for short, then uses that estimate "
            "to decide how hard to fire the thrusters. It weighs being off target "
            "against the cost of firing harder.",
            "If you know pymdp, the cards along the bottom map its A, B, C and D "
            "onto the names cpomdp uses. The preference, C, doesn't appear until "
            "Task 2.",
            "Keep the question on the slide in mind while you code: does it matter, "
            "for what the rocket learns, which way it moves?",
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
        timer_notes(
            1,
            "Kev",
            f"You've got {TASK_MINUTES} minutes. There are two lines to write: what "
            "the sensor observes, and how noisy its readings are.",
            "When you run the cell, it prints a spread. That's the gap between the "
            "most and the least informative move the rocket could make. Paste it in "
            "the chat.",
            "If you finish early, have a go at the bonuses.",
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
            f"[Dan. Planned at {clock('Task 1', DISCUSSION.name)}.]",
            "So, what does a spread of zero tell us? [Pause, and read out an answer "
            "or two from the chat.]",
            "The probe compares how much each of the nine compass moves is expected "
            "to teach the rocket. A spread of 0.000000 nats means every move teaches "
            "exactly the same amount. That's not the same as learning nothing. If you "
            "tried the bonuses, you'll have seen the amount of information change "
            "while the spread stayed at zero.",
            "My usual analogy is the optician's. You sit down, they ask you to read "
            "the letters on the chart, and if you're like me, you can't see a thing. "
            "Now imagine you get up and press your nose against the chart, and the "
            "letters are just as blurry. There's no reason to walk over. In technical "
            "terms, you have no drive to act, because what you'd learn is the same "
            "wherever you stand.",
            "Koudahl and colleagues showed this formally in 2021. In a "
            "linear-Gaussian model every action offers the same information, so "
            "active inference's epistemic drive drops out. Not ideal when your whole "
            "goal is a toolbox for linear-Gaussian active inference agents. You're "
            "now at the roadblock I hit about a month into development. Welcome.",
            f"[Reference: {KOUDAHL_REFERENCE}]",
        )
    )


# --- task 2 -----------------------------------------------------------------------


def task_2_brief(deck: Deck) -> None:
    """Why position-dependent sensor noise can make some moves more informative."""
    page = deck.page("Making information gain depend on the action", "task 2")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.85,
        [
            [Run("Most real-world sensors blur with range.", bold=True)],
            [
                Run("R", bold=True),
                Run(" is the noise in the sensor’s readings. "),
                Run("R(x)", bold=True),
                Run(" means that noise depends on the state x."),
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
                "Readings are less noisy near the beacon. Moving closer means the "
                "rocket expects to learn more.",
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
            f"[Dan. Planned at {clock('Task 2')}.]",
            "Back to the optician's. Now imagine your eyesight depends on where you "
            "stand, which is much closer to real life. Things blur with distance, and "
            "so do almost all real sensors.",
            "We write the sensor's noise as R. In Task 1, R was the same everywhere. "
            "Now we write R(x), meaning the noise depends on x, the rocket's "
            "position. Near the beacon the readings are sharp, and further away they "
            "get noisier. So where the rocket moves now changes how much it can "
            "expect to learn. That's the chain across the middle of the slide.",
            "In the code, k, called beacon_gain, sets how fast the noise grows as the "
            "rocket moves away from the beacon.",
            "At each step the planner scores the nine compass moves, and each score "
            "has two parts. The pragmatic part asks how close the next reading is "
            "likely to be to the one that says you're on the pad. It also counts how "
            "noisy that reading will be. The epistemic part is the expected "
            "information gain: how much the move should tell the rocket about where "
            "it is. That second part is what we mean by curiosity. Expected free "
            "energy is the pragmatic part minus the epistemic part, and the rocket "
            "picks the move where that comes out lowest.",
            "Two things change from Task 1. First, the goal is now a reading, the one "
            "that says the rocket is on the pad, rather than a position. Second, the "
            "rocket holds that goal loosely. A goal has a precision, and the "
            "precision scales the pragmatic part but leaves the epistemic part "
            "alone. So it works as an exchange rate between information and the "
            "pull of the pad. We've set it to 0.1, so the two parts can actually "
            "compete. We'll come back to that choice.",
            "This time we fly every rocket twice, once with curiosity on and once "
            "with it off. We built that switch for the workshop. cpomdp itself "
            "doesn't have one.",
            "[If anyone asks how exact the information numbers are:] When the rocket "
            "plans, it reads the noise at the centre of where it expects a move to "
            "take it. That's a first-order shortcut. It's least accurate when the "
            "rocket is very unsure where it is and the noise changes sharply.",
            "[The line at the bottom of the slide:] Koudahl and colleagues point to "
            "another way to bring curiosity back, multiplicative control. It's "
            "planned for cpomdp but not built yet.",
            "[Reference: Corva, D. (2026). State-Dependent Observation Noise "
            "Reintroduces Epistemic Value in Linear-Gaussian Active Inference. "
            "https://arxiv.org/abs/2607.20306]",
            f"[Further references: {KOUDAHL_REFERENCE}",
            "Franklin, D. W. et al. (2007). Endpoint Stiffness of the Arm Is "
            "Directionally Tuned to Instability in the Environment. "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6672883/]",
            "[When this brief ends, show the timer slide. Its twenty minutes are the "
            "five-minute break and then the coding, so the room is back for the "
            f"discussion at {clock('Task 2', DISCUSSION.name)}.]",
        )
    )


def task_2_timer(deck: Deck) -> None:
    """Task 2's twenty minutes: the break, then the coding."""
    page = timer_page(
        deck, 2,
        (
            f"{FILL_IN} and start with k = 0.08.",
            "Raise k until one of the rockets stops landing. Paste that k in the chat.",
            "Finished early? Try the two bonuses below the cell.",
        ),
    )  # fmt: skip
    page.notes(
        timer_notes(
            2,
            "Kev",
            "This timer runs for twenty minutes, because it includes a five-minute "
            "break. Take the break first, or whenever it suits you, and we will be "
            "back together when the bar fills.",
            "Two lines again. First, set k to 0.08. Then, in my_sensor, fill in the "
            "extra noise a reading picks up away from the beacon. The beacon's "
            "settings are in params, and the comments in the cell show how to read "
            "each one by name.",
            "Run the cell. It flies the same rocket twice, with curiosity on and "
            "with it off, and draws both routes over the beacon's noise map.",
            "Then raise k and keep an eye on both rockets. At what k does one of them "
            "stop landing? That's your number for the chat. If you're stuck, Hint 1 "
            "suggests trying 0.4.",
        )
    )


def task_2_result(deck: Deck) -> None:
    """The reference pair at the starting k, with prompts for the task 2 discussion."""
    k = STARTING_GAIN
    pair = curiosity_pair(k)
    on, off = pair[CURIOSITY_ON], pair[CURIOSITY_OFF]
    beacon = float(np.linalg.norm(on.path - np.asarray(BEACON), axis=1).min())
    page = route_page(deck, "Task 2 · Discussion", "task 2", "route_2_beacon.png")
    left, _, plot_width = ROUTE_FIGURE
    page.text(
        left, 5.99, plot_width, 0.35, [f"Reference run with k = {k:.2f}, seed 3"],
        size=16, bold=True, space_after=0,
    )  # fmt: skip
    page.text(
        left, 6.40, plot_width, 0.75,
        [f"With curiosity on, the rocket passed {beacon:.2f} from the beacon and "
         f"finished {on.distance_to(PAD):.2f} from the pad. With curiosity off, it "
         f"finished {off.distance_to(PAD):.2f} from the pad."],
        size=17, colour=MUTED,
    )  # fmt: skip
    width = RIGHT - ROUTE_COLUMN
    page.text(
        ROUTE_COLUMN, 1.85, width, 1.45,
        ["How did the two routes differ?"],
        size=21, bold=True,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 3.65, width, 2.50,
        ["At what k did the curious rocket stop landing?"],
        size=20, bold=True,
    )  # fmt: skip
    page.notes(
        notes(
            f"[Dan. Planned at {clock('Task 2', DISCUSSION.name)}.]",
            "Welcome back. You've just flown the same rocket twice, once with "
            "curiosity on and once with it off.",
            f"Here's our run at k = {k:.2f}. The amber line has curiosity on. It "
            f"flies up to the beacon, passing {beacon:.2f} from it, and still "
            f"finishes {on.distance_to(PAD):.2f} from the pad. The blue line is the "
            "same rocket with curiosity off. It flies straight and finishes "
            f"{off.distance_to(PAD):.2f} from the pad. Both count as landed, because "
            "both end inside the circle, within 0.5 of the pad.",
            "The two rockets share one model and one sensor. The only difference is "
            "the epistemic part of the score, so the detour is curiosity. On this "
            "flight, curiosity decided 6 of the 36 moves. We checked 10 seeds, and on "
            "every one the routes part by at least 1 and both rockets land.",
            "At what k did your curious rocket stop landing? [Read a few answers "
            "from the chat.]",
            "On our seed, the curious rocket stops landing at k = 0.37, and the "
            "rocket with curiosity off keeps landing up to 0.45. So depending on the "
            "steps you took, you may have seen the curious rocket miss while the "
            "other one landed, or both of them miss. Both are right answers, and the "
            "next two slides explain each one.",
        )
    )


def task_2_balance(deck: Deck) -> None:
    """Show that a bend towards the beacon is curiosity only where the pair parts."""
    page = deck.page("Curious, or just avoiding noise?", "task 2")
    height = page.picture(ASSETS / "route_2_balance.png", MARGIN, 1.3, FULL_WIDTH)
    top = 1.3 + height + 0.1
    page.text(
        MARGIN, top, FULL_WIDTH, 1.45,
        [
            "The pragmatic score counts distance to the pad and the noise along the "
            "way, and the precision scales it. The epistemic score counts "
            "information, and the precision leaves it alone.",
            f"At precision 1 and k = {BALANCE_GAIN:.1f}, both routes bend towards the "
            "beacon, yet curiosity changes at most 1 of the 36 moves on any of 10 "
            "seeds. That bend is the rocket avoiding noise.",
        ],
        size=17, space_after=6,
    )  # fmt: skip
    page.text(
        MARGIN, 6.5, FULL_WIDTH, 0.6,
        ["We picked precision 0.1 so the two scores can compete. Expected free "
         "energy has no other dial for this."],
        size=19, bold=True, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            "[Kev.]",
            "Does a bend towards the beacon always mean the rocket is curious? Not "
            "necessarily, and this slide shows why.",
            f"Both panels fly the same rocket at k = {BALANCE_GAIN:.1f}, with "
            "curiosity on and with it off. We've moved up from 0.08, because at "
            f"{BALANCE_GAIN:.1f} the beacon starts pulling even the rocket with "
            "curiosity off. On the left, the rocket holds its goal at a precision "
            "of 1. On the right, at 0.1.",
            "On the left, both routes bend towards the beacon and they nearly "
            "overlap. Across 10 seeds, curiosity changes at most 1 of the 36 moves. "
            "So the bend isn't curiosity. It comes from the pragmatic score, which "
            "counts how noisy the reading is as well as how far it is from the pad. "
            "Readings near the beacon are cleaner, so heading that way looks good "
            "even to a rocket that doesn't care about information.",
            "On the right, at 0.1, curiosity takes the rocket right up to the beacon. "
            "Across 10 seeds it decides between 6 and 9 of the 36 moves.",
            "So a route that looks curious might not be. The only way to tell which "
            "score chose a move is to fly it again with curiosity off.",
            "[If someone asks why noise is in the pragmatic score at all:] The goal "
            "is a reading rather than a position, because cpomdp "
            f"{WORKSHOP_CPOMDP_VERSION} won't accept a goal over positions when the "
            "noise depends on position. A goal over readings ends up paying for how "
            "noisy each reading is expected to be.",
            "We picked 0.1 on purpose, so the two scores could compete. Expected "
            "free energy has no separate dial for curiosity, so the precision is the "
            "only thing that trades one against the other. Turning the precision "
            "down is the same as turning curiosity up.",
        )
    )


def task_2_meaning(deck: Deck) -> None:
    """At k = 0.4 the noise varies steeply enough for curiosity to stop the landing."""
    k = PARTING_GAIN
    pair = curiosity_pair(k)
    on, off = pair[CURIOSITY_ON], pair[CURIOSITY_OFF]
    page = route_page(
        deck, "Why does the curious rocket stop short?", "task 2",
        "route_2_too_curious.png",
    )  # fmt: skip
    route_strip(page, caption=f"Reference flights at k = {k:.1f}, seed 3")
    width = RIGHT - ROUTE_COLUMN
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, width, 3.6,
        [
            f"With curiosity off, the rocket lands {off.distance_to(PAD):.2f} from "
            "the pad.",
            "With curiosity on, it stays near the beacon, where readings are less "
            f"noisy, and stops {on.distance_to(PAD):.2f} short.",
            "Over 10 seeds, no curious rocket lands, and every rocket with curiosity "
            "off does.",
        ],
        size=19, space_after=14,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 5.2, width, 1.7,
        ["The precision is still 0.1. What changed is k, how steeply the noise "
         "rises away from the beacon."],
        size=19, bold=True, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            "[Kev.]",
            "Same precision as before, 0.1, the setting where both rockets landed "
            f"two slides ago. What we've changed is k, from 0.08 to {k:.1f}, so the "
            "noise now climbs five times as steeply away from the beacon.",
            "The blue rocket, curiosity off, lands "
            f"{off.distance_to(PAD):.2f} from the pad. The amber rocket, curiosity "
            "on, goes to the beacon and won't leave. It stops "
            f"{on.distance_to(PAD):.2f} short.",
            "Why? From the beacon, every move towards the pad is a move away from "
            "clean readings, so each one teaches less. With noise this steep, the "
            "information the rocket would give up outweighs the pull of the pad. "
            "Across 10 seeds, no curious rocket lands, and they stop a median 1.67 "
            "short. Every rocket with curiosity off lands.",
            "You'll notice the blue route bends towards the beacon too. That's the "
            "last slide again: the pragmatic score likes cleaner readings. Push k "
            "higher and that pull wins as well. On this seed the rocket with "
            "curiosity off stops short from k = 0.46, and at k = 0.5 neither rocket "
            "lands on any of 10 seeds.",
            "So two settings decide whether curiosity helps. The precision sets what "
            "information is worth against the pull of the pad. k sets how much there "
            "is to gain by staying near the beacon.",
        )
    )


# --- task 3 -----------------------------------------------------------------------


def task_3_brief(deck: Deck) -> None:
    """The hidden pad, and the three readings the rocket has to find it with."""
    page = deck.page("Task 3 · Find the pad", "task 3")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.65,
        ["The rocket no longer knows where the landing pad is."],
        size=22,
    )  # fmt: skip
    card_row(
        page, 2.3, 2.3,
        (
            headed(
                "The guess",
                "The rocket starts with a guess at where the pad is, and it knows "
                "how unsure that guess is.",
            ),
            headed(
                "The radar",
                "A coarse radar points roughly from the rocket towards the pad. Its "
                "noise is the same everywhere.",
                colour=BLUE,
            ),
            headed(
                "The broadcast",
                "The beacon broadcasts the pad’s exact position. The broadcast is "
                "clear near the beacon and fades with distance.",
                colour=AMBER,
            ),
        ),
    )  # fmt: skip
    page.text(
        MARGIN, 4.85, FULL_WIDTH, 0.8,
        ["We built this world so the information the rocket needs is away from "
         "where it wants to go."],
        size=20,
    )  # fmt: skip
    question(page, 5.85, "Which rocket finds the pad?", height=0.55)
    page.notes(
        notes(
            f"[Kev. Planned at {clock('Task 3')}.]",
            "In Task 2 the rocket knew where the pad was, and curiosity only changed "
            "its route. Now we hide the pad. The rocket starts with a guess, and it "
            "knows roughly how good that guess is.",
            "It has three sensors. It reads its own position. A coarse radar points "
            "roughly from the rocket towards the pad, and it's just as rough "
            "everywhere. And the beacon broadcasts the pad's exact position, but you "
            "only hear it clearly when you're close.",
            "We should say clearly that we built this world on purpose. The radar is "
            "coarse on purpose, and the broadcast fades on purpose. The guess is "
            "right on average, because we draw the true pad from the rocket's own "
            "prior. In the Active Inference book's terms, Figure 2.2, the true pad "
            "lives in the generative process, the world, and the rocket's generative "
            "model only holds a belief about it.",
            "One more choice worth flagging. The rocket's goal is a radar reading, "
            "the one that says the pad is right here. The radar's noise doesn't "
            "change with position, so this time the pragmatic score can't pull the "
            "rocket towards the beacon the way it did in Task 2. So if a rocket goes "
            "to the beacon in this task, it's because of curiosity.",
            "Again we fly the same rocket twice, curiosity on and off. Keep the "
            "question in mind while you code: which one finds the pad?",
        )
    )


def task_3_timer(deck: Deck) -> None:
    """Task 3's fifteen minutes."""
    page = timer_page(
        deck, 3,
        (
            f"{FILL_IN} and run the cell.",
            "Lower pad_spread until the curious rocket stops finding the pad. Paste "
            "that spread in the chat.",
            "Finished early? Try the bonuses.",
        ),
    )  # fmt: skip
    page.notes(
        timer_notes(
            3,
            "Dan",
            "Two lines again. The first is in my_broadcast: how much noise the "
            "broadcast picks up with distance from the beacon. There's no cap this "
            "time. The second is prior_cov, how unsure the rocket starts about its "
            "own position and about the pad's.",
            "This cell builds the whole model, like Task 2's did. The state is now "
            "four numbers, the rocket's position and then the pad's. show_misses "
            "prints how far each rocket ended from the true pad, and plot_pad draws "
            "both routes.",
            "Then lower pad_spread and run the cell again. That only changes how "
            "unsure the rocket is about the pad. The world still places the true pad "
            "with a spread of 1.5. The spread where the curious rocket stops finding "
            "the pad is your number for the chat.",
        )
    )


def task_3_result(deck: Deck) -> None:
    """The reference pair on the hidden pad, with the ten-seed counts beside it."""
    pair = pad_pair()
    on, off = pair[CURIOSITY_ON], pair[CURIOSITY_OFF]
    page = route_page(deck, "Who found the pad?", "task 3", "route_3_pad.png")
    route_strip(
        page,
        caption="Reference flights, seed 3. The dashed ring is where the true pad "
        "is drawn from. A rocket lands if it ends inside the circle round the pad.",
    )
    width = RIGHT - ROUTE_COLUMN
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, width, 0.35, ["Over 10 seeds"],
        size=15, colour=MUTED,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 1.9, width, 2.6,
        [
            "With curiosity on, the rocket lands on 9 seeds, with a median miss of "
            "0.25.",
            "With curiosity off, it lands on 3, with a median miss of 0.61.",
        ],
        size=19, space_after=12,
    )  # fmt: skip
    page.notes(
        notes(
            f"[Kev. Planned at {clock('Task 3', DISCUSSION.name)}.]",
            "Start with the blue route, curiosity off. It heads for its guess and "
            "then on towards the pad, with only the coarse radar to correct it. It "
            f"ends {off.miss:.2f} from the true pad, outside the circle.",
            "Now amber, curiosity on. It goes up to the beacon first, hears where "
            f"the pad is, and flies there. It ends {on.miss:.2f} from the true pad. "
            "The shaded ellipse is where it believes the pad is by the end.",
            "A fair question is whether the blue rocket hears the broadcast too. It "
            "does, at every step. But out there the broadcast is faint, so its "
            "picture of the pad sharpens slowly, and nothing in its score tells it "
            "to go closer and listen.",
            "You lowered pad_spread until the curious rocket stopped finding the "
            "pad. On our seed it still finds the pad at 0.48 and stops at 0.45. A "
            "rocket that's sure of its guess expects to learn nothing at the beacon, "
            "so it doesn't go. More on that in a moment.",
            "Across 10 seeds, the curious rocket lands on 9 and the one with "
            "curiosity off on 3. The median misses are 0.25 and 0.61. On the map "
            "that gap looks small, which is why we print the misses in large type.",
            "Curiosity pays off here because what the rocket learns changes where it "
            "goes. There's a question on this under Checkpoint 3 in the notebook.",
        )
    )


def task_3_tuning(deck: Deck) -> None:
    """How narrow the window is in which curiosity pays, and where that might ease."""
    page = deck.page("The balance takes tuning", "task 3")
    page.text(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 0.55,
        ["The curious rocket over 10 seeds, as we change one setting at a time. "
         "With curiosity off, it lands on 3."],
        size=20,
    )  # fmt: skip
    page.table(
        MARGIN,
        2.15,
        (4.6, 2.6, FULL_WIDTH - 7.2),
        2.75,
        (
            ("Setting", "Curious lands", "Visits the beacon"),
            ("As built: precision 0.1, spread 1.5", "9", "9"),
            ("Precision 0.3", "7", "6"),
            ("Precision 1", "3", "1"),
            ("Spread 0.1, an overconfident rocket", "1", "0"),
        ),
    )
    page.text(
        MARGIN, 5.05, FULL_WIDTH, 0.6,
        ["With a broadcast that carries far, both rockets land on all 10 seeds, "
         "and curiosity buys nothing."],
        size=19,
    )  # fmt: skip
    page.text(
        MARGIN, 5.8, FULL_WIDTH, 1.2,
        ["Where we think this goes: longer horizons and factor-graph backends "
         "should make the balance less fragile. One data point so far: looking two "
         "moves ahead, the curious rocket lands on all 10 seeds."],
        size=19, bold=True, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            "[Dan.]",
            "Task 3 works, but only inside a window, and we want to be honest about "
            "how narrow that window is.",
            "Precision first. At 0.1 the curious rocket lands on 9 of 10 seeds. At "
            "0.3 it lands on 7 and visits the beacon on 6. At 1 it lands on 3, the "
            "same as with curiosity off, and visits the beacon on only 1 seed. On "
            "that seed the pad happens to lie that way, and the rocket with "
            "curiosity off passes the beacon too.",
            "Then the rocket's own uncertainty. At a spread of 0.1 it's "
            "overconfident. It thinks there's nothing to learn, never visits the "
            "beacon, and lands on 1 seed of 10.",
            "Then the world. If the broadcast carries far, both rockets hear the pad "
            "from wherever they are, and both land on all 10 seeds. Curiosity buys "
            "nothing. A sharp radar does the same, which is why ours is coarse. "
            "Bonus 2 lets you try both.",
            "So getting the two scores to balance takes tuning. Where do we expect "
            "that to get easier? This part is a direction, not a result. We expect "
            "longer planning horizons and cpomdp's factor-graph backends to make the "
            "balance less fragile, and the p* programme is where we plan to test "
            "that.",
            "We have one data point so far. Looking two moves ahead, the curious "
            "rocket lands on all 10 seeds instead of 9. We have no factor-graph "
            "evidence in this workshop. The one-node check on the next slide shows "
            "the graph engine matches the Kalman filter. It doesn't show that graphs "
            "make curiosity more stable.",
        )
    )


def backends(deck: Deck) -> None:
    """Show the difference participants measured with the two inference methods."""
    page = deck.page(
        "Do the two methods agree about where the rocket is?", "why it matters"
    )
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
            f"[Dan. Planned at {clock('Why it matters')}. Run the final notebook "
            "cell now, and ask the room to run it too.]",
            "The final cell runs two checks on what we did today. This is the first. "
            "We took the 1960 rocket and ran it two ways: through a Kalman filter, "
            "and as a factor graph with a single node. At every step we compared "
            "where each one thought the rocket was, and how sure it was.",
            "The largest difference over the whole flight is down at the level of "
            "rounding error in the computer's arithmetic. With one node, the graph "
            "should give exactly the Kalman filter's answer. That makes it a case we "
            "can check before we trust the graph with models that connect several "
            "variables.",
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
        ["PROVED means the planner scored every plan in its set. It doesn't tell us "
         "the model is right."],
        size=22, colour=AMBER, bold=True,
    )  # fmt: skip
    page.notes(
        notes(
            "[Kev.]",
            "The second check is the planner's certificate. The table shows two "
            "searches. In the first, the curious rocket looks one move ahead, with "
            "nine moves to choose from. In the second, it looks seven moves ahead "
            "with five moves at each step. That's 78,125 possible plans.",
            "PROVED means the planner scored every plan in the set it was given. "
            "Nothing was sampled, and nothing was skipped.",
            "What it doesn't tell us is whether the model is any good. The last "
            "slide showed that two ways of running the model agree, and this one "
            "shows the planner looked at every option. Neither says the model "
            "describes the world well. For that, you'd have to compare its "
            "predictions with real observations.",
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
                "We made sensor noise depend on position, then made the pad's "
                "position something the rocket had to learn. That gave moving a "
                "reason to teach it something.",
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
            "[Kev.]",
            "So what did cpomdp actually let us do today?",
            "First, it let us change what the rocket could learn. We made the "
            "sensor's noise depend on position, and in Task 3 we made the pad part "
            "of what the rocket had to work out. That's what gave curiosity "
            "something to do.",
            "Second, we checked the position estimates, by running the same model "
            "as a Kalman filter and as a one-node graph and seeing that they agree. "
            "Third, we checked the planning. The certificate shows the planner "
            "scored every plan in the seven-step search.",
            "For context: if you know pymdp, that's the library for discrete states "
            "in Python. cpomdp does the same job for continuous states, like the "
            "rocket's position, and runs on JAX. RxInfer, in Julia, does Bayesian "
            "inference by message passing on factor graphs.",
            "Next, a few other things people have modelled with these tools.",
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
            "[Kev.]",
            "In the maze, the agent takes a detour to read a cue before heading for "
            "the reward. With fixed sensor noise it never takes the detour, and it "
            "ends up stuck in the wrong arm. That's Task 1's lesson again.",
            "In the cell model, we estimate how active a protein called CheA is "
            "inside E. coli, from the signals it passes on through the cell. That "
            "links back to where cpomdp started.",
            "In the compressor simulation, the two flow meters were equally accurate "
            "at the design flow. As the flow dropped, one gave cleaner readings and "
            "the other noisier ones, and that alone changed which side of the surge "
            "line the controller settled on. It's the same idea as the rocket: the "
            "sensor changes which moves are worth making. Dan's preprint sets out "
            "when an action can change how much an agent expects to learn from its "
            "next reading.",
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
            "[Kev.]",
            "Two research programmes build on cpomdp. p*, which is under way now, "
            "is about errors. A rocket might think its sensor is better than it is. "
            "Or it might have the sensor right, but use a shortcut to work out where "
            "it is, like the first-order shortcut we mentioned in Task 2. Each "
            "mistake costs something in free energy. Can we measure those two costs "
            "separately? And beyond measuring them, we want to put bounds on how "
            "large each one can be.",
            "Seam Networks is planned. Picture two agents foraging in the same "
            "patch. When one eats, it changes what the other finds next time. They "
            "never talk to each other, only through the world they share. How does "
            "their behaviour change over time?",
            "The link at the bottom has more on p*.",
        )
    )


# --- close ------------------------------------------------------------------------


def keep_going(deck: Deck) -> None:
    """Three optional experiments in Colab and their follow-up resources."""
    page = deck.page("Three more experiments to try in Colab", "questions")
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
                "Fly Task 3’s two rockets again. This time, each considers 81 pairs "
                "of moves before choosing its next move. Does the rocket with "
                "curiosity off go to the beacon now?",
                colour=AMBER,
            ),
            headed(
                "Give the rocket a probe to tow",
                "The probe takes readings as it follows the rocket. Run the example "
                "to see why a single fixed Kalman filter cannot reproduce this "
                "model.",
                colour=GREEN,
            ),
            headed(
                "Whose detour is it?",
                "Put a patch of very noisy readings on the straight route and fly "
                "the rocket with curiosity on and off. The cell counts the moves "
                "that curiosity decided.",
                colour=BLUE,
            ),
        ),
        size=18,
    )  # fmt: skip
    page.notes(
        notes(
            "[Dan.]",
            "If you want to keep going, there are three more experiments in the "
            "Self-guided section of the notebook, below Checkpoint 3. One lets the "
            "rocket look two moves ahead. One gives it a probe to tow, which makes a "
            "model a single fixed Kalman filter can't reproduce. And one puts a "
            "patch of very noisy readings on the straight route, so you can see "
            "whose detour it really is.",
            "Within 48 hours we'll share the recording and the materials in the "
            "repository, and in the cpomdp channel on the Active Inference Institute "
            "Discord.",
        ),
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
    page.notes(
        notes(
            "[Dan and Kev. Leave this slide up during questions.]",
            "Thank you all for coming. We'll stay on for questions now, so type them "
            "in the chat or just ask. If you'd rather ask later, both our emails are "
            "on the slide.",
        )
    )


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
    task_2_balance,
    task_2_meaning,
    task_3_brief,
    task_3_timer,
    task_3_result,
    task_3_tuning,
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
