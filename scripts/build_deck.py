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
from cpomdp import Warrant
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

import timer_clip
from cpomdp_workshop import (
    BEACON,
    PAD,
    WORKSHOP_CPOMDP_VERSION,
    after_session_3,
    axis_certificate,
    model_with,
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
CPOMDP_REPOSITORY = "https://github.com/inferogenesis/cpomdp"
CPOMDP_ISSUES = f"{CPOMDP_REPOSITORY}/issues"
CPOMDP_DISCUSSIONS = f"{CPOMDP_REPOSITORY}/discussions"
WARRANTLIB = "https://pypi.org/project/warrantlib/"
WARRANTLIB_DOCS = "https://cpomdp.inferogenesis.com/api/warrant/"
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


@dataclass(frozen=True)
class Feature:
    """Something the pinned cpomdp does that the notebook never uses.

    Attributes:
        title: what it does, short enough for one line of a card.
        names: the public cpomdp names that provide it.
        argument: the argument of the first name that turns the feature on, for a
            name the notebook already shows. Empty when the names are unused.
    """

    title: str
    names: tuple[str, ...]
    argument: str = ""

    @property
    def lookup(self) -> str:
        """What to look up in the cpomdp documentation to find the feature."""
        if self.argument:
            return f"the {self.argument} argument of {self.names[0]}"
        return " with ".join(self.names)


@dataclass(frozen=True)
class FeatureGroup:
    """One card of features that belong together.

    Attributes:
        heading: what the features have in common.
        colour: the heading's colour.
        features: the features, in the order the card lists them.
    """

    heading: str
    colour: str
    features: tuple[Feature, ...]


UNUSED_FEATURES = (
    FeatureGroup(
        "Message passing on Forney-style factor graphs",
        AMBER,
        (
            Feature("Trees of many coupled nodes", ("CouplingGraph", "Coupling")),
            Feature("A dynamics factor for each node", ("GaussianTransition",)),
            Feature(
                "Partitioned beliefs across time",
                ("CouplingGraphBackend",),
                argument="partition",
            ),
        ),
    ),
    FeatureGroup(
        "Expected free energy and state-dependent noise",
        GREEN,
        (
            Feature("Curiosity aimed at one node", ("FFGEFESelector",)),
            Feature("A grid over a scalar action", ("ObservationGoal", "EFESelector")),
            Feature("Dynamics noise set by the state", ("CallableDynamicsNoise",)),
        ),
    ),
    FeatureGroup(
        "Tools to differentiate, extend and describe a model",
        BLUE,
        (
            Feature("jit, vmap and grad through JAX", ("expected_free_energy",)),
            Feature("Your own inference backend", ("InferenceBackend",)),
            Feature("Declared factors and channels", ("ModelStructure",)),
        ),
    ),
)
"""What the pinned cpomdp does beyond the three tasks, one group to a card."""


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


def issue(number: int, title: str) -> list[Run]:
    """An open cpomdp issue: its number, padded so titles line up, then its title."""
    label = f"#{number}".ljust(4, "\u00a0") + "\u00a0"
    return [Run(label, colour=MUTED, font=CODE_FONT, size=15), Run(title)]


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
            f"[{COLAB.removeprefix('https://')}]",
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
            "orchestration for machine learning models. JOSS reviewer and an "
            "open-source contributor.",
            "I first came across active inference in late-night conversations with "
            "Kev here, and the two of us started playing around with pymdp. I’ll let "
            "Kev introduce himself…",
            "I'm interested in all things science, and two of my favourites are "
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
            "away. A hint gives you a nudge first, and the code line itself is behind "
            "Show code. Open them whenever you like. This isn't a coding test. We’d "
            "advice not using an AI agent to help, because we think you’ll get more "
            "out of this session if you don’t, but it’s your call.",
            "When a task ends, run its checkpoint cell, even if everything worked. It "
            "set everything the next task relies on, so we all start the next task "
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
            "front of a full room can be daunting, and the form sends us the question "
            "and nothing else. If it needs a longer answer, leave an email and we'll "
            "get back to you after the session. I think the format might mean we need "
            "to prioritise the youtube live comments but if we can we’ll get round to "
            "them. We’ll get back to you in if you leave your email, aiming for 48 "
            "hours, but I plan on going on holiday after this so we’ll see.",
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
            "Keep the question on the slide in mind while you code.",
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
            "to teach the rocket. A spread of 0.000000 nats means every move taught "
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
            "So how did we solve it in cpomdp? Now imagine your eyesight depends on "
            "where you stand, which is much closer to real life. Things blur with "
            "distance, and so do almost all real sensors.",
            "We write the sensor's noise as R. In Task 1, R was the same everywhere. "
            "If we now make noise depend on state, written as R(x), and introduce a "
            "“beacon” where noise is reduced, near the beacon the readings are sharp, "
            "and further away they get noisier. So where the rocket moves now changes "
            "how much it can expect to learn.",
            "In the code, k, called beacon_gain, sets how fast the noise grows as the "
            "rocket moves away from the beacon.",
            "Two things change from Task 1. First, the goal is now a reading, the one "
            "that says the rocket is on the pad, rather than a position. Second, the "
            "rocket holds that goal loosely. A goal has a precision, and the "
            "precision scales the pragmatic part but leaves the epistemic part alone.",
            "If anyone’s interested in the more mathsy side of stuff theres are some "
            "references we can give and we’d be happy to discuss later.",
            "[If anyone asks how exact the information numbers are:] When the rocket "
            "plans, it has to make a computational shortcut and assume the next "
            "reading is from the mean of that reading. This means it’s least "
            "accurate when the rocket is very unsure where it is and the noise "
            "changes sharply. Without it, we hit an intractability wall. There may "
            "be other methods, I believe there is a cpomdp research ticket of exactly "
            "that.",
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
            "Dan",
            "This timer runs for twenty minutes, because it includes a five-minute "
            "break.\nGo make a cuppa or whatever you’d like.",
            "Two lines again. First, look at the k value. Then, look my_sensor. Feel "
            "free to share anything in the chat as you go. We’ll reconvene in just "
            "less than 20 minutes.",
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
            f"[Kev. Planned at {clock('Task 2', DISCUSSION.name)}.]",
            "Welcome back. You've just flown the same rocket twice, once with "
            "curiosity on and once with it off.",
            f"Here's our run at k = {k:.2f}. The amber line has curiosity on. It "
            "flies up to the beacon, and still finishes a distance "
            f"{on.distance_to(PAD):.2f} from the pad. The blue line is the "
            "same rocket with curiosity off. It flies straight and finishes "
            f"{off.distance_to(PAD):.2f} from the pad. Both count as landed, because "
            "both end inside the circle, within 0.5 of the pad.",
            "The two rockets share one model and one sensor. The only difference is "
            "the epistemic part of the score, so the detour is curiosity. On this "
            "flight, curiosity decided 6 of the 36 moves.",
            "At what k did your curious rocket stop landing? [Pause for comments]",
            "On our seed, the curious rocket stops landing at k = 0.37, and the "
            "rocket with curiosity off keeps landing up to 0.45. So depending on the "
            "steps you took, you may have seen the curious rocket miss while the "
            "other one landed, or both of them miss. Both are right answers, and the "
            "next two slides explain each one.",
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
            "You might have seen the rocket miss or never leaving the beacon.",
            "The blue rocket, curiosity off, lands at the pad. The amber rocket, "
            "curiosity on, goes to the beacon and never leaves or reaches the goal."
            "\nThis is an example where epistemics has completely taken over, so "
            "it’s all a balancing act.",
            "Two settings decide whether curiosity helps. The precision sets what "
            "information is worth against the pull of the pad. k sets how much there "
            "is to gain by staying near the beacon.",
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
            "way, and the goal precision scales it. The epistemic score counts "
            "information, and the goal precision leaves it alone.",
            f"At goal precision 1 and k = {BALANCE_GAIN:.1f}, both routes bend "
            "towards the beacon, yet curiosity changes at most 1 of the 36 moves "
            "on any of 10 seeds. That bend is the rocket avoiding noise.",
        ],
        size=17, space_after=6,
    )  # fmt: skip
    page.text(
        MARGIN, 6.5, FULL_WIDTH, 0.6,
        ["We picked goal precision 0.1 so the two scores can compete. Expected "
         "free energy has no other dial for this."],
        size=19, bold=True, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            "[Kev.]",
            "Here we want to highlight another development problem we ran into.",
            "Does a bend towards the beacon always mean the rocket is curious? Not "
            "necessarily.",
            "On the left, the rocket flies at a goal precision of 1, meaning "
            "pragmatics dominates and epistemics barely creates a detour. On the "
            "right, at 0.1 there is a noticeable difference.",
            "So a route that looks curious might not be. The only way to tell which "
            "term chose a move is to fly it again with curiosity off.",
            'The rocket\'s goal is a reading that says "I am at the pad", and a '
            "noisy reading is less likely to match that goal, so the pragmatic score "
            "favours routes with cleaner readings, which lie towards the beacon, "
            "even when the rocket places no value on the beacon — like walking home "
            "along the lit street because you want to be sure of arriving, when you "
            "have no interest in seeing the sights.",
            "[If someone asks why noise is in the pragmatic score at all:] The goal "
            "is a reading rather than a position, because cpomdp "
            f"{WORKSHOP_CPOMDP_VERSION} won't accept a goal over positions when the "
            "noise depends on position. A goal over readings ends up paying for how "
            "noisy each reading is expected to be.",
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
            f"[Dan. Planned at {clock('Task 3')}.]",
            "In Task 2 the rocket knew where the pad was, and curiosity only changed "
            "its route. Now we hide the pad. The rocket starts with a guess, and it "
            "knows roughly how good that guess is.",
            "It has three sensors. It reads its own position. A coarse radar points "
            "roughly from the rocket towards the pad. And the beacon broadcasts the "
            "pad's exact position, but you only hear it clearly when you're close.",
            "The true pad lives in the generative process, the world, and the "
            "rocket's generative model only holds a belief about it.",
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
            "Two lines again, because the state now holds four numbers, the rocket's "
            "position and then the pad's. The first is prior_mean, where the rocket "
            "believes it starts and where it guesses the pad is, and the second is "
            "prior_cov, how unsure it is about both. See you in 15.",
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
        ROUTE_COLUMN, 1.9, width, 1.45,
        [
            "With curiosity on, the rocket lands on\u00a09.",
            "With curiosity off, it lands on\u00a03.",
        ],
        size=19, space_after=12,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 3.6, width, 2.2,
        ["A rocket that is sure of its guess expects to learn nothing at the "
         "beacon, so it never goes."],
        size=19, bold=True, colour=AMBER,
    )  # fmt: skip
    page.notes(
        notes(
            f"[Kev. Planned at {clock('Task 3', DISCUSSION.name)}.]",
            "Starting with the blue route, curiosity off. It heads for its guess and "
            "then on towards the pad, with only the coarse radar to correct it. It "
            f"ends {off.miss:.2f} from the true pad, outside the circle.",
            "Now amber, curiosity on. It goes up to the beacon first, learns where "
            f"the pad is, and flies there. It ends {on.miss:.2f} from the true pad. "
            "The shaded ellipse is where it believes the pad is by the end.",
            "The question is – does the blue rocket hear the broadcast from the "
            "beacon. Yes, but because it has no incentive to learn, it doesn’t bother "
            "travelling closer to be more precise, meaning its guess updates slowly, "
            "but not enough to find the pad.",
            "You lowered pad_spread until the curious rocket stopped finding the pad. "
            "A rocket that's sure of its guess expects to learn nothing at the "
            "beacon, so it doesn't go.",
            "Across 10 seeds, the curious rocket lands 9/10 times and the one with "
            "curiosity off 3/10 times. The median misses are 0.25 and 0.61. On the "
            "map that gap looks small, so the counts are the thing to look at.",
            "We should be honest that this works inside a window. Turn the goal "
            "precision up from 0.1 to 1 and the curious rocket lands on 3 seeds of "
            "10, the same as with curiosity off. Let the broadcast carry far and both "
            "rockets land on all 10, so curiosity buys nothing. Bonus 2 lets you find "
            "those edges yourself.",
            "Curiosity pays off here because what the rocket learns changes where it "
            "goes. There's a question on this under Checkpoint 3 in the notebook.",
        )
    )


def coda(deck: Deck) -> None:
    """The three warrants, with the evidence behind each of today's two checks."""
    planner = after_session_3()["planner"]
    if not isinstance(planner, RecedingHorizonSelector) or planner.certificate is None:
        raise ValueError("The workshop flight did not produce a search certificate.")
    seven_steps = axis_certificate()
    for certificate in (planner.certificate, seven_steps):
        if certificate.warrant is not Warrant.PROVED:
            raise ValueError(
                f"The search did not prove complete coverage: {certificate}"
            )
    rows = (
        ("Warrant", "What stands behind it", "Today"),
        (
            Warrant.PROVED.value,
            "A finite set, checked in full",
            f"The planner scored all {seven_steps.visited:,} plans in its "
            "set\u00a0of\u00a0moves",
        ),
        (
            Warrant.CERTIFIED.value,
            "A bound that holds over a whole\u00a0range",
            "Nothing today",
        ),
        (
            Warrant.CORROBORATED.value,
            "A sample of the possible cases",
            "At most: the curious rocket landing on 9\u00a0of\u00a010\u00a0seeds",
        ),
    )

    page = deck.page("What does PROVED mean here?", "why it matters")
    page.table(MARGIN, 1.6, (2.9, 4.1, FULL_WIDTH - 7.0), 3.0, rows)
    page.text(
        MARGIN, 4.85, FULL_WIDTH, 0.95,
        ["A passing check tells you the number held. It doesn't tell you what kind "
         "of evidence that is."],
        size=22, colour=AMBER, bold=True,
    )  # fmt: skip
    page.text(
        MARGIN, 6.05, FULL_WIDTH, 0.7,
        [[Run("These labels come from warrantlib, which installed with cpomdp "
              "today.  "),
          linked(WARRANTLIB, size=15)]],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            f"[Dan. Planned at {clock('Why it matters')}.]",
            "By now you should have ran the final cell, if not take a minute to do "
            "so. The final cell prints the planner's certificate, PROVED. As I’m sure "
            "many of us are aware in the research community, the pace of research is "
            "increasing. Being a JOSS reviewer and an active open-source contributor, "
            "I have seen this problem first hand and one of my biggest concerns was "
            "cpomdp disappearing into the cracks before it had even started. "
            "Specifically, around grandiose claims backed on shaky foundations. So "
            f"with cpomdp v{WORKSHOP_CPOMDP_VERSION} it also pairs with a separate "
            "project warrantlib. We don’t have too much time to go into it here, but "
            "the basic premise is a coder or AI agent can write a test that passes "
            "anything, but that test gives no credibility to how that number/finding "
            "was found. So…",
            "PROVED here is warrantlib’s vocabulary meaning an exhaustive search. "
            "There is no space that was unobserved, it our example every possibility "
            "in our finite action set was checked and compared. Nothing sampled, and "
            "nothing skipped.",
            'If you were to make a statement like "every action policy across the '
            'action continuum was checked", that would at best be a corroborated '
            "claim, because a continuum holds infinitely many actions and no search "
            "can visit them all. To say something about the whole range without "
            "visiting it, we would need a bound that covers it. That is the middle "
            "label, CERTIFIED, and nothing in today's workshop earns it.",
            "[If asked about the first line the cell prints:] It compares two of "
            "cpomdp's inference engines on the Task 1 rocket. They agree down to "
            "rounding error. It's a check on the library, not on today's rocket.",
        ),
        sources=(("warrantlib on PyPI", WARRANTLIB),),
    )


def where_cpomdp_sits(deck: Deck) -> None:
    """Nine things the room did with cpomdp in the workshop, in three groups."""
    page = deck.page("What did cpomdp let us do?", "why it matters")
    certificate = axis_certificate()
    card_row(
        page, 2.25, 2.65,
        (
            headed(
                "Change what the rocket could\u00a0learn",
                "Noise that depends on position",
                "A hidden pad it had to find",
                "Moves that teach it something",
                colour=AMBER,
            ),
            headed(
                "Weigh curiosity against the rocket’s\u00a0goal",
                "Information gain for each move",
                "Curiosity switched on and off",
                "A dial that balances the two",
                colour=GREEN,
            ),
            headed(
                "Check which plans the rocket\u00a0considered",
                "A search seven moves ahead",
                f"All {certificate.visited:,} plans scored",
                "A certificate that says PROVED",
                colour=BLUE,
            ),
        ),
        size=19,
    )  # fmt: skip
    page.text(
        MARGIN, 5.25, FULL_WIDTH, 0.7,
        [[Run("Every item ran in the notebook we worked through today.  "),
          linked(REPOSITORY, size=15)]],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "[Kev.]",
            "So what did cpomdp actually let us do today?",
            "First, it let us change what the rocket could learn. We made the "
            "sensor's noise depend on position. Task 3 we made the pad a latent, "
            "giving curiosity something to do and essentially building a continuous "
            "T-maze.",
            "Second, it let us weigh curiosity against the goal. In Task 1 we "
            "measured how much each move would teach the rocket, and the answer was "
            "nothing. Task 2 onwards we could switch curiosity on and off, and varied "
            "goal precision to see the pragmatic and epistemic pay offs.",
            "For context: if you know pymdp, that's the library for discrete states "
            "in Python. cpomdp does the same job for continuous states, like the "
            "rocket's position, and runs on JAX. RxInfer, in Julia, does Bayesian "
            "inference by message passing on factor graphs.",
            "Next, what else cpomdp can already do.",
        )
    )


def unused_features(deck: Deck) -> None:
    """Nine features of the pinned cpomdp that the three tasks did not use."""
    page = deck.page("What else can cpomdp already do?", "why it matters")
    card_row(
        page, 2.25, 2.65,
        tuple(
            headed(
                group.heading,
                *(feature.title for feature in group.features),
                colour=group.colour,
            )
            for group in UNUSED_FEATURES
        ),
        size=19,
    )  # fmt: skip
    page.text(
        MARGIN, 5.25, FULL_WIDTH, 0.7,
        [[Run(f"Every item is in cpomdp {WORKSHOP_CPOMDP_VERSION}, the release we "
              "installed today.  "),
          linked(CPOMDP_DOCS, size=15)]],
        size=17, colour=MUTED,
    )  # fmt: skip
    names = ", ".join(
        feature.lookup for group in UNUSED_FEATURES for feature in group.features
    )
    page.notes(
        notes(
            "[Kev.]",
            "We used a small part of cpomdp today. These are features of the "
            "release we installed that our three tasks did not need.",
            "On the left is a second inference engine. Everything today ran on a "
            "Kalman filter, which treats the whole state as one block. A factor "
            "graph draws the model as separate variables with links between them, "
            "and works out what it believes by passing messages along those links. "
            "cpomdp's graph engine runs trees of many coupled nodes. Each node gets "
            "its own dynamics factor, and we can partition the belief that the "
            "graph carries from one time step to the next. The Self-guided section "
            "has an example with two nodes.",
            "[If asked how the messages work:] Each message is a Gaussian stored as a "
            "precision and a potential, so combining two messages is an addition. On "
            "a simple chain, passing those messages gives the Kalman filter.",
            "In the middle is expected free energy. On a graph, we can aim curiosity "
            "at one node. Our rocket chose from a fixed set of compass moves, but "
            "cpomdp can also search a grid over a single continuous action. Our "
            "sensor noise depended on position, and the noise in the dynamics can "
            "depend on the state in the same way.",
            "On the right are the tools. The score is a JAX function, so we can "
            "compile it, run it over a batch of actions and take its gradient. We can "
            "write our own inference backend, because the Kalman filter and the graph "
            "engine meet the same interface. We can also declare which parts of the "
            "state belong together and which readings come from which sensor. "
            "Essentially I’m trying to make obvious that there is a level of granular "
            "control up to the use.",
            f"[The names, if someone asks, in the order of the cards: {names}.]",
            "Next, what is planned.",
        ),
        sources=(("The cpomdp documentation", CPOMDP_DOCS),),
    )  # fmt: skip


def roadmap(deck: Deck) -> None:
    """Nine open cpomdp issues in three groups of planned active inference features."""
    page = deck.page(
        "What’s already planned for the future of cpomdp?", "why it matters"
    )
    card_row(
        page, 2.25, 2.65,
        (
            headed(
                "Persistent goal seeking under a\u00a0homeostatic drive",
                issue(196, "Switched dynamics"),
                issue(201, "Energy preference"),
                issue(203, "Per-step surprise"),
                colour=AMBER,
            ),
            headed(
                "Nonlinear, multi-timescale inference and planning",
                issue(21, "Nonlinear sensors"),
                issue(28, "Multiple timescales"),
                issue(52, "Multi-step graph planning"),
                colour=GREEN,
            ),
            headed(
                "Parameter, category and structure learning",
                issue(231, "Learned sensor noise"),
                issue(232, "Discrete + continuous"),
                issue(233, "Structure learning"),
                colour=BLUE,
            ),
        ),
        size=19,
    )  # fmt: skip
    page.text(
        MARGIN, 5.25, FULL_WIDTH, 1.1,
        [[Run("Every item is an open issue. Comment on one, or propose your own.  "),
          linked(CPOMDP_ISSUES, size=15)],
         [Run("The research questions behind cpomdp are open for discussion too.  "),
          linked(CPOMDP_DISCUSSIONS, size=15)]],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "[Dan.]",
            "Here are a few issues I’ve pulled off the open issues board on "
            "cpomdp. They’re ones I’m personally excited to get working on, and to "
            "share with you guys to show cpomdp has some legs. If you’ve had any "
            "ideas for features or fixes you’d like to see in cpomdp, please let "
            "us know.",
            "The second link is the repository’s discussions page. That’s where "
            "the research questions behind cpomdp live, including the ones Kev and "
            "I are working on now. None of it is finished, and we’d welcome the "
            "conversation there.",
            "[Depending on time read them out].",
            "[If asked about #232:] A combined discrete-continuous graph could help "
            "us represent neuronal activity and other biology.",
            "[If asked whether planning further ahead helps:] We have one data "
            "point. Looking two moves ahead, today's curious rocket lands on all 10 "
            "seeds instead of 9. That's one result, not a trend.",
        ),
        sources=(
            ("Open issues on cpomdp", CPOMDP_ISSUES),
            ("Discussions on cpomdp", CPOMDP_DISCUSSIONS),
            ("Persistent foraging cell, the tracking issue", f"{CPOMDP_ISSUES}/194"),
            ("Category and structure learning, the tracking issue",
             f"{CPOMDP_ISSUES}/226"),
        ),
    )  # fmt: skip


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
            "Hopefully you’ve enjoyed working with some code and seeing what cpomdp "
            "can do. The story today was designed to show the different parts of "
            "active inference working in as much isolation as possible in a "
            "continuous environment. To do this obviously there was a lot of variable "
            "tuning, and this workshop should act as a cautionary tail that you can "
            "essentially produce any behaviour your want if your loose with your "
            "controls.",
            "If you want to keep going, there are three more experiments in the "
            "Self-guided section of the notebook, below Checkpoint 3. There are also "
            "getting started guides in the cpomdp documentation and some working "
            "examples with some pretty gifs.",
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
    """Open questions, with both presenters' emails and five QR codes in two rows."""
    page = deck.page()
    page.mark(MARGIN, 0.55, 0.85)
    side, gap, top = 1.7, 0.45, 1.75  # each QR code, the clear space between, its top
    pitch = side + 0.8  # from one row's top to the next, past the caption
    spill = 0.2  # how far a caption reaches past each side of its code
    per_row = 3
    codes = RIGHT - per_row * side - (per_row - 1) * gap  # where the QR codes start
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
        (
            (PRESENTER_2_LINKEDIN, "Kevin on LinkedIn"),
            (SITE, "inferogenesis.com"),
            (CPOMDP_REPOSITORY, "cpomdp on GitHub"),
            (CPOMDP_DOCS, "cpomdp documentation"),
            (WARRANTLIB_DOCS, "warrantlib documentation"),
        )
    ):
        row, place = divmod(index, per_row)
        left = codes + place * (side + gap)
        code_top = top + row * pitch
        page.qr(url, left, code_top, side)
        # the caption takes most of the gap as well, so the longest fits on one line
        page.text(
            left - spill, code_top + side + 0.15, side + 2 * spill, 0.4, [caption],
            size=13, colour=MUTED, align=PP_ALIGN.CENTER, space_after=0,
        )  # fmt: skip
    page.notes(
        notes(
            "[Dan and Kev. Leave this slide up during questions.]",
            "Thank you all for coming.",
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
    task_2_meaning,
    task_2_balance,
    task_3_brief,
    task_3_timer,
    task_3_result,
    coda,
    where_cpomdp_sits,
    unused_features,
    roadmap,
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
