"""Build the workshop deck from the package, the lesson and the figures.

    uv run --no-sync python scripts/render_assets.py   # the figures, first
    uv run --no-sync python scripts/build_deck.py

The slides carry the route figures, the code the room writes and what the cells
print. Code comes from ``lesson.py``, printed output from running the package on the
pinned release, figures from ``assets/``. Every number in the prose and the notes is
asserted in ``tests/test_measured_claims.py``. The build refuses to write a deck whose
shapes run off a slide, overlap or overflow.
"""

import contextlib
import io
import re
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

from cpomdp import KalmanBackend
from pptx.enum.text import MSO_ANCHOR

import lesson
from cpomdp_workshop import (
    WORKSHOP_CPOMDP_VERSION,
    after_session_3,
    beacon_sensor,
    check_agent,
    check_backends,
    check_my_sensor,
    default_params,
    model_with,
    one_node,
    probe,
    show_probe,
    symptoms,
    the_1960_rocket,
)
from cpomdp_workshop.symptoms import SYMPTOMS
from deck_kit import (
    AMBER,
    BLUE,
    CODE_FONT,
    CONTENT_TOP,
    INK,
    MARGIN,
    MATH_FONT,
    MUTED,
    SLIDE_WIDTH,
    Deck,
    Page,
    Run,
    code_height,
    code_size,
    without_imports,
)
from deck_layout import fit_problems, layout_problems, metrics

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DECK = ROOT / "deck" / "cpomdp_symposium_2026.pptx"

REPOSITORY = "https://github.com/inferogenesis/cpomdp-workshop"
COLAB = (
    "https://colab.research.google.com/github/inferogenesis/cpomdp-workshop"
    "/blob/main/notebooks/build_a_curious_rocket.ipynb"
)
PYPI = "https://pypi.org/project/cpomdp/"

RIGHT = SLIDE_WIDTH - MARGIN  # the right edge of the content
FULL_WIDTH = RIGHT - MARGIN
# Every route slide puts its figure here, so only the route changes between them.
ROUTE_FIGURE = (MARGIN, 1.4, 8.4)  # x, y, width
ROUTE_COLUMN = 9.3  # where the callouts beside a route frame start
TASK_CODE_BOTTOM = 6.0  # a task's code ends above the strip of instructions
TASK_STRIP = 6.2  # where that strip starts

PYMDP_CODE = """\
model = LinearGaussianModel(
    jnp.eye(2),  # B, the dynamics_matrix
    control_matrix=jnp.eye(2),  # B, where a move enters
    observation_matrix=jnp.eye(2),  # A
    observation_noise=0.05 * jnp.eye(2),
    dynamics_noise=1e-3 * jnp.eye(2),
    prior=Belief(mean=LAUNCH, cov=4 * jnp.eye(2)),  # D
)
goal = StateGoal(PAD, precision=jnp.eye(2), effort=100 * jnp.eye(2))  # C"""
"""Task 1's model under pymdp's letters. The deck test runs it."""

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
    ("A", "the observation likelihood", ("observation_matrix",)),
    ("B", "the transition", ("dynamics_matrix,", "control_matrix")),
    ("C", "the preference", ("StateGoal,", "Preference")),
    ("D", "the prior", ("prior",)),
)

DISCLOSURE = (
    f"today's notebook pins {WORKSHOP_CPOMDP_VERSION}. Some of its src was written by "
    "Claude Code under the review described."
)
"""Slide 4's caption, the owner's words."""


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
        size=17, line=AMBER, anchor=MSO_ANCHOR.MIDDLE, space_after=2,
    )  # fmt: skip


def task_page(deck: Deck, title: str, segment: str, attempt: str, paste: str) -> Page:
    """A task's code, the lines to write in amber, and a strip of what to do."""
    page = deck.page(title, segment)
    source = without_imports(attempt)
    size = code_size(source, FULL_WIDTH, TASK_CODE_BOTTOM - CONTENT_TOP)
    page.code(
        MARGIN, CONTENT_TOP, FULL_WIDTH, code_height(source, size), source, size=size
    )
    left, write, hints = MARGIN, 3.5, 3.7
    page.card(
        left, TASK_STRIP, write, 0.75,
        [[Run("Write the two amber lines", colour=AMBER)]],
        size=17, anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip
    left += write + 0.3
    page.card(
        left, TASK_STRIP, hints, 0.75, ["Hints come in chat on a timer"], size=17,
        anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip
    left += hints + 0.3
    paste_card(page, left, TASK_STRIP, RIGHT - left, 0.75, paste, stacked=False)
    return page


def under_code(page: Page, attempt: str, line: str) -> None:
    """One line between a task's code and its strip, where the code leaves room."""
    source = without_imports(attempt)
    size = code_size(source, FULL_WIDTH, TASK_CODE_BOTTOM - CONTENT_TOP)
    top = CONTENT_TOP + code_height(source, size) + 0.2
    page.text(MARGIN, top, FULL_WIDTH, TASK_STRIP - top - 0.1, [line], colour=MUTED)


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


# --- the slides, in order -------------------------------------------------------


def title(deck: Deck) -> None:
    """Slide 1: the title, the pin and the three builds."""
    page = deck.page()
    page.text(MARGIN, 1.5, 11.5, 1.2, ["Build a curious rocket"], size=60, bold=True)
    page.text(
        MARGIN, 2.85, 11.5, 0.6,
        ["cpomdp · continuous active inference in Python, hands-on"],
        size=26, colour=MUTED,
    )  # fmt: skip
    page.text(
        MARGIN, 3.5, 11.5, 0.5, ["6th Applied Active Inference Symposium · 2026"],
        size=20, colour=MUTED,
    )  # fmt: skip
    page.pill(
        MARGIN, 4.3, 2.6, 0.55, f"cpomdp {WORKSHOP_CPOMDP_VERSION}", size=18,
        font=CODE_FONT,
    )  # fmt: skip
    for index, (label, colour) in enumerate(
        (("1  the 1960 rocket", INK), ("2  a beacon", AMBER), ("3  the blackout", BLUE))
    ):
        page.pill(MARGIN + index * 4.1, 5.9, 3.8, 0.7, label, colour=colour, size=20)
    page.notes(
        notes(
            "Planned at 0:00. Keyboard shares the screen with this slide up and pastes "
            "the Colab link in chat. Kev opens.",
            "Everyone writes their own lines today, two per task. Move on while "
            "people are still joining. The install starts now.",
        )
    )


def presenters(deck: Deck) -> None:
    """Slide 2: who presents, and the install, started at 0:00."""
    page = deck.page("Presenters", "framing")
    for index, name in enumerate(("Presenter 1 (keyboard)", "Kev (concepts)")):
        left = MARGIN + index * 6.2
        page.placeholder(left, CONTENT_TOP, 1.9, 1.9, "[photo]")
        page.text(left + 2.2, CONTENT_TOP, 3.7, 0.5, [name], size=22, bold=True)
        page.placeholder(left + 2.2, CONTENT_TOP + 0.65, 3.7, 1.25, "[bio]")
    page.text(
        MARGIN, 4.0, 11.0, 0.5, ["Start now. Open the notebook in Colab"],
        size=20, bold=True,
    )  # fmt: skip
    page.link(MARGIN, 4.6, FULL_WIDTH, 0.4, COLAB, size=12)
    page.text(MARGIN, 5.35, 11.0, 0.5, ["Or run it locally"], size=20, bold=True)
    version = metadata.version("cpomdp-workshop")
    page.code(
        MARGIN, 5.95, 6.0, 0.6, f"pip install cpomdp-workshop=={version}", size=14
    )
    page.notes(
        notes(
            "Planned at 0:00. Keyboard pastes the Colab link in chat again. Colab "
            "needs nothing installed. Running locally works with the pip line, or "
            "with uv as the README shows. Cell 1 stops if cpomdp is not "
            f"{WORKSHOP_CPOMDP_VERSION}.",
            "Kev introduces us both while the installs run.",
            "TODO owner: photos and bios on the cards.",
            "TODO owner: publish cpomdp-workshop to PyPI before the session. Until "
            "then the pip line fails, and cell 1 fails on Colab too.",
        )
    )


def preview(deck: Deck) -> None:
    """Slide 3: cell 2's picture, the whole game first."""
    page = route_page(deck, "What you will build", "framing", "fig_world_rock.png")
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, RIGHT - ROUTE_COLUMN, 3.4,
        [
            "By 1:32 you will have built this picture.",
            "Three rockets, one rock. The dashed line is the 1960 controller.",
            [Run("Cell 2 draws it. "), Run("workshop.preview()", font=CODE_FONT)],
        ],
        size=19, space_after=14,
    )  # fmt: skip
    page.text(
        ROUTE_COLUMN, 6.3, RIGHT - ROUTE_COLUMN, 0.5, ["Seed 3."], size=15, colour=MUTED
    )
    page.notes(
        notes(
            "Kev. The whole game first. By 1:32 the room has built this picture in "
            "three steps.",
            "The dashed line is the 1960 rocket. It hits the rock on step 6. The two "
            "solid lines are curious rockets. At seed 3 both fly round the rock.",
            "Do not explain why yet. Task 3 and the slide after the rock do. Cell 2 "
            "draws this picture in under a minute.",
        )
    )


def disclosure(deck: Deck) -> None:
    """Slide 4: the AI-use panel, the owner's to write, and its caption."""
    page = deck.page("Disclosure", "framing")
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


def theorem(deck: Deck) -> None:
    """Slide 5: why a fixed sensor gives the 1960 controller."""
    page = deck.page("Under a fixed sensor, expected free energy is LQR", "framing")
    width = (FULL_WIDTH - 0.4) / 2
    for index, (glyph, colour, who, lines) in enumerate(
        (
            (
                "R", INK, "the 1960 rocket",
                (
                    "The reading is equally noisy everywhere.",
                    "Every move teaches the same.",
                    "Expected free energy reduces to LQR.",
                ),
            ),
            (
                "R(x)", AMBER, "a curious rocket",
                (
                    "The reading's noise depends on where the rocket is.",
                    "Where a move goes changes how clear the next reading is.",
                    "Tasks 2 and 3 build it.",
                ),
            ),
        )
    ):  # fmt: skip
        page.card(
            MARGIN + index * (width + 0.4), CONTENT_TOP, width, 3.6,
            [
                [Run(glyph, colour=colour, font=CODE_FONT, bold=True, size=44)],
                [Run(who, colour=MUTED)],
                *([Run(line)] for line in lines),
            ],
            size=20, space_after=10,
        )  # fmt: skip
    page.text(
        MARGIN, 5.35, FULL_WIDTH, 1.4,
        [
            "The 1960 controller is a Kalman filter with a goal. Task 1 builds it.",
            [Run("Koudahl, Kouw & de Vries, 2021", colour=MUTED, size=16)],
        ],
        size=21, space_after=10,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. If you have built a continuous active inference agent and watched it "
            "act as a controller, you did nothing wrong.",
            "Under a sensor whose noise never changes, every move teaches the same "
            "amount. Expected free energy then reduces to LQR, the result Koudahl, "
            "Kouw and de Vries published in 2021. Task 1 builds that rocket. The "
            "probe reports it.",
            "The rest of the session changes one thing. The noise depends on where the "
            "rocket is.",
            "TODO Kev: framing.",
        )
    )


def pymdp_names(deck: Deck) -> None:
    """Slide 6: pymdp's letters as cpomdp's field names."""
    page = deck.page("pymdp's letters, cpomdp's names", "framing")
    width = (FULL_WIDTH - 0.9) / 4
    for index, (letter, meaning, names) in enumerate(PYMDP_MAP):
        page.card(
            MARGIN + index * (width + 0.3), CONTENT_TOP, width, 2.0,
            [
                [Run(letter, colour=AMBER, bold=True, size=36)],
                [Run(meaning, colour=MUTED)],
                *([Run(name, font=CODE_FONT)] for name in names),
            ],
            size=15, space_after=2,
        )  # fmt: skip
    top = CONTENT_TOP + 2.25
    size = code_size(PYMDP_CODE, FULL_WIDTH, 2.6)
    page.code(
        MARGIN, top, FULL_WIDTH, code_height(PYMDP_CODE, size), PYMDP_CODE, size=size
    )
    page.text(
        MARGIN, 6.3, FULL_WIDTH, 0.5,
        [
            f"Cell 1 checks for cpomdp {WORKSHOP_CPOMDP_VERSION} and Python 3.11 or "
            "newer."
        ],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. pymdp's letters in cpomdp's names. A is the observation likelihood, "
            "so observation_matrix. B is the transition, so dynamics_matrix, with "
            "control_matrix for where a move enters. C is the preference. The 1960 "
            "rocket takes a StateGoal and the curious one a Preference. D is the "
            "prior.",
            "The code is task 1's model. Keyboard runs cell 1 now if it has not run.",
        )
    )


def continuous_state(deck: Deck) -> None:
    """Slide 7: what continuous state buys you."""
    page = deck.page("What continuous state buys you", "framing")
    page.picture(ASSETS / "tile_task1.png", MARGIN, CONTENT_TOP, 7.4)
    page.text(
        8.4, CONTENT_TOP, RIGHT - 8.4, 4.8,
        [
            "A position is two real numbers.",
            "A belief is a mean and a covariance, drawn as an ellipse.",
            "Each reading shrinks it in closed form. That is the Kalman update.",
            "The reading's noise, R, is what today changes.",
        ],
        size=19, space_after=16,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. Open decision: keep this slide, or swap in the task tiles. Keep is "
            "recommended.",
            "A belief here is a mean and a covariance. The shaded ellipses are two "
            "sigma, after 1, 2, 4 and 8 readings. Each reading shrinks it in closed "
            "form.",
            "At 0:10 everyone runs cell 2 and sees the preview.",
        )
    )


def task_1(deck: Deck) -> None:
    """Slide 8: task 1's code."""
    attempt = lesson.SESSION_1.attempt
    page = task_page(deck, "Task 1 · the 1960 rocket", "task 1", attempt, "spread")
    under_code(page, attempt, "Then the bonus sliders. Try to make it curious.")
    page.notes(
        notes(
            "Planned at 0:12. Keyboard. Hands-on starts.",
            "Two lines to write, the observation matrix and its noise. The rest is "
            "written, the thruster cost included. Hint 1 and hint 2 go in chat on the "
            "timer from hints_chat.txt. The reference cell is collapsed under them. "
            "Using it is fine. The imports sit at the top of the cell.",
        )
    )


def task_1_close(deck: Deck) -> None:
    """Slide 9: the first route frame, and what task 1 printed."""
    page = route_page(
        deck,
        "The 1960 rocket flies the straight line",
        "task 1",
        "route_1_straight.png",
    )
    landing = captured(check_agent, the_1960_rocket())
    route_strip(page, output=f"{landing}\n{captured(show_probe, probe(model_with()))}")
    callout(
        page, "spread", Run("0.000000", font=CODE_FONT, bold=True, size=36),
        "nats between the move that teaches most and the one that teaches least",
    )  # fmt: skip
    task_close(page, "spread", 1)
    page.notes(
        notes(
            "Keyboard shows what the cell printed. The rocket is LQR, and lands 0.13 "
            "from the pad.",
            "The probe asks whether any of the nine compass moves teaches more than "
            "another. spread = 0.000000 nats. None does. Every spread pasted in chat "
            "should read zero, whatever the bonus sliders changed.",
            "Then poll 1 for 60 s. Kev reads the split.",
            plain(lesson.SESSION_1.poll),
        )
    )


def task_2(deck: Deck) -> None:
    """Slide 10: task 2's code."""
    attempt = lesson.SESSION_2.attempt
    page = task_page(deck, "Task 2 · a beacon", "task 2", attempt, "your k")
    page.notes(
        notes(
            "Keyboard. The reading now depends on where the rocket is. Two lines, the "
            "beacon's pull inside my_sensor and k.",
            "The goal changes too. This rocket prefers a reading, the one that says "
            "on the pad. The planner scores the nine compass moves every step.",
            "The room starts at k = 0.08 and looks for a k that bends the path up to "
            "the beacon. Hint 2 gives k = 0.4.",
        )
    )


def task_2_close(deck: Deck) -> None:
    """Slide 11: the second route frame. The beacon bends the route."""
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
            "Keyboard shows check_my_sensor's line for k = 0.4. The rocket bends 1.41 "
            "off the straight line. It lands 0.24 from the pad.",
            "Same map as the last route slide. Only the field and the amber line are "
            "new.",
            "Ask for k in chat, then poll 2 for 60 s. Kev reads the split.",
            plain(lesson.SESSION_2.poll),
            "The bonus cell backs answer 3. At k = 0.6 the rocket stops 1.49 short of "
            "the pad looking one move ahead, 1.65 looking two and 1.86 looking three.",
        )
    )


def break_slide(deck: Deck) -> None:
    """Slide 12: the break."""
    page = deck.page("Break", "break")
    page.text(
        MARGIN, CONTENT_TOP, 4.2, 2.6, ["6"], size=150, bold=True, colour=AMBER,
        anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip
    page.text(MARGIN, 4.2, 4.2, 0.6, ["minutes, cameras off"], size=24, colour=MUTED)
    page.text(
        5.4, CONTENT_TOP + 0.3, RIGHT - 5.4, 3.6,
        [
            "Behind? Run checkpoint 2. Task 3 starts from it.",
            "One up, one down. One thing that worked and one that did not, on the "
            "anonymous form.",
        ],
        size=22, space_after=24,
    )  # fmt: skip
    page.notes(
        notes(
            "Planned at 0:56. Six minutes, cameras off. Keyboard puts a timer on "
            "screen.",
            "Whoever is not presenting reads the form and the one-up-one-down answers. "
            "Back at 1:02.",
        )
    )


def task_3(deck: Deck) -> None:
    """Slide 13: task 3's code."""
    attempt = lesson.SESSION_3.attempt
    page = task_page(
        deck, "Task 3 · the blackout", "task 3", attempt, "max |difference|"
    )
    under_code(page, attempt, "There is a rock in the blackout. No rocket models it.")
    page.notes(
        notes(
            "Planned at 1:02. Keyboard. Two lines again, the blackout term in my_world "
            "and the one-node backend. The slider sets k for the room's rocket.",
            "plot_world flies three rockets past a rock none of them models. "
            "check_backends flies the 1960 rocket on the Kalman filter and on the "
            "graph, and prints max |difference|. Both lines are written.",
        )
    )


def symptoms_reveal(deck: Deck) -> None:
    """Slide 14: the one error read aloud, and the symptoms cell."""
    page = deck.page("Stuck? Read the error, then the fix", "task 3")
    first = SYMPTOMS[0]
    message = captured(symptoms).splitlines()[2].split("cpomdp", 1)[1].strip()
    top = CONTENT_TOP
    for label, line, printed in (
        ("you wrote", first.wrong, False),
        ("cpomdp says", message, True),
        ("write", first.right, False),
    ):
        page.text(MARGIN, top, 4.0, 0.35, [label], size=15, colour=MUTED)
        if printed:
            page.printed(MARGIN, top + 0.4, FULL_WIDTH, 0.8, line, size=12, wrap=True)
            top += 1.4
        else:
            page.code(MARGIN, top + 0.4, FULL_WIDTH, 0.55, line, size=12)
            top += 1.15
    page.text(
        MARGIN, top + 0.1, 5.2, 0.5, ["All five, each with its fix"], size=18, bold=True
    )
    page.code(5.9, top + 0.05, 3.6, 0.55, "workshop.symptoms()", size=14)
    page.notes(
        notes(
            "Keyboard. The one live reveal. Read the error aloud, then the fix.",
            "This is papercut 1, a sensor passed into the noise slot. Before 0.5.0 the "
            "same line raised a raw TypeError. Now cpomdp names the slot.",
            "workshop.symptoms() prints the five errors the room is most likely to "
            "hit, in the pinned release's own words. Each comes with its fix.",
            "TODO owner: the papercut-1 origin line the lost deck carried, in your "
            "words, if you want it back.",
        )
    )


def blackout(deck: Deck) -> None:
    """Slide 15: the third route frame, the blackout before the rock."""
    page = route_page(
        deck, "The blackout, before the rock", "task 3", "route_3_blackout.png"
    )
    route_strip(page, caption="Seed 3. No rock yet.")
    page.text(
        ROUTE_COLUMN, CONTENT_TOP, RIGHT - ROUTE_COLUMN, 5.0,
        [
            "Near the blackout's centre the reading is swamped.",
            "The 1960 rocket flies straight through. Its model has one noise level.",
            "The curious rockets swing wide.",
        ],
        size=20, space_after=18,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. The same map again. The field now has the blackout.",
            "The 1960 rocket flies the line it flew in task 1. Its model has one noise "
            "level. It flies in the world that model describes. The curious rockets "
            "swing wide of the bright patch.",
            "There is no rock yet. The next slide adds it and nothing else.",
            "TODO Kev: framing.",
        )
    )


def rock(deck: Deck) -> None:
    """Slide 16: the last route frame. The rock, and the 1960 rocket's impact."""
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
        "No rocket models the rock.",
    )  # fmt: skip
    task_close(page, "max |difference|", 3)
    page.notes(
        notes(
            "Kev. One thing added, the rock in the blackout. No rocket models it.",
            "The 1960 rocket hits it on step 6, on every seed from 0 to 9. The figure "
            "is seed 3. Both curious rockets clear it there. k = 0.08 peaks at y = "
            "1.71 and ends at (4.83, 0.09). k = 0.20 peaks at 1.91 and ends at "
            "(5.18, 0.15).",
            "Seed 3 is one seed. Over seeds 0 to 9 the k = 0.08 rocket clips the rock "
            "on seeds 1 and 2, and the k = 0.20 rocket on seeds 1, 2 and 9.",
            "Ask for max |difference| in chat. Below 1e-9 means the two engines agree. "
            "Then poll 3 for 60 s.",
            plain(lesson.SESSION_3.poll),
            "Looking two moves ahead is a self-guided cell. It runs fast enough to "
            "show live if there is time.",
        )
    )


def what_steers(deck: Deck) -> None:
    """Slide 17: what changes the route."""
    page = deck.page("What steers the route", "task 3")
    page.card(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 1.8,
        [
            [Run("the expected miss from “on the pad”", colour=MUTED, font="Calibri")],
            [Run("½ (o⁺ − o*)ᵀ Λ (o⁺ − o*)  +  ½ tr(Λ S)", size=24)],
            [
                Run("S = C Σ⁺ Cᵀ + "),
                Run("R(μ⁺)", colour=AMBER, bold=True),
                Run("    the reading's own noise, where the move is expected to land",
                    colour=MUTED, font="Calibri", size=17),
            ],
        ],
        size=20, font=MATH_FONT, space_after=6, anchor=MSO_ANCHOR.MIDDLE,
    )  # fmt: skip
    page.text(
        MARGIN, 3.55, FULL_WIDTH, 2.3,
        [
            "Each step the rocket scores the nine moves. The score's first term is how "
            "far its reading is expected to land from “on the pad”.",
            "That expectation carries the reading's own noise. Where the reading is "
            "noisy, it lands far from the pad on average. So moves into the blackout "
            "score badly.",
            "It takes a sensor whose noise depends on position. The 1960 rocket's "
            "never does.",
        ],
        size=19, space_after=12,
    )  # fmt: skip
    page.text(
        MARGIN, 6.05, FULL_WIDTH, 0.75,
        [
            "Put the same rock in clear sky on their arc and both curious rockets hit "
            "it on step 5."
        ],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. The answer to polls 2 and 3.",
            "Each step the rocket scores the nine moves by expected free energy. Its "
            "first term is the expected miss from the preferred reading, on the pad. "
            "That expectation carries S, and S carries R where the move is expected to "
            "land. Where the reading is noisy, a reading taken on the pad itself lands "
            "far from it on average. Moves into the blackout score badly. Moves toward "
            "the beacon score well.",
            "What steers the route is the reading noise inside the expected miss. That "
            "term moves only when R depends on position. For the 1960 rocket it is the "
            "same for every move. LQR is what is left.",
            "The clear-sky bonus shows what the rockets avoid. With the rock moved out "
            "of the blackout onto their arc, both hit it on step 5. The 1960 rocket "
            "flies under it.",
            "TODO Kev: framing. Whether curious is the word for a rocket steered this "
            "way is yours to settle.",
        )
    )


def backends(deck: Deck) -> None:
    """Slide 18: the 1960 rocket on three engines."""
    page = deck.page("One rocket, three engines", "backends")
    size = code_size(GRAPH_CODE, 8.9, 3.45)
    height = code_height(GRAPH_CODE, size)
    page.code(MARGIN, CONTENT_TOP, 8.9, height, GRAPH_CODE, size=size)
    page.text(
        9.8, CONTENT_TOP, RIGHT - 9.8, 3.45,
        [
            [Run("KalmanBackend", font=CODE_FONT)],
            [Run("the filter", colour=MUTED)],
            [Run("CouplingGraphBackend", font=CODE_FONT)],
            [Run("the same model as a graph of one node", colour=MUTED)],
            [Run("RxInferBackend", font=CODE_FONT)],
            [Run("the fixed-noise case only", colour=MUTED)],
        ],
        size=16, space_after=6,
    )  # fmt: skip
    model = model_with()
    beliefs = captured(check_backends, KalmanBackend(model), one_node(model))
    top = CONTENT_TOP + max(height, 3.45) + 0.3
    page.printed(
        MARGIN, top, FULL_WIDTH, 0.75, "\n".join(beliefs.splitlines()[:2]), size=12
    )
    page.text(
        MARGIN, top + 1.0, FULL_WIDTH, 0.75,
        [
            "The two agree below 1e-9. RxInfer covers the fixed-noise case and refuses "
            "the curious model before Julia starts."
        ],
        size=17, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Keyboard. The 1960 rocket on three engines. The code is what one_node "
            "builds. Agent.from_backend puts a rocket on either backend.",
            "The two print the same belief to six places. max |difference| sits "
            "below 1e-9. The coda shows RxInfer refusing the curious model.",
        )
    )


def coda(deck: Deck) -> None:
    """Slide 19: the certificates, as the coda cell prints them."""
    page = deck.page("What a printed number entitles you to say", "coda")
    height = code_height(lesson.CODA, 16)
    page.code(MARGIN, CONTENT_TOP, FULL_WIDTH, height, lesson.CODA, size=16)
    output = run_cell(lesson.CODA, after_session_3())
    top = CONTENT_TOP + height + 0.25
    page.printed(MARGIN, top, FULL_WIDTH, 1.45, output, size=15, wrap=True)
    page.text(
        MARGIN, top + 1.75, FULL_WIDTH, 6.9 - top - 1.75,
        [
            "PROVED means the search scored every plan in a declared, versioned set. "
            "The certificate counts them.",
            "Nine plans a step for the compass set. Five moves seven deep is 78125 "
            "plans, each one scored.",
        ],
        size=20, space_after=14,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. What a printed number entitles you to say.",
            "The certificate names the set by its version, compass-9, and counts every "
            "plan the search scored. The axis set is five moves. Seven deep it is "
            "78125 plans, all held at once.",
            "The third line is RxInfer refusing the curious model. Its filter fixes Q "
            "and R.",
        )
    )


def past_one_filter(deck: Deck) -> None:
    """Slide 20: the two-node model and its refusal."""
    page = deck.page("Past one filter", "self-guided")
    _, source = lesson.TAIL[2]
    height = code_height(source, 16)
    page.code(MARGIN, CONTENT_TOP, 7.2, height, source, size=16)
    page.text(
        8.1, CONTENT_TOP, RIGHT - 8.1, height,
        ["A rocket towing a probe is two nodes.", "The full field reads the probe."],
        size=20, space_after=12,
    )  # fmt: skip
    top = CONTENT_TOP + height + 0.25
    page.printed(MARGIN, top, FULL_WIDTH, 2.0, run_cell(source, {}), size=15, wrap=True)
    page.text(
        MARGIN, top + 2.3, FULL_WIDTH, 0.8,
        ["No single fixed linear-Gaussian model reproduces it. cpomdp refuses to "
         "flatten it."],
        size=18, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            "Kev. A rocket towing a probe is two nodes. The engine moves the rocket. "
            "The probe trails toward it. The full field reads the probe.",
            "The coupling shifts the probe's predicted mean. The noise is read at that "
            "mean. cpomdp refuses to flatten it into one filter.",
        )
    )


def self_guided(deck: Deck) -> None:
    """Slide 21: what the self-guided tail holds."""
    page = deck.page("Self-guided, below the last checkpoint", "self-guided")
    blurbs = (
        "A loose and a tight preference for the pad, flown past the rock.",
        "The world again, two moves ahead. Each step scores 162 plans in place of 9.",
        "The two-node refusal, to read in full.",
    )
    width = (FULL_WIDTH - 0.6) / 3
    for index, ((name, _), blurb) in enumerate(zip(lesson.TAIL, blurbs, strict=True)):
        page.card(
            MARGIN + index * (width + 0.3), CONTENT_TOP, width, 2.9,
            [[Run(name, bold=True)], blurb], size=21, space_after=16,
        )  # fmt: skip
    page.text(
        MARGIN, 4.85, FULL_WIDTH, 1.6,
        [
            "Nothing below the last checkpoint runs live. Each cell stands on "
            "checkpoint 3.",
            "Each task has a bonus cell too.",
        ],
        size=20, colour=MUTED, space_after=14,
    )  # fmt: skip
    page.notes(
        notes("Kev. Pointers only. Two moves ahead costs 162 plans a step against 9.")
    )


def inferogenesis(deck: Deck) -> None:
    """Slide 22: the Inferogenesis segment, the owner's to write."""
    page = deck.page("Inferogenesis", "Inferogenesis")
    page.placeholder(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 5.2,
        "[owner: the Inferogenesis segment, five minutes at most]", size=20,
    )  # fmt: skip
    page.notes(notes("TODO owner: this segment. Five minutes at most."))


def what_next(deck: Deck) -> None:
    """Slide 23: what comes next, the owner's to restore."""
    page = deck.page("What comes next", "Inferogenesis")
    page.placeholder(
        MARGIN, CONTENT_TOP, FULL_WIDTH, 4.2, "[owner: this slide's content]", size=20
    )
    page.pill(MARGIN, 6.0, 4.6, 0.6, "Status: scoped, starts after p*", size=17)
    page.notes(
        notes(
            "TODO owner: the content of this slide from the lost deck. Its status "
            "line read: scoped, starts after p*. It stays."
        )
    )


def thanks(deck: Deck) -> None:
    """Slide 24: thanks, and the pin the session ran on."""
    page = deck.page()
    page.text(MARGIN, 1.4, 11.5, 1.2, ["Thank you"], size=60, bold=True)
    page.text(
        MARGIN, 2.8, 11.5, 1.2,
        ["Questions now, or on the anonymous form.", "One up, one down before you go."],
        size=24, colour=MUTED, space_after=10,
    )  # fmt: skip
    page.placeholder(MARGIN, 4.3, 5.0, 0.8, "[email]", size=18)
    page.text(
        MARGIN, 5.9, 11.5, 0.6,
        [f"Everything shown runs on cpomdp {WORKSHOP_CPOMDP_VERSION} from PyPI."],
        size=18, colour=MUTED,
    )  # fmt: skip
    page.notes(
        notes(
            f"Kev. Everything shown runs on cpomdp {WORKSHOP_CPOMDP_VERSION} from "
            "PyPI. Recording, notebook, package, slides and notes go to the Discord "
            "channel and the README within 48 hours.",
            "TODO owner: the email.",
        )
    )


def links(deck: Deck) -> None:
    """Slide 25: where everything lives, with the survey."""
    page = deck.page("Links", "close")
    top = CONTENT_TOP
    for label, url in (
        ("notebook", COLAB),
        ("repository", REPOSITORY),
        ("cpomdp on PyPI", PYPI),
    ):
        page.text(MARGIN, top, 4.0, 0.35, [label], size=15, colour=MUTED)
        page.link(MARGIN, top + 0.38, FULL_WIDTH, 0.35, url, size=12)
        top += 0.95
    page.text(MARGIN, top, 4.0, 0.35, ["Discord"], size=15, colour=MUTED)
    page.text(
        MARGIN, top + 0.38, FULL_WIDTH, 0.45, ["the cpomdp channel on the AII Discord"],
        size=18,
    )  # fmt: skip
    width = (FULL_WIDTH - 0.4) / 2
    for index, label in enumerate(
        ("[survey link]", f"[Zenodo DOI of cpomdp {WORKSHOP_CPOMDP_VERSION}]")
    ):
        page.placeholder(MARGIN + index * (width + 0.4), 5.9, width, 0.8, label)
    page.notes(
        notes(
            "Leave this up through Q&A. The survey link goes in chat as well, while "
            "the room is still here.",
            "TODO owner: the survey link, and the Zenodo DOI once minted.",
        )
    )


SLIDES: tuple[Callable[[Deck], None], ...] = (
    title,
    presenters,
    preview,
    disclosure,
    theorem,
    pymdp_names,
    continuous_state,
    task_1,
    task_1_close,
    task_2,
    task_2_close,
    break_slide,
    task_3,
    symptoms_reveal,
    blackout,
    rock,
    what_steers,
    backends,
    coda,
    past_one_filter,
    self_guided,
    inferogenesis,
    what_next,
    thanks,
    links,
)


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
    """Build the deck, check its layout and write it."""
    deck = build()
    found = problems(deck)
    if found:
        print("\n".join(found))
        sys.exit(1)
    deck.save(DECK)
    print(f"wrote {DECK.relative_to(ROOT)}, {len(SLIDES)} slides")


if __name__ == "__main__":
    main()
