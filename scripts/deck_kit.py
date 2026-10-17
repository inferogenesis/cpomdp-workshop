"""Drawing primitives for the workshop deck, in the figures' palette.

Every position and size is in inches on a 16:9 slide, 13.333 wide and 7.5 high. A
page is one slide with its title in the same place as every other page's. Code sits
in a filled panel, printed output in an outlined one. A placeholder the owner has to
fill is drawn dashed in amber, so it cannot pass for finished content.

Shapes are named by kind (``title``, ``code``, ``printed``, ``text``, ``picture``,
``placeholder``, ``card``, ``pill``). The layout check and the deck test read the
names, so a number inside program output is told apart from a number in prose.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.shapes.autoshape import Shape
from pptx.slide import Slide
from pptx.text.text import TextFrame
from pptx.util import Emu, Inches, Pt

from cpomdp_workshop import plot

__all__ = [
    "AMBER",
    "BACKGROUND",
    "BLUE",
    "CODE_FONT",
    "CONTENT_TOP",
    "INK",
    "MARGIN",
    "MATH_FONT",
    "MUTED",
    "PANEL",
    "RULE",
    "SLIDE_HEIGHT",
    "SLIDE_WIDTH",
    "TEXT_FONT",
    "Deck",
    "Page",
    "Run",
    "code_height",
    "code_size",
    "without_imports",
]

# The figures' palette, so a slide and the figure on it agree. Amber is the one
# departure: slides use the brighter #E69F00 where the figures use #C48200.
BACKGROUND = plot.BACKGROUND.lstrip("#")
PANEL = plot.PANEL.lstrip("#")
RULE = plot.RULE.lstrip("#")
INK = plot.INK.lstrip("#")
MUTED = plot.MUTED.lstrip("#")
BLUE = plot.BLUE.lstrip("#")
AMBER = "E69F00"

TEXT_FONT = "Calibri"
CODE_FONT = "Courier New"
MATH_FONT = "Cambria"
# Courier New has no superscript plus or minus, and cpomdp's messages print both (μ⁺,
# μ⁻). They are set in the math face, which has them, so no renderer has to guess a
# substitute. Cambria Math has them too, with a line height three times the code face's.
_CODE_FALLBACK = re.compile("([⁺⁻])")

SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 7.5
MARGIN = 0.6
CONTENT_TOP = 1.45

_TITLE = (MARGIN, 0.42, 10.2, 0.8)  # x, y, w, h
_SEGMENT = (11.03, 0.58, 1.7, 0.42)
_PANEL_INSET = 0.18
_CODE_ADVANCE = 0.6  # Courier New sets every glyph 0.6 em wide
_CODE_LINE = 1.133  # and a line 1.133 em high
_SLOT = re.compile(r"^\s*\w+ = \.\.\.")


@dataclass(frozen=True)
class Run:
    """A stretch of text in one style inside a paragraph.

    Attributes:
        text: the characters.
        colour: hex without the hash, or ``None`` for the paragraph's colour.
        bold: whether it is bold.
        font: the typeface, or ``None`` for the paragraph's.
        size: the size in points, or ``None`` for the paragraph's.
    """

    text: str
    colour: str | None = None
    bold: bool = False
    font: str | None = None
    size: float | None = None


Paragraph = str | Sequence[Run]


def _rgb(colour: str) -> RGBColor:
    return RGBColor.from_string(colour)


def code_size(
    source: str, width: float, height: float, *, largest: float = 14.0
) -> float:
    """The largest size, in half points, at which ``source`` fits a code panel.

    Args:
        source: the code.
        width: the panel's width, in inches.
        height: the most height the panel may take, in inches.
        largest: the size not to go past.

    Returns:
        The size in points.
    """
    lines = source.splitlines()
    inner_width = width - 2 * _PANEL_INSET
    inner_height = height - 1.4 * _PANEL_INSET
    by_width = inner_width / (max(map(len, lines)) * _CODE_ADVANCE / 72)
    by_height = inner_height / (len(lines) * _CODE_LINE / 72)
    return min(largest, math.floor(2 * min(by_width, by_height)) / 2)


def code_height(source: str, size: float) -> float:
    """The height, in inches, of a code panel that holds ``source`` at ``size``."""
    return len(source.splitlines()) * size * _CODE_LINE / 72 + 1.4 * _PANEL_INSET + 0.04


def without_imports(source: str) -> str:
    """``source`` with its leading import block and the blank lines after it cut.

    A slide shows the lines a learner reads. The imports are in the cell.
    """
    lines = source.splitlines()
    start, in_parentheses = 0, False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if in_parentheses:
            in_parentheses = not stripped.startswith(")")
        elif stripped.startswith(("import ", "from ")):
            in_parentheses = stripped.endswith("(")
        elif stripped:
            start = index
            break
    return "\n".join(lines[start:])


class Page:
    """One slide, and the calls that put things on it."""

    def __init__(self, slide: Slide) -> None:
        self.slide = slide

    def _box(self, x: float, y: float, w: float, h: float, name: str) -> TextFrame:
        shape = self.slide.shapes.add_textbox(
            Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.name = name
        frame = shape.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = Inches(0.0)
        frame.margin_top = frame.margin_bottom = Inches(0.0)
        return frame

    def _panel(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        name: str,
        *,
        fill: str | None,
        line: str | None,
        dashed: bool = False,
    ) -> Shape:
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.name = name
        shape.adjustments[0] = min(0.08, 0.12 / min(w, h))
        shape.shadow.inherit = False
        if fill is None:
            shape.fill.background()
        else:
            shape.fill.solid()
            shape.fill.fore_color.rgb = _rgb(fill)
        if line is None:
            shape.line.fill.background()
        else:
            shape.line.color.rgb = _rgb(line)
            shape.line.width = Pt(1.25 if dashed else 1)
            if dashed:
                shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        frame = shape.text_frame
        frame.word_wrap = True
        inset = Inches(_PANEL_INSET)
        frame.margin_left = frame.margin_right = inset
        frame.margin_top = frame.margin_bottom = Inches(_PANEL_INSET * 0.7)
        return shape

    @staticmethod
    def _fill(
        frame: TextFrame,
        paragraphs: Sequence[Paragraph],
        *,
        size: float,
        colour: str,
        font: str,
        bold: bool = False,
        align: PP_ALIGN = PP_ALIGN.LEFT,
        space_after: float = 0.0,
        line_spacing: float = 1.0,
    ) -> None:
        for index, paragraph in enumerate(paragraphs):
            target = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            target.alignment = align
            target.line_spacing = line_spacing
            if index < len(paragraphs) - 1:
                target.space_after = Pt(space_after)
            runs = [Run(paragraph)] if isinstance(paragraph, str) else paragraph
            for piece in runs:
                face = piece.font or font
                for text, fallback in _glyph_runs(piece.text, face):
                    run = target.add_run()
                    run.text = text
                    run.font.size = Pt(piece.size or size)
                    run.font.name = MATH_FONT if fallback else face
                    run.font.bold = piece.bold or bold
                    run.font.color.rgb = _rgb(piece.colour or colour)

    def title(self, text: str, segment: str) -> None:
        """The slide's title, and the part of the session it belongs to, top right."""
        frame = self._box(*_TITLE, name="title")
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        self._fill(frame, [text], size=32, colour=INK, font=TEXT_FONT, bold=True)
        self.pill(*_SEGMENT, segment, colour=MUTED, size=13)

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: Sequence[Paragraph],
        *,
        size: float = 18,
        colour: str = INK,
        bold: bool = False,
        font: str = TEXT_FONT,
        align: PP_ALIGN = PP_ALIGN.LEFT,
        anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
        space_after: float = 8,
        name: str = "text",
    ) -> None:
        """Prose, one paragraph per entry."""
        frame = self._box(x, y, w, h, name)
        frame.vertical_anchor = anchor
        self._fill(
            frame, paragraphs, size=size, colour=colour, font=font, bold=bold,
            align=align, space_after=space_after,
        )  # fmt: skip

    def code(
        self, x: float, y: float, w: float, h: float, source: str, *, size: float = 12
    ) -> None:
        """Python in a filled panel. Lines to write are amber, comments muted."""
        shape = self._panel(x, y, w, h, "code", fill=PANEL, line=RULE)
        shape.text_frame.word_wrap = False
        shape.text_frame.vertical_anchor = MSO_ANCHOR.TOP
        self._fill(
            shape.text_frame, [_code_line(line) for line in source.splitlines()],
            size=size, colour=INK, font=CODE_FONT,
        )  # fmt: skip

    def printed(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        output: str,
        *,
        size: float = 12,
        wrap: bool = False,
    ) -> None:
        """What a cell printed, verbatim, in an outlined panel."""
        shape = self._panel(x, y, w, h, "printed", fill=None, line=RULE)
        shape.text_frame.word_wrap = wrap
        shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        self._fill(
            shape.text_frame, output.splitlines(), size=size, colour=INK, font=CODE_FONT
        )

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: Sequence[Paragraph],
        *,
        size: float = 18,
        colour: str = INK,
        font: str = TEXT_FONT,
        anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
        space_after: float = 8,
        line: str | None = None,
    ) -> None:
        """Text on a filled panel."""
        shape = self._panel(x, y, w, h, "card", fill=PANEL, line=line)
        shape.text_frame.vertical_anchor = anchor
        self._fill(
            shape.text_frame, paragraphs, size=size, colour=colour, font=font,
            space_after=space_after,
        )  # fmt: skip

    def placeholder(
        self, x: float, y: float, w: float, h: float, label: str, *, size: float = 16
    ) -> None:
        """A dashed amber box naming what the owner still has to put here."""
        shape = self._panel(
            x, y, w, h, "placeholder", fill=None, line=AMBER, dashed=True
        )
        shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        self._fill(
            shape.text_frame, [label], size=size, colour=AMBER, font=TEXT_FONT,
            align=PP_ALIGN.CENTER,
        )  # fmt: skip

    def pill(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        colour: str = AMBER,
        size: float = 14,
        font: str = TEXT_FONT,
        line: str = RULE,
    ) -> None:
        """A short label in a rounded outline."""
        shape = self._panel(x, y, w, h, "pill", fill=None, line=line)
        frame = shape.text_frame
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        frame.margin_left = frame.margin_right = Inches(0.08)
        frame.margin_top = frame.margin_bottom = Inches(0.02)
        self._fill(
            frame, [label], size=size, colour=colour, font=font, align=PP_ALIGN.CENTER
        )

    def picture(self, path: Path, x: float, y: float, w: float) -> float:
        """A figure from ``assets/`` at width ``w``. Returns its height."""
        shape = self.slide.shapes.add_picture(
            str(path), Inches(x), Inches(y), Inches(w)
        )
        shape.name = f"picture {path.name}"
        return Emu(shape.height).inches

    def link(self, x: float, y: float, w: float, h: float, url: str, *, size: float):
        """A URL in the code face that opens when clicked."""
        frame = self._box(x, y, w, h, "link")
        frame.word_wrap = False
        run = frame.paragraphs[0].add_run()
        run.text = url.removeprefix("https://")
        run.font.size = Pt(size)
        run.font.name = CODE_FONT
        run.font.color.rgb = _rgb(INK)
        run.hyperlink.address = url

    def notes(self, text: str) -> None:
        """The speaker notes."""
        frame = self.slide.notes_slide.notes_text_frame
        if frame is None:  # the default notes master always has a body
            raise ValueError("this slide's notes page has no body to write in")
        frame.text = text


def _glyph_runs(text: str, face: str) -> list[tuple[str, bool]]:
    """``text`` split where ``face`` lacks a glyph, each piece flagged if it does."""
    if face != CODE_FONT:
        return [(text, False)]
    pieces = _CODE_FALLBACK.split(text)
    return [(piece, index % 2 == 1) for index, piece in enumerate(pieces) if piece]


def _code_line(line: str) -> list[Run]:
    """One line of code as runs: amber if the learner writes it, comments muted."""
    if _SLOT.match(line):
        return [Run(line, colour=AMBER, bold=True)]
    if line.lstrip().startswith("#"):
        return [Run(line, colour=MUTED)]
    code, hash_mark, comment = line.partition("  # ")
    if not hash_mark:
        return [Run(line or " ")]
    return [Run(code), Run(f"{hash_mark}{comment}", colour=MUTED)]


class Deck:
    """A 16:9 deck on the workshop's dark background."""

    def __init__(self, title: str, author: str) -> None:
        self.presentation = Presentation()
        self.presentation.slide_width = Inches(SLIDE_WIDTH)
        self.presentation.slide_height = Inches(SLIDE_HEIGHT)
        # The blank template names its own author. The deck names its own.
        properties = self.presentation.core_properties
        properties.title = title
        properties.author = properties.last_modified_by = author
        self._blank = self.presentation.slide_layouts[6]

    def page(self, title: str | None = None, segment: str = "") -> Page:
        """A new slide on the background, titled unless ``title`` is ``None``."""
        slide = self.presentation.slides.add_slide(self._blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _rgb(BACKGROUND)
        page = Page(slide)
        if title is not None:
            page.title(title, segment)
        return page

    def save(self, path: Path) -> None:
        """Write the deck to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.presentation.save(str(path))
