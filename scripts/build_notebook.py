"""Build the workshop notebook from the lesson and the hint lines, then run it.

    uv run --no-sync python scripts/build_notebook.py           # build and run
    uv run --no-sync python scripts/build_notebook.py --no-run  # build only

The notebook is generated and never edited by hand. Each reference cell is its
session's attempt with both hint lines put in, so the attempt, the hints pasted into
chat and the reference cannot disagree. Running it saves the outputs of every cell
except the attempts and the hints, which a learner fills and runs themselves.
"""

import argparse
import re
from importlib import metadata
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

import lesson

ROOT = Path(__file__).resolve().parents[1]
HINTS = ROOT / "delivery" / "hints_chat.txt"
NOTEBOOK = ROOT / "notebooks" / "build_a_curious_rocket.ipynb"

SKIP = "skip-execution"
"""The tag nbclient skips: attempts hold ``...``, hints are for copying, and bonus code
is for the learner to run."""

_HEADER = re.compile(
    r"^\[session (?P<session>\d), (?P<kind>(?:hint|nudge) [12]|reference)\]$"
)
_SLOT = re.compile(r"^(?P<indent>\s*)(?P<name>\w+) = \.\.\.(?:\s+#.*)?$")


def load_hints(path: Path = HINTS) -> dict[tuple[int, str], str]:
    """The hint, nudge and reference lines, keyed by session and kind.

    Args:
        path: the hints file. Blocks are headed ``[session N, hint 1]``, ``[session
            N, hint 2]``, ``[session N, nudge 1]``, ``[session N, nudge 2]`` or
            ``[session N, reference]``. A hint's nudge is optional. Text before the
            first header is a note to the presenter and is skipped.

    Returns:
        Each block's text, stripped.
    """
    blocks: dict[tuple[int, str], list[str]] = {}
    key = None
    for line in path.read_text().splitlines():
        header = _HEADER.match(line.strip())
        if header:
            key = (int(header["session"]), header["kind"])
            blocks[key] = []
        elif key is not None:
            blocks[key].append(line)
    return {key: "\n".join(lines).strip() for key, lines in blocks.items()}


def slots(source: str) -> list[str]:
    """The names of the ``name = ...`` lines a learner still has to write."""
    return [m["name"] for line in source.splitlines() if (m := _SLOT.match(line))]


def fill(attempt: str, hint: str) -> str:
    """``attempt`` with the slot ``hint`` assigns replaced by ``hint``.

    Raises:
        ValueError: unless exactly one slot carries the name the hint assigns.
    """
    name = hint.split("=", 1)[0].strip()
    lines = attempt.splitlines()
    found = [
        i
        for i, line in enumerate(lines)
        if (m := _SLOT.match(line)) and m["name"] == name
    ]
    if len(found) != 1:
        raise ValueError(f"the hint for {name!r} matches {len(found)} slots, not one")
    slot = _SLOT.match(lines[found[0]])
    assert slot is not None  # found above
    lines[found[0]] = slot["indent"] + hint
    return "\n".join(lines)


def reference(session: lesson.Session, hints: dict[tuple[int, str], str]) -> str:
    """The session's attempt with both hints put in.

    Raises:
        ValueError: if a slot is left over once both hints are in.
    """
    source = session.attempt
    for kind in ("hint 1", "hint 2"):
        source = fill(source, hints[(session.number, kind)])
    if slots(source):
        raise ValueError(f"session {session.number} leaves {slots(source)} unwritten")
    return source


def _markdown(source: str, tag: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(source, metadata={"tags": [tag]})


def _code(source: str, *tags: str) -> NotebookNode:
    return nbformat.v4.new_code_cell(source, metadata={"tags": list(tags)})


def _collapsed(title: str, source: str, *tags: str) -> NotebookNode:
    """A code cell Colab shows as its title alone, and JupyterLab folds."""
    cell = _code(f"# @title {title}\n{source}", *tags)
    cell.metadata["cellView"] = "form"
    cell.metadata["jupyter"] = {"source_hidden": True}
    return cell


def _hint(number: int, which: int, hints: dict[tuple[int, str], str]) -> NotebookNode:
    """A hint cell: its nudge shown under the title, its line behind Show code."""
    source = hints[(number, f"hint {which}")]
    nudge = hints.get((number, f"nudge {which}"))
    if nudge is not None:
        shown = "\n".join(f"# @markdown {line}" for line in nudge.splitlines())
        source = f"{shown}\n{source}"
    return _collapsed(f"Hint {which}", source, "hint", SKIP)


def _bonus(which: int, bonus: lesson.Bonus) -> list[NotebookNode]:
    """A bonus's text cell, and the code cell under it when it has one.

    The text shows the question in full, then a hint and an answer, each folded until
    opened. Colab needs the blank line after each ``<summary>`` line, or the text
    inside shows as raw HTML.
    """
    text = (
        f"### Bonus {which} · {bonus.title}\n\n{bonus.question}\n\n"
        f"<details><summary>Hint</summary>\n\n{bonus.hint}\n\n</details>\n\n"
        f"<details><summary>Answer</summary>\n\n{bonus.answer}\n\n</details>"
    )
    cells = [_markdown(text, "bonus")]
    if bonus.code is not None:
        cells.append(_code(bonus.code, "bonus-code", SKIP))
    return cells


def _collapsed_symptoms() -> NotebookNode:
    title, source = lesson.SYMPTOMS.split("\n", 1)
    return _collapsed(title.removeprefix("# @title "), source, "symptoms")


def build(hints: dict[tuple[int, str], str] | None = None) -> NotebookNode:
    """The whole notebook, cells in order, without outputs.

    Args:
        hints: the hint lines, read from ``delivery/hints_chat.txt`` when ``None``.

    Returns:
        The notebook.
    """
    hints = load_hints() if hints is None else hints
    version = metadata.version("cpomdp-workshop")
    cells = [
        _markdown(lesson.TITLE, "title"),
        _code(lesson.INSTALL.format(version=version), "install"),
        _markdown(lesson.PREVIEW_INTRO, "preview"),
        _code(lesson.PREVIEW, "preview"),
        _markdown(lesson.PREFLIGHT, "preflight"),
        _collapsed_symptoms(),
    ]
    for session in lesson.SESSIONS:
        n = session.number
        cells += [
            _markdown(session.brief, "brief"),
            _code(session.attempt, "attempt", SKIP),
            _collapsed_symptoms(),
            _hint(n, 1, hints),
            _hint(n, 2, hints),
            _collapsed(f"Reference: task {n}", reference(session, hints), "reference"),
            *(
                cell
                for which, bonus in enumerate(session.bonuses, start=1)
                for cell in _bonus(which, bonus)
            ),
            _code(lesson.CHECKPOINT.format(number=n), "checkpoint"),
            _markdown(session.poll, "poll"),
        ]
    cells += [_markdown(lesson.CODA_INTRO, "coda"), _code(lesson.CODA, "coda")]
    cells.append(_markdown(lesson.TAIL_INTRO, "self-guided"))
    cells += [_collapsed(title, source, "self-guided") for title, source in lesson.TAIL]
    # nbformat draws a random id per cell. A position and a tag keep a rebuild's diff
    # down to the cells that changed.
    for index, cell in enumerate(cells):
        cell["id"] = f"{index:02d}-{cell.metadata['tags'][0]}"
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    notebook.metadata["colab"] = {"name": NOTEBOOK.name, "provenance": []}
    return notebook


def run(notebook: NotebookNode) -> NotebookNode:
    """Execute every cell not tagged ``skip-execution``, in place, and return it."""
    NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=600,
        skip_cells_with_tag=SKIP,
        record_timing=False,
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    return notebook


def main() -> None:
    """Build the notebook, run it unless told not to, and write it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-run", action="store_true", help="build without running")
    arguments = parser.parse_args()
    notebook = build()
    if not arguments.no_run:
        run(notebook)
    NOTEBOOK.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, NOTEBOOK)
    print(f"wrote {NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
