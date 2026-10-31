# Build a curious rocket

A two-hour hands-on workshop on [cpomdp](https://github.com/inferogenesis/cpomdp),
continuous active inference in Python, for the 6th Applied Active Inference Symposium.
You build three rockets. The first is the 1960 controller, a Kalman filter with a goal,
and the library reports that no move can teach it more than another. The second reads its
position through a sensor that depends on where it is, and bends its path toward a
beacon. In the third, three rockets fly past a rock none of them was told about.

## Open it

The live session runs on Colab. There is nothing to install. A browser and a Google
account are enough. Edits vanish when the tab closes unless you save a copy to Drive.

[Open the notebook in Colab](https://colab.research.google.com/github/inferogenesis/cpomdp-workshop/blob/main/notebooks/build_a_curious_rocket.ipynb)

Please write the lines yourself rather than asking a generative AI tool for them. The
point of the session is the two lines you work out. The hints and the reference cells
are there for when you are stuck. Using them is fine.

## Run it locally

With [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/inferogenesis/cpomdp-workshop
cd cpomdp-workshop
uv sync --extra nb
uv run jupyter lab notebooks/build_a_curious_rocket.ipynb
```

Python 3.11 or newer. The workshop pins the cpomdp release it was measured on, and the
first cell stops if a different one is installed.

Do not name a file in the folder you run from after a standard-library module, such as
`types.py`, `dis.py` or `random.py`. Python imports yours instead. The error that
follows points nowhere near the cause.

## What is in the repo

| Path | What it holds |
| --- | --- |
| `src/cpomdp_workshop/` | The package the notebook imports: the world, the two rockets, the flight loop, the checks, the figures |
| `notebooks/` | The notebook, generated and run by `scripts/build_notebook.py`. Never edited by hand |
| `scripts/lesson.py` | What the notebook says, cell by cell |
| `delivery/hints_chat.txt` | The hint and reference lines. The notebook's collapsed hint and reference cells are built from it |
| `assets/` | Figures, regenerated from the package by `scripts/render_assets.py` |
| `deck/` | The slides, built from the package, the lesson and the figures by `scripts/build_deck.py`. The committed deck has an empty frame for each presenter's photo. With the photos in `deck/photos/`, the build also writes a presenting copy that holds them. Git ignores both |
| `tests/test_measured_claims.py` | One test per number the notebook or the deck shows |
| `tests/test_api_contract.py` | The cpomdp forms the workshop is written against |

## Working on it

Every command runs from the repo root.

```sh
uv sync --extra dev --extra nb --extra deck
uv run --no-sync pytest -q -m "not slow"     # fast tests
uv run --no-sync pytest -q                   # everything, the notebook run included
uv run --no-sync pre-commit run --all-files  # the gate CI runs
uv run --no-sync python scripts/build_notebook.py
uv run --no-sync python scripts/render_assets.py
uv run --no-sync python scripts/build_deck.py
```

A change to the lesson, the hints or the package means rebuilding the notebook. A stale
notebook fails a test.

The deck is built the same way, after the figures. A stale deck fails a test too. The
build checks that no shape leaves its slide or overlaps another and that every text
frame holds its text. It renders nothing, so a look at the slides in PowerPoint or
LibreOffice is still owed after a change.

## Releasing

The notebook's first cell installs the exact version it was built from, so bump
`version` in `pyproject.toml`, rebuild the notebook and merge. Then publish a GitHub
release tagged `v` plus that version. `.github/workflows/publish.yml` runs the tests,
installs the wheel into a clean environment and uploads to PyPI through Trusted
Publishing.

## Licence

MIT. See `LICENSE`.
