# Build a curious rocket · build plan for Claude Code

Brief for the agent building the workshop material. The spec is `aii_symposium_workshop_scoping.md` rev 20 in the project, the speaker notes are `aii_symposium_slides_notes.md`, the deck is `cpomdp_symposium_2026.pptx` (25 slides, 2026-09-09). Where this plan and the spec disagree, this plan wins and the spec is patched in the same PR.

Dates. Today 2026-09-14. AII submission deadline 2026-10-31. Symposium 12–14 Nov 2026. Freeze 2026-11-09.

---

## 0. Standing rules

- Work in a new repo, `inferogenesis/cpomdp-workshop` (decision D1). Nothing in this plan edits `cpomdp` src. Papercut fixes go through `cpomdp` PRs on their own schedule. This repo works around them on the pinned release.
- Pin `cpomdp==0.4.4` from PyPI. One constant, `WORKSHOP_CPOMDP_VERSION`, read by `pyproject.toml`, the notebook's cell 1 and the tests. Section 8 covers the move to 0.4.5.
- Tests first. Every printed value the deck or spec quotes is asserted by a test before the cell that prints it exists. No number appears in notebook prose unless a check prints it.
- Do not guess the 0.4.4 API. Read `docs/api/` in the synced cpomdp repo and the installed package. Slide 3 tells the room that a model guessing this API produced papercut 1. Do not produce papercut 12.
- Prose in the notebook, README and hint text follows the `CLAUDE.md` prose rules. Hints are code, never prose.
- The deck's generator source was lost with the sandbox on 2026-09-14. The `.pptx` is the source of truth from here. Edits go through `python-pptx` with a render check (`soffice` to PDF, `pdftoppm` to JPEG, look at the touched slides). Make none of the changes listed as undecided in section 9.

## 1. Repo layout

```text
cpomdp-workshop/
  pyproject.toml            # uv project, cpomdp==0.4.4, extras [dev] and [nb]
  README.md                 # uv setup, the shadowing line, Colab link, the no-AI request
  src/cpomdp_workshop/
    __init__.py             # WORKSHOP_CPOMDP_VERSION, public names, hello()
    world.py                # geometry, parameters, reference sensors
    planner.py              # compass_planner
    build.py                # model_with, goal routing, canonical agents, one_node
    sim.py                  # fly(): the loop, the noise draw, the hazard
    check.py                # check_agent, check_my_sensor, check_backends
    plot.py                 # plot_path, plot_world, tiles
    checkpoints.py          # after_session_1/2/3
    solutions.py            # reference and bonus cells
    selfguided.py           # CouplingGraph, warrantlib, p* shape, savemat
  notebooks/build_a_curious_rocket.ipynb
  assets/                   # regenerated figures, plus the crossover gif and png copied with the commit noted
  deck/                     # .pptx, notes.md, extract_notes.py, render.py
  delivery/                 # hints_chat.txt, pre_session_email.md, timer.md, form link
  submission/               # every AII form field as a text block
  tests/
  .github/workflows/ci.yml  # pinned job (required), main-tracking job (allowed to fail)
```

## 2. The reference world, frozen

Every measured claim in the deck came from these values on 0.4.4. Change nothing here without re-running section 5.

**Geometry.** Launch (0, 0), pad (5, 0), nav beacon (2.5, 1.4), blackout (2.5, 0), rock centred on the blackout with radius 0.42.

**Model.** `dynamics = I`, `control = I`, `sensor_model = I`, `dynamics_noise = 1e-3·I`, `sensor_noise = 0.05·I`, prior `Belief(mean=[0, 0], cov=4·I)`. The state-dependent sensor goes in as `observation=CallableSensor(I, R, p)`. Never `sensor_noise=`, which is papercut 1.

**Sensor.** `R(x) = I · (R0 + k·min(‖x − beacon‖², cap) + M·exp(−‖x − blackout‖² / s))` with `R0 = 0.05`, `cap = 4`, `M = 60`, `s = 0.45`. Session 2 drops the M term.

**Goal.** `StateGoal(pad, precision=I)` when k = 0. `ObservationGoal(pad, (−0.5, 0.5), precision=I)` when k > 0. The library enforces the pairing both ways (papercut 2). The scaffold pre-fills it.

**Planner.** `FiniteActionSet` of nine moves, stay plus eight compass directions at step 0.5, `version="c"`. `EnumeratedEfeSearch(model, actions, horizon=1)` inside `RecedingHorizonSelector`. Nine moves. `cue_maze.axis_action_set` yields five for 2-D, and the spec's package section, which names it, is wrong.

**Loop, per step.** `obs = x + sqrt(diag(R(x))) · randn(2)` with R evaluated at the true position (fixed 0.05·I when k = 0). `agent.infer_states(obs)`. `x += agent.sample_action()`. Then the hazard check `‖x − rock‖ < 0.42` on the true x, path truncated at impact. T = 36. Seed 3 by default, one `np.random.seed` per run.

**k.** Session 3 plots k ∈ {0, 0.08, 0.20}, slider default 0.08. Session 2's reference and hint 2 use k = 0.4 (open item 11: at 0.08 the path reads as straight without the blackout).

**Measured on 0.4.4, seed 3.** k = 0 impacts on step 6. The straight line passes through the rock's centre, so every seed does. k = 0.08 arcs over the blackout past the beacon and lands. k = 0.20 lands and dithers. Session 2 at k = 0.4 reaches the beacon's height and ends within 0.25 of the pad. `RecedingHorizonSelector.select` costs about 0.5 s per step after warm-up, so three rockets take about a minute. `plot_world` shows a progress bar.

**Palette for charts.** Background `#0C1916`, panel `#14261F`, ink `#F3F0E6`, muted `#8C968F`, amber series `#C48200`, blue series `#3D8ED4`. The 1960 rocket is dashed ink. Labels sit in ink or muted, never in a series colour. The sensor field is `log10 R(x)` in one muted green ramp under the paths. The deck's slide palette is the same with amber `#E69F00`.

## 3. Package, in build order

Build, test, then the next item. The signatures are the contract the notebook is written against.

1. `world.py`. The constants above. `reference_sensor(x, p)`, `beacon_sensor(x, p)`, `default_params(k)`.
2. `planner.py`. `compass_planner(model, step=0.5, horizon=1)`. Test: nine actions, and `evaluate(...).certificate` prints `PROVED (finite set, |A|^H = 9, visited 9)` for a k > 0 model at horizon 1.
3. `build.py`. `model_with(sensor=None, p=None)`, `the_1960_rocket()`, `curious_rocket(k)`, `one_node(model)` wrapping a `LinearGaussianModel` as a single-node `CouplingGraph`. Goal routing follows sensor type. Test: `probe_model` on the 1960 rocket reports `flattens = True`, `epistemic_varies = False` and a spread of 0.000000 nats across the nine actions. On `curious_rocket(0.08)` it reports `flattens = False` and a spread above 0.01 nats.
4. `sim.py`. `fly(agent, sensor, p, T=36, seed=3, hazard=None) -> Flight(path, covs, crashed, impact_step)`. Tests: each measured fact in section 2 as its own assertion. k = 0 across seeds 0–9 all impact at step 6.
5. `check.py`. `check_agent(agent)`, `check_my_sensor(fn)`, `check_backends(model, *backends)`. Graded messages, never pass/fail. A sensor that varies with position and spreads under 0.01 nats gets "isn't reaching far enough", not a red cross. `check_backends` prints posterior mean, posterior cov and `max |difference|`. Test: `KalmanBackend` and `CouplingGraphBackend(one_node(model))` agree below 1e-9 on the session-1 model.
6. `plot.py`. `plot_path(agent)`, `plot_world(sensor, ks, horizon=1, hazard=(BLACKOUT, 0.42), seed=3)`. Vendor the palette and covariance-ellipse helpers from `examples/gallery.py` with attribution, since examples are not pip-installed. Test: renders headless, crash marker present when k = 0.
7. `checkpoints.py`. `after_session_1()`, `after_session_2()`, `after_session_3()` return the canonical names from the package. Test: a fresh namespace plus a checkpoint runs the next session's reference cell.
8. `solutions.py`. Three reference cells. Bonus cells: momentum (state (x, y, vx, vy), the README point-mass model), Coriolis (flip the sign), rock in clear sky, `horizon=2` in `plot_world`, `policy_efe_trace` along the path, Λ sweep. Test: every bonus cell executes under the pin in under 60 s. `horizon=2` is marked slow.
9. `selfguided.py`. Two-node `CouplingGraph` with the flatten refusal (test: `to_flat_model()` raises `IncompatibleLinearizationError` with "mean-shifting coupling" in the message). The E. coli CheA three-node tree from the gallery. A `warrantlib` example. The p* shape cell with every number labelled *computed, not certified* and a pointer at `cpomdp.harness` on main. `savemat` export plus `trajectories.m`. The surge-margin demo is a pointer to the branch, not vendored.
10. Hints. The six hint lines and the three "reference cell is collapsed below" lines live once, in `delivery/hints_chat.txt`. The notebook's collapsed cells are generated from that file so the paste and the cell never drift.

## 4. Notebook

`notebooks/build_a_curious_rocket.ipynb`, built by a script with `nbformat` from the package, never by hand, so cells regenerate when a solution changes.

- Cell 1: `!pip install cpomdp-workshop -q` with the pin in the comment, then `import cpomdp_workshop as workshop; workshop.hello()`, which prints the cpomdp version and a one-line world summary. Nothing else in that cell.
- Per session, the six-cell rhythm from the spec: brief (markdown), attempt (`arg = ...` on the interesting lines only, goal and planner pre-filled with one comment each), hint 1 and hint 2 as `#@title` collapsed code cells, reference solution `#@title` collapsed, bonus collapsed, checkpoint loud and always run.
- Session 3's `k` is a `# @param` slider, 0 to 0.4 in steps of 0.02, default 0.08.
- Self-guided tail below the last checkpoint, fenced with a markdown banner.
- Pre-run with outputs saved. Shared as `colab.research.google.com/github/inferogenesis/cpomdp-workshop/blob/main/notebooks/build_a_curious_rocket.ipynb`.
- The same notebook runs as plain code under uv: `jupyter nbconvert --to script`, then `python`, sliders inert.
- The uv README carries the line about not naming scripts after stdlib modules (`dis.py`, `types.py`), because the failure is opaque.

## 5. Tests that gate everything

`tests/test_measured_claims.py` is the contract between the code and the deck. One test per quoted value.

- Task 1 probe: `flattens = True`, `epistemic_varies = False`, `spread = 0.000000 nats`.
- Reference beacon sensor: `flattens = False`.
- Certificate string: `PROVED (finite set, |A|^H = 9, visited 9)`.
- k = 0 impacts at step 6, seeds 0–9.
- k = 0.08 and 0.20 clear the rock. k = 0.08 ends within 0.5 of the pad.
- Session 2, k = 0.4, seed 3: max y ≥ 1.0, ends within 0.3 of the pad.
- `check_backends` on the session-1 model: `max |difference|` below 1e-9.
- Flatten refusal message contains "mean-shifting coupling".
- Timing: three rockets at horizon 1 under 120 s on the CI runner, written to a JSON artefact so the trend is visible.
- Papercut 1 regression: `LinearGaussianModel(sensor_noise=CallableSensor(...))` raises `TypeError` on 0.4.4. Marked `xfail(strict=True)` on the main-tracking job, so the day the typed guard lands the test flips and someone notices.

`tests/test_notebook.py` executes the notebook with `nbclient` under the pin on the reference path. It asserts every session has its six cells, every `#@title` is present, every hint line drops into its `...` slot and the cell runs, and every checkpoint restores a namespace that runs the next reference cell.

## 6. Figures and deck assets

Regenerate from the package into `assets/`, never from loose scripts. `fig_world_rock.png` (slide 15: three rockets and the rock, legend upper left, impact star in amber). `tile_task1.png` (the 1960 rocket with belief ellipses after 1, 2, 4 and 8 steps). `tile_task2.png` (k = 0.4 arc over the field). `tile_task3.png` (crop of the world without the rock). `crossover_horizon.gif` and `.png` copied from `docs/assets/` on cpomdp main with the commit noted. Each figure has a generating function and a test that it renders.

Deck: `deck/cpomdp_symposium_2026.pptx` and `deck/notes.md`. `extract_notes.py` regenerates the notes from the notes pane. `render.py` renders a contact sheet. Placeholders still to fill, all Dan's: slide 2 photos and bios, slide 6 Colab link, slide 24 email row, slide 25 Zenodo DOI of the release used.

## 7. Submission, `submission/`

The 2025 form was a Coda form and the call did not publish its fields. Confirm the 2026 fields when the form opens, and keep every answer as a text block in `submission/aii_2026.md` so the form is a paste. Prepare:

- Title: Build a curious rocket. Subtitle: cpomdp · continuous active inference in Python, hands-on.
- Format: workshop, 120 minutes, live. Theme: Open Science & Tool Development.
- Abstract, drafted below. Prerequisite line verbatim from the spec.
- Presenters: names, affiliations, two-line bios, headshots (slide 2 uses the same).
- Links: repo, Colab notebook, docs, the cpomdp channel on the AII Discord, arXiv:2607.20306, Zenodo DOI.
- Open-source outputs promised: the notebook, the package on PyPI, the recording.
- Technical needs: screen share, chat, the anonymous form, no breakout rooms.
- Time preference: a slot the UK afternoon or evening can hold. Both presenters are on GMT in November. Open item 8: Discord says 12–13 Nov and the site says 12–14. Ask.
- Submit by 2026-10-12, not at the deadline. The notebook must run end to end on Colab before submitting, since the form may ask for the link.

Abstract draft. Dan edits, Kev reads once.

> If you have built a continuous active inference agent and watched it behave as a controller, you were not doing it wrong. Under a fixed linear-Gaussian sensor every policy carries the same epistemic value, so expected free energy reduces to LQR (Koudahl, Kouw & de Vries 2021). The controller you got is the 1960 Kalman filter with a goal bolted on.
>
> This hands-on session builds that rocket first, in ten lines of Python with pymdp's shape, then makes it curious with the smallest change that escapes the theorem: sensor noise that depends on where the rocket is. Participants write every line themselves from scaffolds in a Colab notebook, with code hints on a timer and a reference cell for anyone who wants the finished thing. Three builds. A rocket flown to a pad the 1960 way, with the library reporting that no policy can be curious. A nav beacon that bends the flight path. A blackout zone on the flight path with three rockets on one plot, where the 1960 rocket flies blind into a rock it never modelled and the curious rockets clear it. The session closes with the same rocket on three backends, RxInfer among them, and a short coda on what a printed number entitles you to say. Everything shown runs on the PyPI release of cpomdp. The notebook and package are open source.
>
> Prerequisites: comfortable editing a few lines of Python. Familiarity with pymdp helps but is not required. Nothing to install; a browser is enough.

## 8. Moving to 0.4.5

0.4.5 is expected before the symposium. The build does not wait for it.

- CI runs two jobs. `pinned` on 0.4.4, required. `main` on `cpomdp @ git+https://github.com/inferogenesis/cpomdp@main`, allowed to fail, nightly, posting the measured-claims diff as an artefact. Breakage shows up weeks before the re-pin.
- Re-pin only if 0.4.5 is on PyPI by 2026-11-01. After that date the workshop ships on 0.4.4 whatever lands.
- Re-pin checklist, one PR: bump the constant. Run section 5 and read every diff in the printed strings (certificate wording, `probe_model` fields, error messages). Regenerate section 6. Re-time `plot_world`, and if papercut 4 is fixed update the progress-bar note and the slide 9 notes. Flip the papercut 1 xfail if the typed guard shipped. Re-run the notebook on Colab. Update the pin on slide 1, slide 6 and the last-slide caption.
- The disclosure consequence, decision D2. 0.4.5 will contain src written by Claude Code under the post-0.4.4 regime. Slide 4's caption ("today's notebook pins 0.4.4, the last release before that line") and the Q&A answer to *did Claude write this* ("for 0.4.4, no") hold only under the 0.4.4 pin. A re-pin to 0.4.5 rewrites both: the caption goes, and the answer for the release the room is running becomes "some of it, under the review described". The re-pin decision carries that rewrite with it.

## 9. Decisions Dan owes. Build nothing under them until decided

- D1 Repo location. A separate `cpomdp-workshop` repo (recommended: independent pin, independent CI), or `examples/workshop/` in the monorepo.
- D2 Pin for November. Stay on 0.4.4 (disclosure story intact, papercuts worked around), or re-pin to 0.4.5 (fixes, slide 4 rewritten). Default in this plan: re-pin if on PyPI by 1 Nov.
- D3 Slide 7. Keep "What continuous state buys you", or replace it with the four-tile "What you build today" (built and rendered on 2026-09-09, unshipped).
- D4 Early install. Colab link and `pip install` on slide 2 at 0:00, so task 1 starts near 0:10.
- D5 From the workshop-design research, unbuilt. Each is a separate yes or no.
  - One diagnostic poll per task boundary.
  - A "paste one number in chat" close to each task.
  - A symptoms cell, and one reveal opened from the papercut 1 error.
  - The anonymous question form, plus one-up-one-down at the buffer and the close.
  - Naming the buffer a break.
  - Building the world figure and the factor graph in steps on the slides.
  - Two learner personas checked against every scaffold comment.
  - A one-page session handbook.
  - Capping the Inferogenesis tail at five minutes.
- D6 Sponsors URL on slide 25, or notes only.
- D7 Session 2 bend. k = 0.4 in the reference and hint 2 (the default here), or move the beacon toward the flight path and keep 0.08.

## 10. Order and dates

| By | Milestone | Gate |
| --- | --- | --- |
| 21 Sep | Repo, world, planner, build, sim. Section 5 measured-claims tests green on 0.4.4 | tests |
| 28 Sep | Checkers, plotters, checkpoints. Figures regenerated, tiles rendered | figures match the deck's |
| 5 Oct | Notebook runs end to end on Colab and under uv. README. Hints file | `test_notebook.py` |
| 12 Oct | Submission sent with the Colab link | form confirmation |
| 19 Oct | Self-guided tail. Delivery artefacts. RxInfer clip recorded (Dan, not the agent) | clip exists |
| 26 Oct | Dry run 1, solo | timings written down |
| 1 Nov | Re-pin window closes | D2 executed or closed |
| 2–8 Nov | Dry runs 2 and 3, one with live handoffs. Kev has run the notebook once | handoff token agreed |
| 9 Nov | Freeze. Typo fixes only after this | tag `v2026.11` |
