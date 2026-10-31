# Build a curious rocket · build spec v2, on cpomdp 0.5.0

Supersedes `aii_symposium_build_plan.md` (2026-09-14). Written 2026-10-16 against `main`, re-verified 2026-10-17 against `cpomdp==0.5.0` installed from PyPI: every signature in section 3 was read with `inspect` and every number in section 4 was re-measured. Where the two disagree, this one wins.

Sources for the pedagogy: Software Carpentry's lessons-learned paper, fast.ai's teaching philosophy, the Django Girls organiser manual, Wilson's lesson-design tips, the Carpentries instructor training, SciPy's and NeurIPS's tutorial calls. Sources for the API: the 0.5.0 section of `CHANGELOG.md`, `README.md`, `docs/api/agent.md`, all on `main`.

Dates. Symposium 12–14 Nov 2026. Freeze 2026-11-09. Today 2026-10-16.

---

## 0. What the evidence says, and what each point changes here

| What worked, and where it was measured | What it changes in this build |
| --- | --- |
| Watching code emerge beats slides. Software Carpentry, 400+ workshops, 12,000 learners: "watching code emerge on the screen is much more convincing than looking at pre-made slides", and learners rated seeing instructors' mistakes and fixes highest. | DanBoringName still writes no code live. One reveal per session opens from a real error (papercut 1's `TypeError`), read aloud and fixed. A `symptoms` cell ships the top five errors with their fixes. |
| The whole game first. fast.ai: a working result "within the first 30 minutes", then dig down. David Perkins's "elementitis" is what loses people. | Cell 2 is `workshop.preview()`: the finished three-rocket plot with the asteroid, rendered from the package in under a minute, before anyone builds anything. The session then rebuilds that picture in three steps. |
| Setup is the first failure. Django Girls runs an "installation party" the evening before and says attendees' machines being ready is the most important thing. Carpentries sends setup days ahead and opens early. | Colab link and `pip install` on the speaker slide at 0:00. A 30-minute drop-in on the cpomdp Discord channel the evening before. The uv README a week ahead. Zoom opens 15 minutes early for triage. |
| Attention has a ceiling. Carpentry's week-long format failed because "by the last two days, attendees' brains are full". Its two-day format with ~50% hands-on is what scaled. SciPy asks for "at least 50% of the time for hands-on exercises". | Two hours: an hour of talk, 45 minutes of code in three 15-minute builds, and 15 minutes left open at the end. No scheduled break. The closing section on why cpomdp matters runs 12 minutes. |
| Formative checks every 10–15 minutes, from every learner, acted on at once. Wilson 2019. Carpentry's red/green sticky notes. | One diagnostic multiple-choice poll at each task boundary, 60 s, wrong answers targeting real misconceptions. Each task closes with "paste one number in chat" (`spread`, your `k`, `max \|difference\|`), so progress is visible without anyone composing a sentence. Zoom ✅/✋ for the quiet signal. |
| Helpers at 1:8, coaches at 1:3 for novices. Carpentry aims for "at least one helper for every eight learners". Django Girls recommends 3:1 attendee to coach. | Two presenters cover sixteen. Above that, recruit one helper per eight from the Discord channel, briefed on the symptoms cell only, answering in chat. Above forty, the reference cell carries the overflow and the notebook says so. |
| Faded examples and scaffolds, with the reference path legitimate. Wilson's worked-example fading. Carpentries' "the reference cell is not failure". | Unchanged: `arg = ...` scaffolds, two code hints on a timer, collapsed reference cell, loud checkpoint. |
| Pre-assessment sorts the room. Carpentry streams learners by a pre-workshop questionnaire. | Not adopted. Registration asks no questions: a questionnaire is invasive and the workshop does not need one. The task-1 scaffold keeps the gloss lines it has. |
| Materials outlive the session. NeurIPS wants notebooks that stay useful. Carpentry's MOOC and video attempts failed on maintenance cost. | The notebook is built from the package by a script, runs from the recording with no hints on a timer, and the package is on PyPI. No videos. |
| Anonymous channels get the questions the chat does not. Carpentry's minute cards. The virtual-course rule set's living Q&A document. | One anonymous form linked from the notebook header and every task slide. One-up-one-down at the break and at the close. The survey link on the last slide while the room is still in it. |

## 1. Standing rules

- Repo `inferogenesis/cpomdp-workshop`, its own pin, its own CI. Nothing here edits cpomdp src. A papercut found while building goes to a cpomdp issue and is worked around in this repo on the pin.
- Pin `cpomdp==0.5.0` from PyPI (published 2026-10-17, `warrantlib` 0.3.0 with it). One constant, `WORKSHOP_CPOMDP_VERSION`, read by `pyproject.toml`, cell 1 and the tests. No SHA pinning.
- Tests first. Every value the deck quotes is asserted before the cell that prints it exists. No number appears in notebook prose unless a check prints it.
- Do not guess the API. Field names, constructor forms and report fields changed in 0.5.0 and section 3 lists what changed. Read `docs/api/` and the installed package before writing a line that touches cpomdp. Slide 3 tells the room a model guessing this API produced papercut 1. The build must not produce papercut 12.
- `cpomdp._lab` is private (ADR-069). The self-guided p\* cell imports from it, says so in its first line, and is the only place that does.
- Prose in the notebook, README and hints follows the repo `CLAUDE.md` prose rules. Hints are code, never prose.
- The deck is generated by `scripts/build_deck.py` and never edited by hand. Section 8 describes it. After a change, render it in PowerPoint or LibreOffice and look at every slide that changed.

## 2. Repo layout

Unchanged from v1 except the pin and one new module.

```text
src/cpomdp_workshop/
  __init__.py        # WORKSHOP_CPOMDP_PIN (SHA or version), hello(), preview()
  world.py  planner.py  build.py  sim.py  check.py  plot.py
  checkpoints.py  solutions.py  selfguided.py
  symptoms.py        # the five errors the room will hit, each with its fix, runnable
notebooks/build_a_curious_rocket.ipynb   # generated, never hand-edited
delivery/  deck/  submission/  assets/  tests/  .github/workflows/ci.yml
```

## 3. API delta, v1 plan → 0.5.0

Every row was wrong or missing in the September plan. The right-hand column is what the package and notebook use.

| Topic | September plan said | 0.5.0 |
| --- | --- | --- |
| Model fields | `dynamics=`, `control=`, `sensor_model=`, `sensor_noise=`, `observation=` | `dynamics_matrix=`, `control_matrix=`, `observation_matrix=`, `dynamics_noise=`, `observation_noise=`, `observation_model=CallableSensor(...)`. Beside a sensor, `observation_matrix` and `observation_noise` are optional and read from the sensor. A value passed beside the sensor must equal the sensor's at `prior.mean` or construction raises (ADR-064, ADR-071). |
| Papercut 1 | sensor passed into the noise slot gives a raw `TypeError` | Fixed. `observation_noise=CallableSensor(...)` now raises `TypeError: observation_noise takes a matrix, got CallableSensor, an ObservationModel. Pass it as observation_model.` The regression test asserts that message. Slide 3's line stays true as history, and the symptoms cell shows this exact text. |
| The 1960 rocket | `Agent(model, StateGoal(pad))` | `StateGoal(pad, precision=I, effort=100·I)`. With the default effort the LQR gain is 0.618 and the rocket crosses the rock in one step. At `effort = 100·I` the gain is 0.0951 and the impact lands on step 6 for every seed 0–9. The effort is the thruster cost and the brief says so in one line. `A · target = target` holds with `A = I`. |
| The curious rocket | `Agent(model, ObservationGoal(...), selector=compass_planner(model))` | A goal beside `selector=` is refused (ADR-080). Pass `Agent(model, Preference(pad, precision=I), selector=compass_planner(model))`. The `Preference` is read in observation space by the search drivers. |
| Compass planner | `RecedingHorizonSelector(EnumeratedEfeSearch(model, actions, horizon=1))` | The classes are `EnumeratedEFESearch`, `ChunkedEFESearch`, `ExhaustiveEFESearch` and `FFGEFESelector`, upper-case EFE. The changelog prose spells them otherwise and the code wins. `EnumeratedEFESearch(model, action_set, horizon=1)`, `RecedingHorizonSelector(search)`, `search.evaluate(belief, preference)`. Measured: 36 steps at nine policies in 0.4 s, three rockets in about 1.5 s. The progress bar goes. Horizon 2 runs six steps in 0.8 s and `ChunkedEFESearch` at horizon 3 the same, so the trapped-agent-escapes bonus is a live beat. |
| Backend swap | `CouplingGraphBackend(one_node(model))` passed to `Agent(..., backend=)` | `backend=` is gone. `CouplingGraph(dims=[2], couplings=[], observation_models={0: FixedSensor(I, observation_noise=0.05·I)})`, then `CouplingGraphBackend(graph, [GaussianTransition(I, dynamics_noise=1e-3·I)], prior=prior, control_matrix=I)`, then `Agent.from_backend(backend, goal)`. `infer_states(observation, prior, action)` on both backends. Measured agreement with `KalmanBackend` on one step: 7e-16 on mean and covariance. `infer_root(readings, prior)` and `stack_observations` exist on the graph path. |
| Probe | `SensorReport` with `flattens`, `epistemic_varies`, `epistemic_range`, `noise_spread` | `probe_model(model, belief, actions)` → `ProbeReport` with fields `observation_noise_varies`, `observation_noise_spread`, `dynamics_noise_varies`, `dynamics_noise_spread`, `epistemic_varies`, `epistemic_range`, plus `flattens` as a property and `summary()` as a method. The deck's `spread = 0.000000 nats` is `max − min` of `epistemic_range`. The 1960 rocket measures (4.3947, 4.3947), the k = 0.08 rocket (2.4490, 2.4692). |
| RxInfer beat | any model | Dropped. The workshop shows cpomdp, so nothing runs or prints RxInfer. Slide 18 may name it in a footnote. |
| Agent introspection | none | `print(agent)` gives `Agent(objective=StateGoal, selector=LQRSelector(controller=LQRController(gain=[[0.0951, 0.0], [0.0, 0.0951]])), backend=KalmanBackend)`. `check_agent` prints it. `agent.selector.cost_per_cycle` reads 9 at horizon 1 and 162 at horizon 2. |
| Model mutation | checkpoints rebuilt agents by assignment | `agent.model`, `backend.model` are read-only. Checkpoints build fresh objects. |
| Python | ≥ 3.10 | ≥ 3.11, `jax >= 0.4.36`. Installs clean on 3.13 with jax 0.11.2. Cell 1 asserts both and prints them. Check the Colab default runtime at build time. |
| Research apparatus | `cpomdp.harness`, `cpomdp.constructors`, `cpomdp.backends.degraded` as public | All under `cpomdp._lab`, private. Self-guided cell only. |
| Disclosure (slide 4) | "the notebook pins 0.4.4, the last release before that line" | 0.5.0 contains src written under the post-0.4.4 regime. The caption becomes: "today's notebook pins 0.5.0. Some of its src was written by Claude Code under the review described." The Q&A answer for the release the room runs becomes "some of it". Decision D2 from v1 is closed by this spec. |

| Certificate text | `PROVED (finite set, \|A\|^H = 9, visited 9)` | `PROVED (set compass-9, \|A\|^H = 9^1 = 9, visited 9)`. The set's `version` string is printed, so name it `compass-9`. The coda's 78125 line takes the same form. |
| EFESelector | 1-D only | Still 1-D. The error now names the path: `EFESelector searches a 1-D action grid (p=1); got p=2. For p>1 use EnumeratedEFESearch with a declared FiniteActionSet`. That is what papercut 3 asked for. |

Two corrections that are not 0.5.0's doing. The pymdp name map on slide 6 was wrong on pymdp's side: `A` is the observation likelihood, `B` the transition, `C` the preference, `D` the prior. The map is `A → observation_matrix`, `B → dynamics_matrix` with `control_matrix`, `C → StateGoal` or `Preference`, `D → prior`. And `compass_planner` is nine moves (stay plus eight compass directions at step 0.5), not `cue_maze.axis_action_set`, which gives five in 2-D.

## 4. The reference world

Geometry, parameters, seeds and k values are unchanged from v1 section 2 except the 1960 rocket's effort, which is new. Measured on 0.5.0, seed 3: k = 0 impacts on step 6 (every seed 0–9 at `effort = 100·I`). k = 0.08 clears the rock, peaks at y = 1.71 and ends at (4.83, 0.09). k = 0.20 clears, peaks at 1.91 and ends at (5.18, 0.15). Session 2 at k = 0.4 peaks at 1.41 and ends at (5.12, 0.21). If any claim moves again, the deck follows the measurement.

Launch (0, 0), pad (5, 0), beacon (2.5, 1.4), blackout (2.5, 0), rock radius 0.42 on the blackout. `R(x) = I · (R0 + k·min(‖x − beacon‖², cap) + M·exp(−‖x − blackout‖² / s))`, `R0 = 0.05`, `cap = 4`, `M = 60`, `s = 0.45`. Model: `LinearGaussianModel(I, control_matrix=I, dynamics_noise=1e-3·I, prior=Belief([0, 0], 4·I), observation_model=CallableSensor(I, R, p))`, with `observation_matrix=I, observation_noise=0.05·I` in place of the sensor for the 1960 rocket. 1960 goal `StateGoal(pad, precision=I, effort=100·I)`. Curious goal `Preference(pad, precision=I)` with `selector=RecedingHorizonSelector(EnumeratedEFESearch(model, COMPASS, horizon=1))`, `COMPASS = FiniteActionSet(stay + 8 directions at 0.5, version="compass-9")`. Observations drawn at the true state, hazard checked on the true state, T = 36, seed 3. Session 3 `k ∈ {0, 0.08, 0.20}`. Session 2 reference `k = 0.4`.

## 5. Package, in build order

1. `world.py`: constants, `reference_sensor`, `beacon_sensor`, `default_params(k)`.
2. `planner.py`: `compass_planner(model, step=0.5, horizon=1)`. Test: nine actions. Certificate string `PROVED (set compass-9, |A|^H = 9^1 = 9, visited 9)`. One trace across two plans (`jax` trace counter, the pattern `tests/test_search_compilation.py` uses upstream).
3. `build.py`: `model_with(sensor=None, p=None)` using the 0.5.0 field names, `the_1960_rocket()` with `effort=100·I`, `curious_rocket(k)`, `one_node(model)` returning the `CouplingGraph` and `CouplingGraphBackend` pair from section 3. Test: `probe_model` on the 1960 rocket reports `flattens = True` and a spread of 0.000000 nats. On `curious_rocket(0.08)`, `flattens = False` and a spread above 0.01 nats.
4. `sim.py`: `fly(agent, sensor, p, T=36, seed=3, hazard=None) -> Flight`. Tests: k = 0 impacts at step 6 across seeds 0–9. k = 0.08 and 0.20 clear the rock. Session 2 at k = 0.4 reaches the beacon's height and ends within 0.3 of the pad.
5. `check.py`: `check_agent` (prints the agent, then arrival), `check_my_sensor` (graded), `check_backends(model, *backends)` (mean, cov, `max |difference|`). Test: Kalman and one-node graph agree below 1e-9.
6. `plot.py`: `plot_path`, `plot_world`, the tiles, and `preview()` for cell 2. Palette from v1. Test: renders headless. Crash marker at k = 0. `preview` under 60 s on the CI runner.
7. `symptoms.py`: the five errors, each a function that raises it and a line that fixes it: the noise slot beside a sensor. A goal beside `selector=`. `StateGoal` under `R(x)`. A `Preference` of the wrong length. `agent.model = ...`. Test: each raises the named error class and the fix runs.
8. `checkpoints.py`: `after_session_1/2/3`, building fresh objects. Test: a fresh namespace plus a checkpoint runs the next reference cell.
9. `solutions.py`: three reference cells. Bonus cells (momentum, Coriolis, clear-sky rock, `horizon=2` through `ChunkedEfeSearch`, `policy_efe_trace`, Λ sweep). Test: each runs under 60 s except `horizon=2`, marked slow.
10. `selfguided.py`: two-node `CouplingGraph` with the flatten refusal. CheA tree. A `warrantlib` example with `record_check`. The p\* shape cell on `cpomdp._lab.harness` labelled private and *computed, not certified*. `savemat` plus `trajectories.m`. Test: `to_flat_model()` raises `IncompatibleLinearizationError` naming the mean-shifting coupling.
11. `delivery/hints_chat.txt`: the six hint lines and three reference-cell lines, the one place they live. The notebook's collapsed cells are generated from it.

## 6. Notebook

Generated by `scripts/build_notebook.py` with `nbformat`. Cells, in order:

1. Install and assert: pin, Python ≥ 3.11, `jax` version, `workshop.hello()`. Nothing else.
2. `workshop.preview()`. The finished picture. One markdown line above it: *this is what you will have built by 1:32.*
3. Per session, the six-cell rhythm: brief, attempt (`arg = ...`, goal and planner pre-filled with one comment each), hint 1, hint 2, reference, bonus, checkpoint. The attempt cell ends with the one number to paste in chat.
4. After each checkpoint, the poll cell: a markdown cell with the question and four answers, and the Zoom poll mirrors it.
5. `workshop.symptoms()` as a collapsed cell under each session, and once in the header.
6. Self-guided tail below the last checkpoint, fenced.

Session 3's `k` is a `# @param` slider, 0 to 0.4 in steps of 0.02, default 0.08. The notebook runs as a script under uv with sliders inert. Pre-run with outputs saved. Shared from GitHub through Colab.

## 7. Tests that gate everything

`tests/test_measured_claims.py` carries one assertion per quoted value: task-1 probe strings, certificate string, impact step, clearances, session-2 arc, backend agreement, flatten-refusal message, timing of `plot_world` and `preview` recorded to JSON. `tests/test_api_contract.py` asserts the 0.5.0 forms in section 3: field names accepted, old names refused, goal-beside-selector refused, `from_backend` present, `probe_model` report fields. This file is what fails first on the next cpomdp release. `tests/test_notebook.py` executes the generated notebook under `nbclient` on the reference path and asserts the cell rhythm, every hint dropping into its slot, every checkpoint restoring a namespace that runs the next reference cell.

CI: `pinned` job on the SHA or version, required. `main` job tracking cpomdp `main` nightly, allowed to fail, posting the measured-claims diff.

## 8. The deck

Remade on 2026-10-28 from the 0.5.0 deck, on the light theme of Inferogenesis's site. `scripts/build_deck.py` builds it from the package, `lesson.py` and the route frames, and one agenda table in it times every slide. `tests/test_deck.py` fails if the committed `.pptx` is not what the script builds. 31 slides:

| Part | Starts, length | Slides |
| --- | --- | --- |
| Welcome | 0:00, 12 min | 1 title, 2 who we are, 3 disclosure, 4 the two hours, 5 how we'll work, 6 get set up (Colab or local Python), 7 where we're heading |
| Task 1 | 0:12, 27 min | 8 brief with pymdp's letters, 9 code, 10 timer, 11 the straight line, 12 under a fixed sensor EFE is LQR |
| Task 2 | 0:39, 27 min | 13 brief, 14 code, 15 timer, 16 the beacon bends the route, 17 what steers the route |
| Task 3 | 1:06, 27 min | 18 brief, 19 code, 20 timer, 21 the blackout, 22 the rock, 23 a rock never changes a route, 24 one rocket on two engines |
| Why it matters | 1:33, 12 min | 25 certificates, 26 a working model is the beginning, 27 what makes cpomdp different, 28 already built on cpomdp, 29 where the research goes next |
| Questions | 1:45, 15 min | 30 keep going, 31 thank you |

Each task is a 4-minute brief, 15 minutes of code and 8 minutes of discussion. The timer slide stays up while the room codes: its bar fills over 15 minutes from when the slide shows, with no click, and then tells the room to run the checkpoint. Leaving the slide restarts the bar.

RxInfer is named once, on slide 27, as a neighbour. Nothing runs it. The reveal of papercut 1 has no slide. The timer slides point at `workshop.symptoms()`.

Edits still DanBoringName's: slide 2 photos and bios, slide 3 the AI-use panel, slide 29 The Individuating Membrane (I found no source for it), slide 30 the Zenodo DOI of 0.5.0 once minted, slide 31 the survey link and email. The lost deck's "What comes next" slide carried the status "scoped, starts after p*". Slide 29's notes ask which programme it belongs to.

## 9. Delivery, before, during, after

Before. The uv README and Colab link a week ahead on the cpomdp Discord channel, with the no-generative-AI request in the same message. A 30-minute install drop-in on Discord the evening before. Helpers recruited at one per eight registered above sixteen, briefed on `symptoms` only. Zoom open 15 minutes early.

During. Install at 0:00 while Kev frames. Task 1 at 0:12, task 2 at 0:39, task 3 at 1:06, each with its timer slide up for the 15 minutes of code. Hints pasted on the timer from `hints_chat.txt`. One number in chat at each task close. One poll at each boundary, 60 s, Kev reads the split. Why it matters at 1:33. Questions from 1:45, and anyone may leave early. No scheduled break. If the room needs one, it comes out of the open time. Anonymous form watched by whoever is not presenting. No breakout rooms. No individual debugging from the front.

After. Survey link on slide 25 while the room is in it. Recording, notebook, package, slides and notes posted to the Discord channel and the README within 48 hours. One-up-one-down answers read into the next revision of the notebook.

## 10. Decisions still owed

- D1 Repo location (default: separate repo).
- D3 Slide 7, keep or replace with the four tiles (the `preview` cell makes the tile slide less necessary. Recommend keep).
- D6 Sponsors URL on slide 25 or notes only.
- D8 The asteroid in the abstract: keep the spoiler or let "a blackout zone" carry it.
- D9 Helper recruitment: whether to ask the Discord channel at all, or cap registration at sixteen.

## 11. Order and dates

| By | Milestone | Gate |
| --- | --- | --- |
| 19 Oct | Repo, pin `cpomdp==0.5.0`, `world`, `planner`, `build`, `sim`. `test_api_contract.py` and `test_measured_claims.py` green on the pin | tests |
| 23 Oct | `check`, `plot`, `preview`, `symptoms`, checkpoints. Figures regenerated | figures match the deck's |
| 27 Oct | Notebook generated, runs on Colab and under uv. README. Hints file. Deck edits from section 8 | `test_notebook.py`, render check |
| 30 Oct | Self-guided tail. Delivery artefacts | `test_notebook.py` runs the tail |
| 2 Nov | Dry run 1, solo, timed | timings written down |
| 5–8 Nov | Dry runs 2 and 3, one with live handoffs. Kev has run the notebook once. Helpers briefed | handoff token agreed |
| 9 Nov | Freeze. Typo fixes only | tag `v2026.11` |
| 11 Nov | Install drop-in on Discord, evening | attendance noted |
