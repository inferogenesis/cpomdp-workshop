"""What the notebook says, cell by cell. ``build_notebook.py`` assembles it.

Each session is a brief, an attempt cell with two ``...`` lines to write, two bonuses
and a question to think about. The hints and the reference cell are not here. The
hints live in ``delivery/hints_chat.txt``, and the reference cell is the attempt with
both hints put in, so neither can drift from the attempt.
"""

from dataclasses import dataclass

TITLE = """\
# Build a curious rocket

What makes a rocket curious? In this hands-on workshop, we'll explore that question
using `cpomdp`, a Python library for continuous active inference. We'll begin with a
rocket that uses a Kalman filter to estimate its position and a linear quadratic
regulator (LQR) to guide it towards a landing pad. Next, we'll introduce a sensor whose
reliability depends on where the rocket flies and investigate how this changes its
behaviour. Finally, we will hide the landing pad and compare two rockets that differ
only in curiosity.

Work through the notebook from top to bottom, running the code cells as you go. Each
task includes two lines of code for you to complete, along with two hints and a
reference solution if you need some help. Feel free to use these at any point. The aim
is to explore the ideas behind the code, rather than test your programming skills. If
you finish a task early, each one also has two optional bonus exercises.

At the end of each task, **make sure you run the checkpoint cell**, even if everything
worked correctly. This restores the reference setup needed for the next task, so you can
continue without difficulty if something didn't work as expected.

If you have any questions during the workshop, you can also send them through
[this form](https://docs.google.com/forms/d/e/1FAIpQLSf7mgwxYEFpnDAkqaYolxWzP1sesG6L5Rf_5pVswE3UwtkcwA/viewform).
It's anonymous unless you leave an email for a written reply."""

GETTING_STARTED = """\
## Getting started

We recommend using **Google Colab**, which allows you to run the notebook directly in
your browser without needing to set up Python on your computer. If you prefer to work
locally using VS Code, that's absolutely fine too.

To run a code cell, click the play button beside it, or press **Shift + Enter**. Start
by running the setup cell below, which will install the required workshop packages from
PyPI and check that everything is ready. If you're using VS Code, make sure the required
packages are installed in your selected Python environment.

The first time you run a cell, Colab warns that the notebook was not authored by Google.
It opens from our GitHub repository, so click **Run anyway**. Colab does not keep your
changes once you close the tab, so if you would like to keep your work, use **File →
Save a copy in Drive** before you start.

Colab may offer an **Explain error** button under an error, or suggest code as you type.
We would ask you to write the two lines yourself, because working them out is the point
of each task. The hints and the reference cells are there whenever you are stuck, and
using them is completely fine."""

INSTALL = """\
# Installs the workshop on Colab, then checks the versions it was measured on.
import sys

if "google.colab" in sys.modules:
    !pip install -q cpomdp-workshop=={version}
import cpomdp_workshop as workshop

workshop.hello()"""

PREVIEW_INTRO = """\
## A preview of what's to come

Before we begin building our rockets, let's take a look at where we're heading. Run the
next cell to see a preview of the last two rockets we will build. Don't worry if you
can't yet explain what you're seeing. We'll work towards understanding their behaviour
as we progress through the exercises."""

PREVIEW = "workshop.preview()"

HELP = """\
## Need a little help?

If you encounter an error while working through the notebook, do not worry. The
troubleshooting cell below lists the five errors we expect people to meet most often,
each with its cause and its fix. The same cell sits under every task, so you can open it
at any point during the workshop."""

SYMPTOMS = """\
# @title Stuck on an error? The five most common, with their fixes
import cpomdp_workshop as workshop

workshop.symptoms()"""

CHECKPOINT = """\
# Checkpoint {number}. Always run it. It puts the reference names in place.
import cpomdp_workshop as workshop

checkpoint = workshop.after_session_{number}()
globals().update(checkpoint)
print("restored:", ", ".join(checkpoint))"""


@dataclass(frozen=True)
class Bonus:
    """One bonus: a text cell that asks, hints and answers, and code if it needs it.

    Attributes:
        title: what the heading says after "Bonus N ·".
        question: what to try, shown in full.
        hint: shown when the learner opens Hint.
        answer: shown when the learner opens Answer. Each number in it has a claim
            test.
        code: a code cell under the text, or ``None`` when the bonus is worked in the
            attempt cell above.
    """

    title: str
    question: str
    hint: str
    answer: str
    code: str | None = None


@dataclass(frozen=True)
class Session:
    """One task, as the notebook lays it out.

    Attributes:
        number: 1, 2 or 3.
        brief: the markdown above the attempt.
        attempt: the code cell, with exactly two ``...`` lines.
        bonuses: the two collapsed bonus cells, in order.
        poll: the question to think about after the checkpoint. No slide gives it
            time.
    """

    number: int
    brief: str
    attempt: str
    bonuses: tuple["Bonus", ...]
    poll: str


SESSION_1 = Session(
    number=1,
    brief="""\
## Task 1 · The 1960 rocket

We'll begin with a rocket that uses a Kalman filter to estimate its position from noisy
sensor readings and a linear quadratic regulator (LQR) to guide it towards a landing
pad. In this first model, the sensor is equally reliable wherever the rocket flies. But
does this mean that every possible movement is equally informative?

In the code below, you'll find two missing lines marked with `...`. Your task is to
complete these lines to define what the sensor observes and how much noise is present in
its readings. Everything else is already provided.

Once you've completed the two lines, run the cell. The `check_agent` function will show
you the rocket you've built and simulate its flight. Next, `probe_model` will evaluate
how much information the rocket could expect to gain from each possible action, and
`show_probe` will display the results. In the theory, this is the epistemic term of
expected free energy, the term pymdp switches on with `use_states_info_gain`
([the Active Inference book](https://doi.org/10.7551/mitpress/12441.001.0001), eq. 2.6,
p. 33).

Pay particular attention to the **spread**, which tells us how much the expected
information gain varies between the available movements.

**Share your spread value in the workshop chat.**""",
    attempt="""\
import jax.numpy as jnp
from cpomdp import (
    Agent,
    Belief,
    LinearGaussianModel,
    ProbeReport,
    StateGoal,
    probe_model,
)

from cpomdp_workshop import LAUNCH, PAD, check_agent, compass_moves, show_probe

observation_matrix = ...  # pymdp's A, with R below: the reading sees the position
observation_noise = ...  # R: variance 0.05 on each axis, the same everywhere

model = LinearGaussianModel(
    dynamics_matrix=jnp.eye(2),  # pymdp's B: the rocket stays put unless moved
    control_matrix=jnp.eye(2),  # a move adds straight to the position
    observation_matrix=observation_matrix,
    observation_noise=observation_noise,
    dynamics_noise=1e-3 * jnp.eye(2),  # Q: a little drift each step
    prior=Belief(mean=LAUNCH, cov=4 * jnp.eye(2)),  # pymdp's D: unsure where it is
)
# effort is the thruster cost
goal = StateGoal(target=PAD, precision=jnp.eye(2), effort=100 * jnp.eye(2))
rocket = Agent(model, objective=goal)
check_agent(rocket)

report: ProbeReport = probe_model(model, belief=model.prior, actions=compass_moves())
show_probe(report)  # paste the spread in chat""",
    bonuses=(
        Bonus(
            title="Try changing the other numbers",
            question="""\
In Task 1, you completed two lines, but the cell contains other numbers too. Try
changing them one at a time, rerunning the cell each time:

- the noise level, the `0.05` in `observation_noise`
- the rocket's initial uncertainty, the `4` in `prior`
- the size of each movement, 0.5 by default, by writing `compass_moves(1.5)` instead
  of `compass_moves()`

Keep an eye on the spread in the last line. Does it ever move away from 0.000000?""",
            hint="""\
A bigger number in front of `jnp.eye(2)` means more noise, or more initial uncertainty.
`compass_moves(1.5)` makes every movement 1.5 units long.""",
            answer="""\
It never does. Whatever the noise level, the initial uncertainty or the step size, the
spread stays at 0.000000. Because the noise is the same everywhere, every movement
leaves the rocket equally uncertain. This is the Koudahl collapse.
[Koudahl, Kouw and de Vries (2021)](https://doi.org/10.3390/e23121565) proved that when
a linear Gaussian model's noise is fixed, the expected information gain is the same for
every action, so curiosity can never change what the rocket does. Only noise that
changes with position can make some movements more informative than others. When you're
done, run the checkpoint to restore the task's original numbers.""",
        ),
        Bonus(
            title="Can unequal sensor noise make the rocket curious?",
            question="""\
So far, our rocket's sensor has been equally reliable along both axes. But what happens
if the sensor measures one axis much more accurately than the other?

Try modifying the observation noise matrix so that measurements along one axis are much
noisier than along the other. Keep the sensor's reliability constant across locations.

Before running the code, consider whether this change should make some movements more
informative than others.

Run `probe_model` again and compare the expected information gain and its spread across
the available actions.

**For an additional challenge:** Try making the rocket's prior uncertainty different
along the two axes as well.

What changes in the results, and what stays the same? Can you explain your observations
in terms of how the Kalman filter updates uncertainty?""",
            hint="""\
`jnp.diag(jnp.array([0.05, 2.0]))` builds a matrix with 0.05 and 2.0 on its diagonal, so
x is read much more accurately than y. Try using it as `observation_noise` in a copy of
your Task 1 cell. The expected information gain is in `report.epistemic_range`, which
holds the lowest and highest values across the nine movements. `show_probe` prints only
the spread, so add `print(report.epistemic_range)` under it, and run Task 1's numbers
the same way to have a value to compare with.""",
            answer="""\
Each reading becomes less informative: 2.7467 nats with 0.05 on x and 2.0 on y, compared
with 4.3947 in Task 1. But every movement is still equally informative, so the spread
stays at 0.000000. Making the prior uneven changes the amount of information again, and
the spread still stays at 0. The reason lies in how the Kalman filter updates its
uncertainty: it uses the noise and the previous uncertainty, but never the movement or
the position. When the noise is the same everywhere, a movement changes where the rocket
is, not how well it can see.""",
            code="""\
# Your code here. A copy of your task 1 cell is a good start.""",
        ),
    ),
    poll="""\
### A question to think about

The 1960 rocket's spread is zero whatever you change. Why?

1. The prior is too wide for any one reading to matter.
2. The noise is the same everywhere, so every move leaves the rocket equally unsure.
3. A step of 0.5 is too short to reach anywhere different.
4. LQR ignores what the probe reports.

<details><summary>Answer</summary>

2. The uncertainty after a reading depends on the noise where the reading is taken.
Here that noise never changes, so no move can teach more than another. The other
three change nothing about that, which bonus 1 lets you check.

</details>""",
)

SESSION_2 = Session(
    number=2,
    brief="""\
## Task 2 · The beacon

In Task 1, the rocket's sensor was equally reliable everywhere. Now, we'll introduce a
navigation beacon. Observations are more reliable near the beacon and become noisier
with increasing distance. The parameter `k` (`beacon_gain` in the code) controls how
quickly this noise increases, up to a maximum value. In the theory, this noise sets the
ambiguity in expected free energy (Active Inference book, eq. 2.6, p. 33), and now it
depends on where the rocket is. With the same noise everywhere, Task 1's spread stayed
at zero.

We're also introducing a **preference over observations**, rather than states. The
rocket prefers a sensor reading that indicates it is on the landing pad. At each step,
its planner evaluates nine possible actions, taking into account the expected
observations and their uncertainty. Each action is a one-step policy, scored by its
expected free energy (Active Inference book, section 2.7, p. 32).

The rocket holds this preference only loosely, with a precision of 0.1, so the
information a move offers can compete with the pull of the pad. Expected free energy
has no other weight between the two, so the precision is the only setting that trades
one against the other.

To see what curiosity does, we will fly the same rocket twice. The first flight has
curiosity on. The second ranks the same nine moves with the epistemic term taken out,
so it weighs only how close each move is expected to bring its reading to the pad. The
switch that turns curiosity off is a helper in this workshop's package, and `cpomdp`
itself has no such switch.

**Your task:** Complete the two missing lines marked with `...`:

1. Choose a value for `k` to control how quickly the noise increases with distance.
2. Calculate the additional sensor noise in `my_sensor` using the parameters provided.

`default_params(k)` gathers `k` and the beacon's other settings into a dictionary called
`params`. The comments in the code list each setting and how to read it by name, for
example `params["beacon_cap"]`.

Run the cell to test your sensor and plot both flights. A rocket has landed if it ends
within 0.5 of the pad, inside the circle the plot draws round it. `show_landings` prints
how far from the pad each rocket ended, and whether it landed.

Start with `k = 0.08`, then raise it and watch both rockets. At what `k` does one of
them stop landing? What might explain the difference, when both rockets share one model
and one sensor?

**Share that `k` value in the workshop chat.**""",
    attempt="""\
import jax.numpy as jnp
from cpomdp import (
    ActionSelector,
    Agent,
    Belief,
    CallableSensor,
    LinearGaussianModel,
    Preference,
)

from cpomdp_workshop import (
    LAUNCH,
    PAD,
    check_my_sensor,
    compass_planner,
    default_params,
    fly,
    plot_flights,
    show_landings,
)

beacon_gain = ...  # k: type 0.08 first, then raise it until one rocket stops landing
params = default_params(beacon_gain)
# The settings params holds for this task. The only one we're concerned with for this
# exercise is k, "beacon_gain". The rest are chosen for you to best display the
# behaviour:
#   params["beacon_gain"]  k            how fast the noise grows away from the beacon
#   params["beacon_cap"]   4.0          the largest distance_squared that counts
#   params["beacon"]       jnp.array([2.5, 1.4])  where the beacon is
#   params["noise_floor"]  0.05         the noise at the beacon itself


def my_sensor(x, params):  # the reading's noise at position x
    distance_squared = jnp.sum((x - params["beacon"]) ** 2)
    pull = ...  # k times jnp.minimum(distance_squared, cap), read from params
    return jnp.eye(2) * (params["noise_floor"] + pull)


check_my_sensor(my_sensor, params)

curious_model = LinearGaussianModel(
    dynamics_matrix=jnp.eye(2),
    control_matrix=jnp.eye(2),
    dynamics_noise=1e-3 * jnp.eye(2),
    prior=Belief(mean=LAUNCH, cov=4 * jnp.eye(2)),
    observation_model=CallableSensor(
        observation_matrix=jnp.eye(2), noise_fn=my_sensor, noise_params=params
    ),
)
# a precision of 0.1 is loose enough for the information to compete
preference = Preference(target=PAD, precision=0.1 * jnp.eye(2))
# the planner scores all nine compass moves every step
planner: ActionSelector = compass_planner(curious_model)
curious = Agent(model=curious_model, objective=preference, selector=planner)
incurious = Agent(
    model=curious_model,
    objective=preference,
    selector=compass_planner(curious_model, curiosity=False),
)
flights = {"curiosity on": fly(curious), "curiosity off": fly(incurious)}
print(f"your k = {beacon_gain}")  # paste the k where one rocket stopped landing
show_landings(flights)
plot_flights(flights, my_sensor, params)""",
    bonuses=(
        Bonus(
            title="Turn the preference up",
            question="""\
So far, our rocket has held its preference for the pad loosely, at a precision of 0.1.
What happens to curiosity when the rocket wants the pad more?

The cell below flies the rocket with curiosity on and with curiosity off at four
precisions, 0.1, 0.3, 1 and 3, with `k = 0.08` on seed 3. For each precision it prints
the gap, the largest distance between the two routes at the same step, and how many of
each rocket landed within 0.5 of the pad. The cell uses `my_sensor` from your Task 2
cell, so if your runtime has restarted, run Checkpoint 2 below first to restore it.

How does the gap change as the precision rises? What could explain it?

**For an additional challenge:** Set `beacon_gain = 0.2` and `seeds = range(10)`. The
cell then prints how many of the ten seeds part by 1 or more, and the median gap. Do
your conclusions still hold?

**One more thing to try:** Go back to your Task 2 cell and keep raising `k` past the
value you shared. What happens to the rocket with curiosity off?""",
            hint="""\
`fly_both` builds a new pair of rockets for every seed and every precision, because a
rocket can only fly once. `on.route_gap(off)` is the gap between two flights. To try
another precision, add it to `(0.1, 0.3, 1, 3)` in the last loop. Then compare how the
two terms of expected free energy depend on the precision.""",
            answer="""\
On seed 3, the two routes are 1.53 apart at a precision of 0.1 and 0.77 apart at 0.3.
At 1 and at 3 they are the same route, and every rocket lands. Over ten seeds at
`k = 0.08`, the routes part by 1 or more on all ten at 0.1, on one at 0.3 and on none
at 1 or 3. At `k = 0.2` they part on all ten at 0.1, on eight at 0.3 and on none at 1
or 3.

The pragmatic term scales with the precision and the epistemic term does not, so a
higher precision drowns out the information a move offers. At a precision of 1, the
value Task 1's goal used, curiosity decides none of the 36 moves at `k = 0.08` on any
of the ten seeds.

A bend towards the beacon does not show that a rocket is curious. With curiosity off,
the rocket still bends towards the beacon at `k = 0.2`: on seed 3 it passes 1.06 from
the beacon, against 1.40 for the straight route. The readings there are less noisy,
and the pragmatic term pays for noise as well as for distance. The rocket with
curiosity off flies the same route at any precision, so its bend owes nothing to
curiosity.

The same term explains what you see if you keep raising `k`. On seed 3, the rocket with
curiosity off lands up to `k = 0.45` and stops 1.42 short of the pad at `k = 0.46`. At
`k = 0.5`, neither rocket lands on any of the ten seeds. Near the pad the readings are
then so noisy that the pragmatic term alone prefers to stop short.""",
            code='''\
import jax.numpy as jnp
import numpy as np
from cpomdp import Agent, Preference

from cpomdp_workshop import PAD, compass_planner, default_params, fly, model_with

beacon_gain = 0.08  # for the additional challenge, try 0.2
seeds = [3]  # one flight per rocket. For the additional challenge, try range(10)


def fly_both(precision, seed):
    """The same rocket flown with curiosity on, then with curiosity off."""
    flights = []
    for curiosity in (True, False):
        # A rocket can only fly once, so every flight builds a new one. model_with
        # builds the same model as your task 2 cell.
        model = model_with(my_sensor, default_params(beacon_gain))
        preference = Preference(target=PAD, precision=precision * jnp.eye(2))
        planner = compass_planner(model, curiosity=curiosity)
        rocket = Agent(model, objective=preference, selector=planner)
        flights.append(fly(rocket, seed=seed))
    return flights


for precision in (0.1, 0.3, 1, 3):
    gaps, on_landed, off_landed = [], 0, 0
    for seed in seeds:
        on, off = fly_both(precision, seed)
        gaps.append(on.route_gap(off))
        on_landed += on.distance_to(PAD) < 0.5
        off_landed += off.distance_to(PAD) < 0.5
    parted = sum(gap >= 1 for gap in gaps)
    print(
        f"precision {precision:>3}: median gap {np.median(gaps):.2f}, "
        f"parted by 1 or more on {parted} of {len(seeds)}, "
        f"landed with curiosity on {on_landed}, off {off_landed}"
    )''',
        ),
        Bonus(
            title="What does curiosity need?",
            question="""\
The epistemic term of one move is ln(1 + σ²/r). Here σ² is how unsure the rocket is
about its position after the move, and r is the reading noise where the move takes it.
A move teaches more where r is small and where the rocket is unsure.

Try three one-line edits to your Task 2 cell with `k = 0.08`, one at a time. Before
each run, predict what it will do to the curious rocket's route.

1. Under the `params` line, add `params["beacon_cap"] = 1.0`.
2. In `curious_model`, change `dynamics_noise` to `1e-2 * jnp.eye(2)`. The rocket
   then believes it drifts ten times as much between readings. The world itself adds
   no drift, so only the rocket's belief changes.
3. In `curious_model`, change the prior to
   `prior=Belief(mean=LAUNCH, cov=25 * jnp.eye(2))`.

Which edits change what curiosity does, and which leave it alone?""",
            hint="""\
With a cap of 1, every point of the straight route is already past the cap. A rocket
that believes it drifts more grows less sure of where it is between readings. A wider
prior only changes how unsure the rocket is before its first reading. Undo each edit
before you try the next.""",
            answer="""\
With a cap of 1, the noise is the same all along the straight route, so no move there
teaches more than another. On seed 3 the two rockets fly the same straight route. Over
ten seeds they fly the same route on nine. On the tenth, a noisy reading pushes the
rocket within reach of the beacon, where the noise varies again, and the routes split.

When the rocket believes it drifts ten times as much, σ² stays large, so a move towards
the beacon always offers more information. The curious rocket stays near the beacon and
stops 2.67 short of the pad on seed 3. The rocket with curiosity off lands 0.16 from it.
Over ten seeds, no curious rocket lands and eight of the incurious ones do.

A wider prior leaves the curious rocket's route unchanged on all ten seeds. Its first
reading leaves it about as sure as it would have been, and the beacon's field does not
reach the launch site, so the first move has nothing to learn either way. When you are
done, run the checkpoint to restore the task's original numbers.""",
        ),
    ),
    poll="""\
### A question to think about

At `k = 0.4` the curious rocket stops short of the pad, and the same rocket with
curiosity off lands. Why?

1. Its sensor is worse than the incurious rocket's.
2. Moves towards the pad lead away from the beacon, so each one teaches less. With a
   loose preference, the information it would lose outweighs the pull of the pad.
3. It cannot see the pad from near the beacon.
4. It runs out of moves.

<details><summary>Answer</summary>

2. Both rockets share one model and one sensor, so 1 is out. A reading near the beacon
still says where the rocket is, so 3 is out too. Neither rocket runs out of moves in 36
steps.

</details>""",
)

SESSION_3 = Session(
    number=3,
    brief="""\
## Task 3 · Find the pad

In Task 2, the rocket knew where the pad was and curiosity only changed its route. Now
the rocket does not know where the pad is. To achieve this in silicon, we promote
the pad's location to a latent state (a state the rocket cannot observe directly and
has to infer from its readings). There are now two realities, the true pad's location,
which the generative process (the world) dictates, and the rocket's belief about the
pad's location, which its generative model holds (Active Inference book, Figure 2.2,
p. 23). As we will see, they are not the same thing.

In Tasks 1 and 2, the expected information gain was about the rocket's own position.
Now the epistemic term of expected free energy also covers the pad's position.

The world is a continuous T-maze, if there were such a thing. In the classic T-maze, a
rat chooses between guessing to go down a path or gaining a cue first, then making a
better guess. To do that in continuous space we use three sensors:

- a reading of its own position
- a coarse radar, which points roughly from the rocket towards the pad, with the same
  noise everywhere
- the beacon's broadcast of the pad's exact position, which is clear near the beacon
  and fades with distance

**Your task:** Because we have promoted the pad's location to a latent state, the
state now holds four numbers instead of two, so two variables must become
four-dimensional as well. Complete the two missing lines marked with `...`:

1. Set `prior_mean`. In Task 2, the prior's mean was `LAUNCH`, where the rocket
   believes it starts. Now it also holds `PAD_GUESS`, where the rocket guesses the pad
   is.
2. Set `prior_cov`, how unsure the rocket starts about its own position and about
   the pad's. The cell starts with a `pad_spread` of 1.5, which is also how far the
   true pad strays from the guess in the world.

Run the cell. As before, **a rocket has landed if it ends within 0.5 of the true pad**.
Which rocket finds the pad? What might explain the difference, when both share one
model and the same sensors?

Then lower `pad_spread` and run the cell again. This changes only how unsure the rocket
is about the pad. The world still draws the true pad with a spread of 1.5. At what
`pad_spread` does the curious rocket stop finding the pad? What might explain it?

**Share that `pad_spread` in the workshop chat.**""",
    attempt="""\
import jax.numpy as jnp
from cpomdp import Agent, Belief, CallableSensor, LinearGaussianModel, Preference

from cpomdp_workshop import (
    LAUNCH,
    PAD_GUESS,
    PAD_READINGS,
    broadcast_noise,
    compass_planner,
    fly_pad,
    pad_params,
    pad_sensor,
    plot_pad,
    radar_preference,
    show_misses,
)

params = pad_params()
pad_spread = 1.5  # how unsure the rocket is about the pad. Lower it after a first run
# The state is four numbers: the rocket's position, then the pad's.
prior_mean = ...  # LAUNCH's two numbers, then PAD_GUESS's two, in one jnp.array
prior_cov = ...  # jnp.diag of 0.25, 0.25, then pad_spread**2 twice, in a jnp.array
model = LinearGaussianModel(
    dynamics_matrix=jnp.eye(4),  # nothing moves by itself
    control_matrix=jnp.vstack([jnp.eye(2), jnp.zeros((2, 2))]),  # moves the rocket only
    dynamics_noise=jnp.diag(jnp.array([1e-3, 1e-3, 1e-9, 1e-9])),  # the pad stays put
    prior=Belief(mean=prior_mean, cov=prior_cov),
    # six readings: the rocket's position, the radar, and the broadcast with its noise
    observation_model=CallableSensor(
        observation_matrix=PAD_READINGS,
        noise_fn=pad_sensor(broadcast_noise),
        noise_params=params,
    ),
)
# the goal is a radar reading that says the pad is right here
preference: Preference = radar_preference()
curious = Agent(model, objective=preference, selector=compass_planner(model))
incurious = Agent(
    model, objective=preference, selector=compass_planner(model, curiosity=False)
)
flights = {"curiosity on": fly_pad(curious), "curiosity off": fly_pad(incurious)}
print(f"your pad_spread = {pad_spread}")  # paste the one where curiosity on missed
show_misses(flights)
plot_pad(flights, broadcast_noise, params)""",
    bonuses=(
        Bonus(
            title="How sure is the rocket?",
            question="""\
In Task 3, you lowered `pad_spread` until the curious rocket stopped finding the pad on
one flight. Does the same happen on other flights, and what if the rocket is less sure
of its guess than it should be?

The cell below flies both rockets over seeds 0 to 9 at three spreads, 0.1, 1.5 and 3.0.
The world does not change: the true pad is always drawn with a spread of 1.5. Only the
rocket's own uncertainty changes. A rocket lands if it ends within 0.5 of the true pad.
The cell uses `params` from your Task 3 cell, so if your runtime has restarted, run
Checkpoint 3 below first to restore it.

Before you run it, what do you expect an overconfident rocket to do?""",
            hint="""\
The epistemic term rewards a move for what it is expected to teach. Think about how
much a rocket that is already sure where the pad is expects to learn from the beacon's
broadcast.""",
            answer="""\
With a spread of 0.1, the rocket believes there is almost nothing to learn about the
pad, so the curious rocket never detours to the beacon. Both rockets land on 1 of the 10
seeds and miss by a median of 1.42. With a spread of 1.5, the curious rocket lands on 9
seeds, with a median miss of 0.25, and the incurious one on 3, with a median miss of
0.61. With a spread of 3.0, the curious rocket lands on all 10 and the incurious one on
5.

Curiosity pays off here only when the rocket's own uncertainty about the pad is large
enough to make the beacon worth a visit. The balance between the two terms is sensitive
to settings like this one, and we will come back to that on the slides.""",
            code='''\
import numpy as np
from cpomdp import Agent

from cpomdp_workshop import (
    broadcast_noise,
    compass_planner,
    fly_pad,
    pad_model,
    radar_preference,
)

seeds = range(10)


def misses(pad_spread, curiosity):
    """How far the rocket ends from the true pad on each seed."""
    found = []
    for seed in seeds:
        # A rocket can only fly once, so every flight builds a new one. pad_model
        # builds the same model as your task 3 cell.
        model = pad_model(broadcast_noise, params, pad_spread)
        planner = compass_planner(model, curiosity=curiosity)
        rocket = Agent(model, objective=radar_preference(), selector=planner)
        found.append(fly_pad(rocket, seed=seed).miss)
    return np.array(found)


for pad_spread in (0.1, 1.5, 3.0):
    for name, curiosity in (("curiosity on", True), ("curiosity off", False)):
        miss = misses(pad_spread, curiosity)
        print(
            f"spread {pad_spread:>3}, {name:<13}: landed {(miss < 0.5).sum():>2} of "
            f"{len(seeds)}, median miss {np.median(miss):.2f}"
        )''',
        ),
        Bonus(
            title="When is curiosity worth it?",
            question="""\
We built Task 3's world so that curiosity pays off. What happens if we make the
information easier to get?

The cell below flies both rockets over seeds 0 to 9 and counts the landings. Run it as
it is, then try three changes, one at a time:

1. A broadcast that carries far: `settings["broadcast_gain"] = 0.1`.
2. A capped broadcast, like Task 2's beacon: set `broadcast = capped_broadcast`.
3. A sharp radar: `settings["radar_noise"] = 1.0`, in place of the default 25.

In which worlds does the curious rocket still land more often than the incurious one?
Why do you think we made the radar coarse?""",
            hint="""\
Undo each change before you try the next. For each world, ask where the rocket can find
out where the pad is. If it can find out from where it already is, or on the way to the
pad, what is a detour worth?""",
            answer="""\
In the world as built, the curious rocket lands on 9 of the 10 seeds and the incurious
one on 3. With a broadcast gain of 0.1, the broadcast can be heard clearly from far
away, and both rockets land on all 10. With a capped broadcast, the noise never grows
past a level that still lets the rocket hear the pad's position, and both land on all 10
again. With a sharp radar, the rocket finds the pad with the radar alone, and both land
on all 10. The curious rocket still detours to the beacon on 4 of those seeds, but the
detour buys it nothing.

This is why the radar is coarse by default. The task asks whether a rocket will go and
get information that is only available somewhere else. With a good radar, or a
broadcast that reaches everywhere, the question disappears.""",
            code='''\
import jax.numpy as jnp
from cpomdp import Agent

from cpomdp_workshop import (
    broadcast_noise,
    compass_planner,
    fly_pad,
    pad_model,
    pad_params,
    radar_preference,
)


def capped_broadcast(x, params):
    """Task 2's field on the broadcast: k = 0.08, capped at a squared distance of 4."""
    distance_squared = jnp.sum((x - params["beacon"]) ** 2)
    return params["noise_floor"] + 0.08 * jnp.minimum(distance_squared, 4.0)


seeds = range(10)
settings = pad_params()  # change one setting here, such as settings["radar_noise"]
broadcast = broadcast_noise  # or capped_broadcast


def landings(curiosity):
    """On how many seeds the rocket ends within 0.5 of the true pad."""
    landed = 0
    for seed in seeds:
        # A rocket can only fly once, so every flight builds a new one. pad_model
        # builds the same model as your task 3 cell.
        model = pad_model(broadcast, settings, 1.5)
        planner = compass_planner(model, curiosity=curiosity)
        rocket = Agent(model, objective=radar_preference(), selector=planner)
        landed += fly_pad(rocket, seed=seed).miss < 0.5
    return landed


print(f"curiosity on  landed {landings(True)} of {len(seeds)}")
print(f"curiosity off landed {landings(False)} of {len(seeds)}")''',
        ),
    ),
    poll="""\
### A question to think about

The incurious rocket missed the pad and the curious one found it. Why?

1. The incurious rocket's radar is worse.
2. It never went where the pad's position can be heard clearly, so it had only a
   coarse radar to correct its guess with.
3. It looked too few moves ahead.
4. Its preference for the pad was too strong.

<details><summary>Answer</summary>

2. Both rockets share one model and one radar, so 1 is out. For 3, try the self-guided
cell "Two moves ahead": on seed 3, the incurious rocket looking two moves ahead still
does not go to the beacon. 4 is backwards. At a precision of 1, the curious rocket
stops checking the beacon too.

</details>""",
)

SESSIONS = (SESSION_1, SESSION_2, SESSION_3)

CODA_INTRO = """\
## Coda · What can we conclude from our results?

Before we finish, let's consider what the numbers produced by our models actually tell
us.

The final cell runs two checks. The first flies Task 1's rocket on two of `cpomdp`'s
inference engines, the Kalman filter and a graph of one node, and prints the largest
difference between their beliefs. The second is the planner's `certificate`. At each
step, our curious rocket evaluates nine possible actions, and the certificate records
that every one of them was scored. This can quickly become expensive: with five possible
actions and a planning horizon of seven steps, there are **78,125 possible action
sequences** to evaluate.

Run the final cell and examine the outputs.

**Consider:** What do these results establish about the computations performed, and what
can we not yet conclude about the correctness of our model?"""

CODA = """\
from cpomdp import KalmanBackend

import cpomdp_workshop as workshop

# the two methods agree about where the 1960 rocket is
workshop.check_backends(
    KalmanBackend(model=workshop.model_with()),
    workshop.one_node(workshop.model_with()),
)
print(curious.selector.certificate)
print(workshop.axis_certificate())"""

TAIL_INTRO = """\
---

## Self-guided

Nothing below this line is run live. Each cell stands on the checkpoint above it."""

TAIL = (
    (
        "Two moves ahead",
        """\
from cpomdp_workshop import broadcast_noise, pad_pair, plot_pad, show_misses

# both rockets again, each scoring all 81 two-move plans at every step
flights_ahead = pad_pair(broadcast_noise, params, pad_spread=pad_spread, horizon=2)
show_misses(flights_ahead)
plot_pad(flights_ahead, broadcast_noise, params, title="looking two moves ahead")""",
    ),
    (
        "A model one filter cannot hold",
        """\
from cpomdp_workshop import two_node_backend

try:
    two_node_backend().to_flat_model()
except Exception as refusal:
    print(f"{type(refusal).__name__}: {refusal}")""",
    ),
    (
        "Whose detour is it?",
        """\
from cpomdp_workshop import (
    curiosity_decided,
    curious_rocket,
    default_params,
    fly,
    pad_preference,
    plot_flights,
    reference_sensor,
)

# A world with a blackout on the straight route, where readings are very noisy.
# The same rocket flies it with curiosity on and off.
k = 0.08
rocket = curious_rocket(k)
detours = {
    "curiosity on": fly(rocket),
    "curiosity off": fly(curious_rocket(k, curiosity=False)),
}
decided = curiosity_decided(detours["curiosity on"], rocket.model, pad_preference())
print(f"curiosity decided {decided.sum()} of {len(decided)} moves")
print("on steps", [step + 1 for step, changed in enumerate(decided) if changed])
plot_flights(detours, reference_sensor, default_params(k))""",
    ),
)
