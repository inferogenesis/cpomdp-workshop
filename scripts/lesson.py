"""What the notebook says, cell by cell. ``build_notebook.py`` assembles it.

Each session is a brief, an attempt cell with two ``...`` lines to write, two bonuses
and a poll. The hints and the reference cell are not here. The hints live in
``delivery/hints_chat.txt``, and the reference cell is the attempt with both hints
put in, so neither can drift from the attempt.
"""

from dataclasses import dataclass

TITLE = """\
# Build a curious rocket

What makes a rocket curious? In this hands-on workshop, we'll explore that question
using `cpomdp`, a Python library for continuous active inference. We'll begin with a
rocket that uses a Kalman filter to estimate its position and a linear quadratic
regulator (LQR) to guide it towards a landing pad. Next, we'll introduce a sensor whose
reliability depends on where the rocket flies and investigate how this changes its
behaviour. Finally, we'll compare three rockets navigating an environment containing an
obstacle that none of them knows is there.

Work through the notebook from top to bottom, running the code cells as you go. Each
task includes two lines of code for you to complete, along with two hints and a
reference solution if you need some help. Feel free to use these at any point. The aim
is to explore the ideas behind the code, rather than test your programming skills. If
you finish a task early, each one also has two optional bonus exercises.

At the end of each task, **make sure you run the checkpoint cell**, even if everything
worked correctly. This restores the reference setup needed for the next task, so you can
continue without difficulty if something didn't work as expected.

If you have any questions during the workshop, you can also submit them anonymously
using the form (link to come)."""

GETTING_STARTED = """\
## Getting started

We recommend using **Google Colab**, which allows you to run the notebook directly in
your browser without needing to set up Python on your computer. If you prefer to work
locally using VS Code, that's absolutely fine too.

To run a code cell, click the play button beside it, or press **Shift + Enter**. Start
by running the setup cell below, which will install the required workshop packages from
PyPI and check that everything is ready. If you're using VS Code, make sure the required
packages are installed in your selected Python environment."""

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
next cell to see a preview of the three rockets we'll be exploring. Don't worry if you
can't yet explain what you're seeing. We'll work towards understanding their behaviour
as we progress through the exercises."""

PREVIEW = "workshop.preview()"

HELP = """\
## Need a little help?

If you encounter an error while working through the notebook, don't worry. The
troubleshooting section below covers five common issues and how to resolve them. You can
return to it at any point during the workshop."""

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
        poll: the markdown poll after the checkpoint.
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
from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal, probe_model

from cpomdp_workshop import LAUNCH, PAD, check_agent, compass_moves, show_probe

observation_matrix = ...  # pymdp's A, with R below: the reading sees the position
observation_noise = ...  # R: variance 0.05 on each axis, the same everywhere

model = LinearGaussianModel(
    jnp.eye(2),  # dynamics, pymdp's B: the rocket stays put unless moved
    control_matrix=jnp.eye(2),  # a move adds straight to the position
    observation_matrix=observation_matrix,
    observation_noise=observation_noise,
    dynamics_noise=1e-3 * jnp.eye(2),  # Q: a little drift each step
    prior=Belief(mean=LAUNCH, cov=4 * jnp.eye(2)),  # pymdp's D: unsure where it is
)
goal = StateGoal(PAD, precision=jnp.eye(2), effort=100 * jnp.eye(2))  # thruster cost
rocket = Agent(model, goal)
check_agent(rocket)

report = probe_model(model, model.prior, compass_moves())
show_probe(report)  # paste the spread in chat""",
    bonuses=(
        Bonus(
            title="Try changing the other numbers",
            question="""\
In Task 1, you completed two lines, but the cell contains other numbers too. Try
changing them one at a time, rerunning the cell each time:

- the noise level, the `0.05` in `observation_noise`
- the rocket's initial uncertainty, the `4` in `prior`
- the size of each movement, by writing `compass_moves(1.5)` instead of
  `compass_moves()`

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
holds the lowest and highest values across the nine movements.""",
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
### Poll 1

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

**Your task:** Complete the two missing lines marked with `...`:

1. Choose a value for `k` to control how quickly the noise increases with distance.
2. Calculate the additional sensor noise in `my_sensor` using the parameters provided.

`default_params(k)` gathers `k` and the beacon's other settings into a dictionary called
`params`. The comments in the code list each setting and how to read it by name, for
example `params["beacon_cap"]`.

Run the cell to test your sensor and plot the rocket's flight path.

Start with `k = 0.08`, then experiment with different values. Can you find a value that
makes the rocket take a different route towards the beacon and still reach the landing
pad? What might explain this change in behaviour?

**Share your `k` value in the workshop chat.**""",
    attempt="""\
import jax.numpy as jnp
from cpomdp import Agent, Belief, CallableSensor, LinearGaussianModel, Preference

from cpomdp_workshop import (
    LAUNCH,
    PAD,
    check_my_sensor,
    compass_planner,
    default_params,
    fly,
    plot_path,
)

beacon_gain = ...  # k: type 0.08 first, then raise it until the path bends
params = default_params(beacon_gain)
# The settings params holds for this task. The only one we're concerned with for this
# exercise is k, "beacon_gain". The rest are chosen for you to best display the
# behaviour:
#   params["beacon_gain"]  k            how fast the noise grows away from the beacon
#   params["beacon_cap"]   4.0          the largest distance_squared that counts
#   params["beacon"]       (2.5, 1.4)   where the beacon is
#   params["noise_floor"]  0.05         the noise at the beacon itself
# It also holds the blackout's settings, for task 3.


def my_sensor(x, params):  # the reading's noise at position x
    distance_squared = jnp.sum((x - params["beacon"]) ** 2)
    pull = ...  # k times jnp.minimum(distance_squared, cap), read from params
    return jnp.eye(2) * (params["noise_floor"] + pull)


check_my_sensor(my_sensor, params)

curious_model = LinearGaussianModel(
    jnp.eye(2),
    control_matrix=jnp.eye(2),
    dynamics_noise=1e-3 * jnp.eye(2),
    prior=Belief(mean=LAUNCH, cov=4 * jnp.eye(2)),
    observation_model=CallableSensor(jnp.eye(2), my_sensor, params),
)
planner = compass_planner(curious_model)  # scores all nine compass moves every step
curious = Agent(curious_model, Preference(PAD, precision=jnp.eye(2)), selector=planner)
flight = fly(curious)
print(f"your k = {beacon_gain}")  # paste in chat
plot_path(flight, my_sensor, params)""",
    bonuses=(
        Bonus(
            title="What else shapes the route?",
            question="""\
You've seen how `k` changes the rocket's route. But `params` holds other settings too.
Under the `params` line in your Task 2 cell, add a line that changes one of them, then
run the cell again. Keep `beacon_gain = 0.4` so that you can compare the routes fairly.
For example:

```python
params["beacon_cap"] = 9.0
```

Try the cap, the noise floor (`noise_floor`) and the beacon's position (`beacon`). Which
of them change the route?""",
            hint="""\
`params["beacon"] = jnp.array([2.5, 0.0])` moves the beacon onto the straight line
between the launch site and the pad. Change one setting at a time, and delete the line
to go back to the original.""",
            answer="""\
The cap and the beacon's position both change the route. With a cap of 1, the path
barely bends, only 0.35 off the straight line. With a cap of 9, the rocket stops 1.42
short of the pad. Moving the beacon onto the straight line, at (2.5, 0), holds the
rocket 1.09 short. The noise floor, on the other hand, barely matters: anywhere from
0.01 to 0.5, the route bends just as far and lands in the same place. What shapes the
route is how the noise differs from place to place, not its overall level.""",
        ),
        Bonus(
            title="How much does the goal matter?",
            question="""\
So far, we've changed the reliability of the rocket's observations. But its behaviour
also depends on how strongly it prefers observations indicating that it has reached the
landing pad.

This preference is controlled by a precision parameter, the `precision=` in
`Preference(PAD, precision=jnp.eye(2))`.

The cell below flies rockets with three preference precisions, 0.1, 1 and 10. It flies
them at two values of `k`: 0.4, and 0.6, where observations near the pad are noisier. A
rocket counts as landed if it finishes within 0.5 of the pad. Run it and compare the
results.

Does increasing preference precision always make the rocket more likely to reach the
landing pad? What happens when observations near the pad are particularly noisy?

**For an additional challenge:** Each rocket flies once, on seed 3. Change `seeds` to
`range(10)` to fly each one over ten seeds. Do your conclusions still hold? Can you
explain how sensor reliability and preference precision interact to influence the
rocket's behaviour?""",
            hint="""\
`landings` builds a new rocket for every seed and every precision, because a rocket can
only fly once. `flight.distance_to(PAD)` is how far from the pad a flight ended. To try
another precision, add it to `(0.1, 1, 10)` in the last loop.""",
            answer="""\
Not always. On seed 3, at `k = 0.4`, a precision of 0.1 doesn't land, while 1 and 10
both do. Above 1, it barely makes a difference: on seed 3, precisions of 1 and 10 fly
exactly the same route. And when observations near the pad are noisy, at `k = 0.6`, no
precision gets the rocket to land. Over seeds 0 to 9 the picture holds. At `k = 0.4`, a
precision of 0.1 lands no rockets, 1 lands 8 and 10 lands 9. At `k = 0.6`, none of the
three lands a single rocket. A higher precision scales the whole expected miss,
including the reading's own noise, so it can't make a noisy pad read any better.""",
            code='''\
import jax.numpy as jnp
from cpomdp import Agent, Preference

from cpomdp_workshop import PAD, compass_planner, default_params, fly, model_with

seeds = [3]  # one flight per rocket. For the additional challenge, try range(10)


def landings(beacon_gain, precision):
    """On how many seeds a rocket with this k and preference precision lands."""
    landed = 0
    for seed in seeds:
        # A rocket can only fly once, so every flight builds a new one. model_with
        # builds the same model as your task 2 cell.
        model = model_with(my_sensor, default_params(beacon_gain))
        preference = Preference(PAD, precision=precision * jnp.eye(2))
        rocket = Agent(model, preference, selector=compass_planner(model))
        flight = fly(rocket, seed=seed)
        landed += flight.distance_to(PAD) < 0.5
    return landed


print(f"landings on seeds {list(seeds)}")
for precision in (0.1, 1, 10):
    print(
        f"precision {precision:>4}:  k = 0.4 lands {landings(0.4, precision)},  "
        f"k = 0.6 lands {landings(0.6, precision)}"
    )''',
        ),
    ),
    poll="""\
### Poll 2

At `k = 0.6` the rocket stops short of the pad. Why?

1. From near the beacon it cannot see the pad.
2. It still looks too few moves ahead.
3. It prefers a *reading* of the pad. At that `k` the pad reads so badly that a spot
   nearer the beacon is expected to read closer to "on the pad".
4. The thruster cost stops it.

<details><summary>Answer</summary>

3. The preference is over readings, and the expected miss includes the reading's own
noise. Where the noise is large, standing on the pad still reads far from it on
average. The curious rocket carries no thruster cost, so 4 cannot apply.

</details>""",
)

SESSION_3 = Session(
    number=3,
    brief="""\
## Task 3 · The blackout

In Task 2, we introduced a beacon that provided more reliable observations nearby. Now,
we'll add a blackout region along the rocket's flight path, where sensor noise increases
sharply towards its centre. There's also a rock inside the blackout, but importantly,
**none of the rockets knows the rock is there**. The rock is in the world (the
generative process) but not in any rocket's generative model (Active Inference book,
Figure 2.2, p. 23).

**Your task:** Complete the two missing lines marked with `...`:

1. In `my_world`, calculate the additional observation noise caused by the blackout,
   using its height and width from `params`.
2. Use `one_node` to represent the original rocket's model as a single-node graph. The
   `check_backends` function will compare this representation with the original Kalman
   backend and report the maximum absolute difference between their results.

Once you've completed both lines, run the cell. The `plot_world` function will compare
three rockets: our original 1960 rocket, your rocket with the selected value of `k`, and
another with `k = 0.20`.

Experiment with the `k` slider and observe how the trajectories change. Do the rockets
follow similar paths? What might explain any differences, given that none of them knows
about the rock?

**Share your `max |difference|` value in the workshop chat.**""",
    attempt="""\
import jax.numpy as jnp
from cpomdp import KalmanBackend

from cpomdp_workshop import check_backends, one_node, plot_world


def my_world(x, params):
    height, width = params["blackout_gain"], params["blackout_width"]
    from_blackout = jnp.sum((x - params["blackout"]) ** 2)
    blindness = ...  # height times jnp.exp of minus from_blackout over width
    return my_sensor(x, params) + jnp.eye(2) * blindness


beacon_gain = 0.08  # @param {type:"slider", min:0.02, max:0.4, step:0.02}

backend = ...  # task 1's model as a graph of one node
check_backends(KalmanBackend(model), backend)  # paste max |difference| in chat
plot_world(my_world, (0.0, beacon_gain, 0.20))""",
    bonuses=(
        Bonus(
            title="Is the rocket really avoiding the rock?",
            question="""\
In our simulation, the curious rockets sometimes navigate around the rock without
colliding. But remember, none of the rockets has any representation of the rock in its
model.

Does this mean they are avoiding the obstacle, or could their behaviour be explained by
something else?

**Your challenge:** Investigate what happens when we change the rock's location while
keeping the sensory environment unchanged.

The cell below flies each rocket over 10 random seeds and records the seeds on which it
collides with the rock. Run it with the rock inside the blackout. Then move the rock to
(4.2, -0.7), an area with more reliable observations, and run it again.

Consider the following questions:

- Does moving the rock change the rocket's planned trajectory?
- How does the collision rate change between the two environments?
- What can we conclude about the relationship between observation uncertainty and
  obstacle avoidance?

**For an additional challenge:** Present your results as a small table or figure and
explain what they tell us about the limitations of the rocket's internal model.""",
            hint="""\
Change `rock_centre` and run the cell again. To check whether a route changes, compare
`fly(...).path` with and without `hazard=rock`, using the same seed.""",
            answer="""\
The rockets aren't avoiding the rock at all. A rock never changes a route: with or
without it, a rocket flies exactly the same path until the moment it collides. Inside
the blackout, the `k = 0.08` rocket hits the rock on 2 of the 10 seeds and the `k =
0.20` rocket on 3, while the 1960 rocket hits it every time. Moved into clear sky at
(4.2, -0.7), the rock is hit on 6 and 5 of the 10 seeds by the curious rockets, and
never by the 1960 rocket. The curious rockets steer away from noisy readings, and the
rock sits where the readings are noisiest. So here the blackout makes them hit it less
often, although they know nothing about the rock. The seed 3 picture also shows the
less common route: on 19 of 30 seeds, the `k = 0.08` rocket passes below the blackout,
not above it.""",
            code='''\
from cpomdp_workshop import curious_rocket, fly, the_1960_rocket

rock_centre = (2.5, 0.0)  # in the blackout. Then try (4.2, -0.7), in clear sky
rock = (rock_centre, 0.42)
seeds = range(10)


def hit_seeds(beacon_gain):
    """Seeds where the rocket with this k hits the rock. k = 0 is the 1960 rocket."""
    found = []
    for seed in seeds:
        if beacon_gain == 0:
            rocket = the_1960_rocket()
        else:
            rocket = curious_rocket(beacon_gain, my_world)
        if fly(rocket, seed=seed, hazard=rock).crashed:
            found.append(seed)
    return found


for rocket_gain, name in ((0.0, "1960 rocket"), (0.08, "k = 0.08"), (0.2, "k = 0.20")):
    found = hit_seeds(rocket_gain)
    print(f"{name:12} hit the rock on {len(found)} of {len(seeds)} seeds: {found}")''',
        ),
        Bonus(
            title="Can we design a more robust rocket?",
            question="""\
We've seen that changing observation reliability can influence the rocket's trajectory.
But how sensitive is its behaviour to the particular parameters we've chosen?

**Your challenge:** Keep the world as it is, with the same blackout, rock and readings,
and change the rocket instead. The cell below runs a small parameter search over two of
the rocket's settings: the `k` it believes the beacon has, and its preference precision.
For each combination, it flies the rocket over seeds 0 to 9 and counts how often it
lands and how often it hits the rock. A landing means finishing within 0.5 of the pad
without hitting the rock.

Run it. Can you identify a combination that produces relatively reliable landings while
reducing collisions?

Now test the combinations on new random seeds that weren't in the search, by changing
`seeds` to `range(10, 30)`.

- Does the apparent improvement persist?
- On the new seeds, what happens if the rock is moved to (4.2, -0.7)?
- How confident can we be that the behaviour generalises beyond the conditions we
  tested?

**Take-home challenge:** Visualise your results using a heatmap showing collision rates
or landing performance across the parameter combinations.""",
            hint="""\
A rocket's model and the world it flies in don't have to match. Each rocket builds its
model with the `k` it believes in, while `fly` always draws the readings from the
world's `k = 0.08`. When you compare a combination with the default, `k = 0.08` with a
precision of 1, count how many flights separate them. Then look at how much the
default's own count changes from one set of seeds to the next.""",
            answer="""\
On seeds 0 to 9, the best combinations believe the beacon is weak: `k = 0.02` with a
precision of 0.3 or 1 lands 9 times and hits the rock once. The default, `k = 0.08` with
a precision of 1, lands 8 times and hits the rock twice. On the new seeds, 10 to 29, the
winner lands all 20 times and the default lands 19, so the lead holds, but only by one
flight each time. That's too small a margin to trust from 30 flights. In fact, the seeds
matter more than the settings: the same default rocket lands 8 of 10 times on the first
seeds and 19 of 20 on the new ones. And if you move the rock to (4.2, -0.7), every
combination hits it on 10 to 12 of the 20 new seeds, the winner and the default on 12
each. The search tuned the rocket to one rock in one place.""",
            code='''\
import jax.numpy as jnp
from cpomdp import Agent, Preference

from cpomdp_workshop import PAD, compass_planner, default_params, fly, model_with

seeds = range(10)  # the search. Then try new seeds, range(10, 30)
rock_centre = (2.5, 0.0)  # in the blackout. Then try (4.2, -0.7), in clear sky
rock = (rock_centre, 0.42)
world_params = default_params(0.08)  # the world's k never changes


def tally(believed_gain, precision):
    """Landings and rock hits over the seeds, for one k and preference precision."""
    landed = hit = 0
    for seed in seeds:
        # A new rocket for every flight. Its model uses the k it believes in.
        model = model_with(my_world, default_params(believed_gain))
        preference = Preference(PAD, precision=precision * jnp.eye(2))
        rocket = Agent(model, preference, selector=compass_planner(model))
        flight = fly(rocket, my_world, world_params, seed=seed, hazard=rock)
        hit += flight.crashed
        landed += not flight.crashed and flight.distance_to(PAD) < 0.5
    return landed, hit


for believed_gain in (0.02, 0.08, 0.20):
    for precision in (0.3, 1, 3):
        landed, hit = tally(believed_gain, precision)
        print(
            f"k = {believed_gain:.2f}, precision {precision:>3}: landed {landed}, "
            f"hit the rock {hit}, of {len(seeds)} flights"
        )''',
        ),
    ),
    poll="""\
### Poll 3

Why did the curious rockets clear the rock in the blackout?

1. They modelled the rock.
2. They looked far enough ahead to see it coming.
3. A reading in the blackout is swamped, so a move into it is expected to read far
   from "on the pad". The rock sits in the blackout.
4. Luck. Another seed would hit it.

<details><summary>Answer</summary>

3. No rocket knows the rock is there, and each looks one move ahead. Where the noise
is large, a reading lands far from "on the pad" on average, so moves into the blackout
score badly. 4 is partly true. Over the first ten seeds the `k = 0.08` rocket clips the
rock twice and the `k = 0.20` rocket three times. The figure is seed 3. Bonus 1 moves
the rock into clear sky where their routes pass, and they hit it more often.

</details>""",
)

SESSIONS = (SESSION_1, SESSION_2, SESSION_3)

CODA_INTRO = """\
## Coda · What can we conclude from our results?

Before we finish, let's consider what the numbers produced by our models actually tell
us.

At each step, our curious rocket evaluates nine possible actions. The planner's
`certificate` records that every one of them was scored. But this can quickly become
expensive: with five possible actions and a planning horizon of seven steps, there are
**78,125 possible action sequences** to evaluate.

Run the final cell and examine the outputs.

**Consider:** What do these results establish about the computations performed, and what
can we not yet conclude about the correctness of our model?"""

CODA = """\
import cpomdp_workshop as workshop

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
from cpomdp_workshop import plot_world

plot_world(
    my_world, (0.0, beacon_gain, 0.20), horizon=2, title="looking two moves ahead"
)""",
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
)
