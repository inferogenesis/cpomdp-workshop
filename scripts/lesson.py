"""What the notebook says, cell by cell. ``build_notebook.py`` assembles it.

Each session is a brief, an attempt cell with two ``...`` lines to write, two bonuses
and a poll. The hints and the reference cell are not here. The hints live in
``delivery/hints_chat.txt``, and the reference cell is the attempt with both hints
put in, so neither can drift from the attempt.
"""

from dataclasses import dataclass

TITLE = """\
# Build a curious rocket

A hands-on session on cpomdp, continuous active inference in Python. You build three
rockets. The first is the 1960 controller, a Kalman filter with a goal. In the last,
three rockets fly past a rock none of them was told about.

Run the cells in order. Each task has an attempt cell with two lines for you to write,
two hints and a collapsed reference cell. Using the reference is fine. The checkpoint
after each task puts the reference names in place, so a task that went wrong costs
nothing later.

Ask anonymously on the form (link to come)."""

INSTALL = """\
# Installs the workshop on Colab, then checks the versions it was measured on.
import sys

if "google.colab" in sys.modules:
    !pip install -q cpomdp-workshop=={version}
import cpomdp_workshop as workshop

workshop.hello()"""

PREVIEW_INTRO = "*This is what you will have built by 1:32.*"

PREVIEW = "workshop.preview()"

PREFLIGHT = """\
Missed the three questions at registration? They take thirty seconds: the pre-flight
form (link to come)."""

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
## Task 1 · the 1960 rocket

The rocket flies from the launch site to the pad. It reads its position through a
sensor whose noise is the same everywhere. That is the Kalman filter's world. LQR is
its controller.

Write the two `...` lines. The rest is written. `check_agent` prints the rocket you
built and flies it. Then `probe_model` asks cpomdp whether any of the nine compass
moves would teach the rocket more than another.

Then try to make it curious. Change the noise level, the prior's spread, the step
size or the goal, and probe again.

**Paste your spread in chat.**""",
    attempt="""\
import jax.numpy as jnp
from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal, probe_model

from cpomdp_workshop import LAUNCH, PAD, check_agent, compass_moves, show_probe

observation_matrix = ...  # C (pymdp's A): the reading sees the whole position
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
            title="change the other numbers",
            question="""\
In your task 1 cell, change one number at a time and run the cell again:

- the `0.05` in `observation_noise`, the noise level
- the `4` in `prior`, how unsure the rocket starts
- the step size, by writing `compass_moves(1.5)` in place of `compass_moves()`

Watch the last line it prints. Does the spread ever leave 0.000000?""",
            hint="""\
A bigger number in front of `jnp.eye(2)` is more noise or more doubt.
`compass_moves(1.5)` makes every move 1.5 long.""",
            answer="""\
Never. Whatever the noise level, the starting doubt or the step, the spread stays
0.000000. The noise is the same everywhere, so every move leaves the rocket equally
unsure. Only noise that changes with position can make the moves differ. Run the
checkpoint to put the task's own numbers back.""",
        ),
        Bonus(
            title="a sensor sharper in one direction",
            question="""\
What happens if the sensor's noise differs between x and y? Predict first: will moves
along the sharper axis teach the rocket more? Then try it in the cell below. Compare
how much one reading teaches, as well as the spread.

For an additional challenge, make the starting doubt differ between x and y too. What
changes, and what stays the same? Explain it with how the Kalman filter updates its
uncertainty.""",
            hint="""\
`jnp.diag(jnp.array([0.05, 2.0]))` builds a matrix with 0.05 and 2.0 on its diagonal.
Use it as `observation_noise` in a copy of your task 1 cell. `report.epistemic_range`
holds the least and the most one reading teaches over the nine moves.""",
            answer="""\
Each reading teaches less: 2.7467 nats with 0.05 on x and 2.0 on y, against 4.3947 in
task 1. Every move still teaches the same, so the spread stays 0.000000. An uneven
start changes the amount again and leaves the spread at 0. The Kalman filter's
uncertainty update uses the noise and the old uncertainty. It never uses the move or
the position. With the same noise everywhere, a move changes where the rocket is and
nothing about how well it sees.""",
            code="# Your code here. A copy of your task 1 cell is a good start.",
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
## Task 2 · a beacon

Now the reading depends on where the rocket is. Near the nav beacon it is sharp.
Away from it, the noise grows with the squared distance up to a cap. `k` sets how
fast it grows.

Pick `k` first. `default_params(k)` puts k and the world's other settings into
`params`, a dictionary. A setting is read by its name in square brackets, as in
`params["beacon_cap"]`. The attempt cell lists every setting this task reads.

Then write the beacon's pull into `my_sensor`. It is a function. Given a position `x`,
it returns the reading's noise there. `check_my_sensor` flies a rocket with your sensor
and says what it did.

The goal changes too. This rocket prefers a reading, the one that says "on the
pad". Every step it scores all nine compass moves and takes the one that best trades
getting there against reading clearly.

Start at `k = 0.08`. Find a `k` that bends the path up to the beacon.

**Paste your k in chat.**""",
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
            title="change the beacon's other settings",
            question="""\
Under the `params` line in your task 2 cell, add one line that changes another
setting, then run the cell. Keep `beacon_gain = 0.4` so the routes compare. For
example:

```python
params["beacon_cap"] = 9.0
```

Try the cap, the floor (`noise_floor`) and the beacon's position (`beacon`). Which
ones move the route?""",
            hint="""\
`params["beacon"] = jnp.array([2.5, 0.0])` puts the beacon on the straight line.
Change one setting at a time, and delete the line to go back.""",
            answer="""\
The cap and the beacon's position move the route. At cap 1 the path barely bends,
0.35 off the straight line. At cap 9 the rocket stops 1.42 short of the pad. A beacon
on the straight line, at (2.5, 0), holds it 1.09 short. The floor barely matters. From
0.01 to 0.5 the route bends just as far and lands in the same place. The route follows
how the noise differs from place to place. Its overall level barely counts.""",
        ),
        Bonus(
            title="how much does the goal matter?",
            question="""\
The route also depends on how strongly the rocket prefers readings that say it is on
the pad. That strength is the preference's precision, the `precision=` in
`Preference(PAD, precision=jnp.eye(2))`.

Keep the sensor and `k` fixed. Build several rockets with different precisions, fly
each one, and compare their routes.

Does a higher precision always make the rocket more likely to reach the pad? What
happens when readings near the pad are particularly noisy?

For an additional challenge, repeat the comparison at two values of `k`. How do the
sensor's noise and the preference's precision work together to shape the route?""",
            hint="""\
`Preference(PAD, precision=10 * jnp.eye(2))` sets precision 10. A rocket flies only
once, so build a fresh model, planner and `Agent` for each precision. Try 0.1, 1 and
10 at `k = 0.4`, then at `k = 0.6`, over several seeds with
`fly(rocket, seed=seed)`.""",
            answer="""\
Not always. From very loose to normal it helps a lot. At `k = 0.4`, over seeds 0 to 9,
precision 0.1 lands no rocket and precision 1 lands 8. Above 1 it barely matters: at
seed 3, precisions 1, 3 and 10 fly the same route. When the pad's readings are noisy,
at `k = 0.6`, no precision lands the rocket. It lands 0 of 10 at 0.1, at 1 and at 10.
A higher precision scales the whole expected miss, the reading's noise included, so it
cannot make a noisy pad read better. A landing here means ending within 0.5 of the
pad.""",
            code="# Your code here. Start from the flight lines of your task 2 cell.",
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
## Task 3 · the blackout

Put a blackout on the flight path. Near its centre the reading is swamped. There is a
rock in it, and no rocket models the rock.

Write the blackout term into `my_world`. Its height and width are read from `params`
for you. The slider sets `k` for your rocket.
`plot_world` flies three rockets past the rock: the 1960 rocket, yours and one at
`k = 0.20`.

Before the plot, the 1960 rocket runs on a second engine. `one_node` rewrites its
model as a graph of one node, the form cpomdp uses for branching models.
`check_backends` flies it on both.

**Paste max |difference| in chat.**""",
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
            title="is the rocket really avoiding the rock?",
            question="""\
The curious rockets fly round the rock at seed 3, yet no rocket has the rock in its
model. Are they avoiding it, or does something else explain it?

The cell below flies each rocket on 10 seeds and lists the seeds where it hits the
rock. Run it with the rock in the blackout. Then move the rock to (4.2, -0.7), where
readings are clear, and run it again. The world stays the same.

- Does moving the rock change the route a rocket flies?
- How do the hits change between the two places?
- What does that say about readings and avoiding obstacles?

For an additional challenge, put both results in a small table or figure. What do they
say about the limits of the rocket's model?""",
            hint="""\
Change `rock_centre` and run the cell again. To check a route, compare `fly(...).path`
with and without `hazard=rock` on the same seed.""",
            answer="""\
No rocket is avoiding the rock. A rock never changes a route. With or without it, a
rocket flies the same path until the moment it hits. In the blackout, the `k = 0.08`
rocket hits on 2 of 10 seeds and the `k = 0.20` rocket on 3, while the 1960 rocket hits
on all 10. In clear sky at (4.2, -0.7), they hit on 6 and 5 of 10, and the 1960 rocket
on none. The curious rockets keep away from noisy readings, so a rock is only safe from
them where the readings are noisy. The seed 3 picture also shows the less common
route: on 19 of 30 seeds the `k = 0.08` rocket passes below the blackout.""",
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


for beacon_gain, name in ((0.0, "1960 rocket"), (0.08, "k = 0.08"), (0.2, "k = 0.20")):
    found = hit_seeds(beacon_gain)
    print(f"{name:12} hit the rock on {len(found)} of {len(seeds)} seeds: {found}")''',
        ),
        Bonus(
            title="can you design a more robust rocket?",
            question="""\
The world stays as it is: the blackout, the rock and the readings. What you can change
is the rocket, through the `k` it believes the beacon has and its preference
precision. How much does its behaviour depend on those two choices?

Search a small grid: three values of `k` and three precisions. For each pair, fly 10
seeds and count how often the rocket lands and how often it hits the rock. Which pair
lands most reliably with the fewest hits?

Then fly your chosen pair on 20 seeds the search did not use.

- Does the improvement persist?
- What happens if the rock moves to (4.2, -0.7)?
- How confident can you be that the result holds beyond the conditions you tested?

For an additional challenge, draw the landing or hit counts as a heatmap over the
grid.""",
            hint="""\
A rocket's model and the world can differ. Build the model with
`model_with(my_world, default_params(k))`, give the rocket
`Preference(PAD, precision=c * jnp.eye(2))` and `compass_planner(model)`, then fly it
in the fixed world: `fly(rocket, my_world, default_params(0.08), seed=seed,
hazard=ROCK)`. A flight lands if it did not crash and ends within 0.5 of the pad. Try
`k` at 0.02, 0.08 and 0.20, and precision at 0.3, 1 and 3. The grid is 90 flights.""",
            answer="""\
On seeds 0 to 9 the best pairs believe in a weak beacon. `k = 0.02` with precision
0.3 or 1 lands 9 times and hits the rock once. The default, `k = 0.08` with precision
1, lands 8 and hits twice. On seeds 10 to 29 the winner lands all 20 and the default
19, so the lead holds, by one flight each time. That margin is too small to trust from
30 flights. The seeds matter more than the setting: the same default rocket lands 8 of
10 on the first seeds and 19 of 20 on the new ones. Move the rock to (4.2, -0.7) and
every pair hits it on 12 of the 20 new seeds, winner and default alike. The search
tuned the rocket to one rock in one place.""",
            code="# Your code here. The hint shows a rocket flown in a fixed world.",
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
the rock into clear sky where their routes pass, and they hit it far more often.

</details>""",
)

SESSIONS = (SESSION_1, SESSION_2, SESSION_3)

CODA_INTRO = """\
## Coda · what a printed number entitles you to say

Every step, the curious rocket scored all nine moves. Its certificate counts them.
The same search over five moves, seven deep, scores all 78125 plans.

The RxInfer oracle covers the fixed-noise case. Handed the curious model, it refuses
before Julia starts."""

CODA = """\
import cpomdp_workshop as workshop

print(curious.selector.certificate)
print(workshop.axis_certificate())
print(workshop.rxinfer_refusal(curious_model))"""

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
