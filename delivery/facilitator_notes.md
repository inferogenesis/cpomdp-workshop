# Facilitator notes

For the presenters, from the 2026 wording review. Participants do not see these.

## Task 1, bonus 2 · Can unequal sensor noise make the rocket curious?

In this linear-Gaussian setup, unequal noise across the axes need not make information
gain action-dependent. The useful distinction is between a changed overall information
gain and a nonzero **spread across actions**. This bridges into the location-dependent
observation model in Task 2.

## Task 2 · The balancing act

The curious rocket prefers the pad at a precision of 0.1. Expected free energy has no
separate weight on its epistemic term, so this precision is the only setting that trades
information against the pull of the pad. Say plainly that we chose it so the two terms
can compete.

Every curious flight in Task 2 is shown beside the same rocket with curiosity off. The
switch is a helper in the workshop package and not a cpomdp feature. It ranks the same
compass moves by the pragmatic term alone, and with the epistemic term weighted back in
it flies the compass planner's route move for move.

Slides 13 and 14 change different settings. Slide 13 changes the precision at k = 0.2.
Slide 14 keeps the precision at 0.1 and raises k to 0.4:

- At precision 1, curiosity decides almost nothing. At k = 0.2 it changes at most 1 of
  the 36 moves on each of ten seeds, yet both routes bend towards the beacon. The bend
  comes from the pragmatic term, which pays for the reading's noise as well as for its
  distance from the pad.
- At precision 0.1 and k = 0.4, the noise rises steeply enough away from the beacon
  for curiosity to keep the rocket from landing. No curious rocket lands on seeds 0 to
  9, and every incurious one does.

"Curiosity off still lands" holds only in a narrow range of k. On seed 3 the two
rockets part from k = 0.37 to 0.45. From 0.46 the incurious rocket stops short too, and
at k = 0.5 neither lands on any of ten seeds. A learner stepping k by 0.1 from 0.08
reaches 0.38 and then 0.48, so expect both answers in the chat. The explanation is slide
13's: the pragmatic term pays for the reading's noise.

A rocket has landed if it ends within 0.5 of the pad. The plots draw that circle, and
`show_landings` prints the distance for each rocket.

Slide 13 compares the precisions at k = 0.2 because seed 3 stands for the other seeds
there. At k = 0.4 and precision 1, curiosity decides no move on seed 3 but parts the
routes by 1 or more on six seeds of ten.

## Task 2, bonus 1 · Turn the preference up

Raising the precision scales the pragmatic term and leaves the epistemic term alone, so
the gap between the two routes closes. The incurious rocket's route does not depend on
the precision at all. A participant who sees a bend at precision 1 has found a route
that looks curious and is not.

## Task 2, bonus 2 · What does curiosity need?

Each edit acts on ln(1 + σ²/r). A cap of 1 makes r flat along the route, so curiosity has
nothing to choose between. More drift keeps σ² large, so curiosity holds the rocket near
the beacon. A wider prior leaves the curious route alone, because the first reading
resets the rocket's uncertainty and the beacon's field does not reach the launch site.

## Task 3 · Find the pad

The pad is hidden, and the rocket has to learn where it is. Say plainly that we built
this world. The radar is coarse on purpose, the broadcast fades on purpose, and the
rocket's guess is right on average because the true pad is drawn from its own prior.

The goal is a radar reading, whose noise is the same everywhere. That is a choice too.
It keeps the reading's noise out of the pragmatic term, so the pragmatic term cannot do
the beacon's work the way it bent routes in Task 2.

On seed 3 the curious rocket ends 0.17 from the true pad and the incurious one 0.85.
Over seeds 0 to 9 the curious rocket lands on 9 and the incurious one on 3, with median
misses of 0.25 and 0.61. The landing gap is small on the map, so the plot prints both
misses in large type.

The attempt cell builds the model in full, so the room sees that a rocket which has to
learn where the pad is takes a state of four numbers and six readings. The second line
to write is the prior's covariance. cpomdp refuses a covariance of the wrong shape and
its message names the shape it needs, so that mistake has no workshop check.

The number for the chat is the `pad_spread` at which the curious rocket stops finding
the pad. Lowering it changes only the rocket's belief, and the world still draws the pad
with a spread of 1.5. On seed 3 the rocket finds the pad at 0.48 and stops at 0.45, so
expect answers from 0.4 to 0.5 depending on the step people take. Over ten seeds the
change is gradual: the curious rocket lands on 7 seeds at 0.5 and on 6 at 0.4.

`check_my_broadcast` grades the learner's broadcast on its values before anything flies.
The mistake to expect is a broadcast that returns a matrix, as Task 2's sensor did. The
troubleshooting cell lists it.

## Task 3, bonus 1 · How sure is the rocket?

An overconfident rocket, with a spread of 0.1, believes there is nothing to learn. It
visits the beacon on no seed and lands on 1 of 10, the same as with curiosity off. This
bridges into slide 18.

## Task 3, bonus 2 · When is curiosity worth it?

"Why not use a better radar?" is the question a sharp attendee asks first, and this
bonus answers it. With a sharp radar, a broadcast that carries far or a capped
broadcast, both rockets land on all ten seeds. With a sharp radar the curious rocket
still detours on 4 seeds, and the detour buys it nothing. The task is about whether a
rocket goes to get information that is only available somewhere else.

## Slide 18 · The balance takes tuning

The window is narrow. At a precision of 0.3 the curious rocket lands on 7 seeds, and at
1 it lands on 3, the same as with curiosity off. Longer horizons and factor-graph
backends are a direction for future work. The one measured point is that, looking two
moves ahead, the curious rocket lands on all ten seeds. The workshop has no factor-graph
evidence: the one-node check shows the graph engine matches the Kalman filter and says
nothing about stability. Say "we expect" and "planned".

## Scientific framing to preserve when presenting

1. Fixed observation noise can yield constant relative information gain across actions,
   even if overall uncertainty or anisotropy changes.
2. A bend towards less noisy readings does not, by itself, establish an epistemic
   motive. In Task 2 the reading's noise enters the pragmatic term, and only the
   comparison with curiosity off shows which term chose a move.
3. Task 3's world was built so that curiosity pays off. A rocket that finds the pad
   there shows what curiosity does in that world. Whether it generalises is untested.
4. A computation log, backend agreement test, or successful demonstration supports a
   **specific** claim; it is not a general guarantee of model correctness or
   generalisation.
