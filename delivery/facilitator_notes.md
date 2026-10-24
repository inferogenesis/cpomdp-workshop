# Facilitator notes

For the presenters, from the 2026 wording review. Participants do not see these.

## Task 1, bonus 2 · Can unequal sensor noise make the rocket curious?

In this linear-Gaussian setup, unequal noise across the axes need not make information
gain action-dependent. The useful distinction is between a changed overall information
gain and a nonzero **spread across actions**. This bridges into the location-dependent
observation model in Task 2.

## Task 2, bonus 2 · How much does the goal matter?

This challenge explores preferences over **observations**, not states. A stronger
preference for a particular observation is not guaranteed to produce a better physical
landing, particularly when observations around the pad are noisy.

## Task 3, bonus 2 · Can we design a more robust rocket?

The rocket has no representation of the obstacle. Even if tuning its parameters reduces
collisions in a chosen environment, that does not demonstrate that the agent has learned
to represent or anticipate the rock. Parameter selection can indirectly incorporate the
experimenter's knowledge of the obstacle.

## Scientific framing to preserve when presenting

1. Fixed observation noise can yield constant relative information gain across actions,
   even if overall uncertainty or anisotropy changes.
2. A preference for precise observations does not, by itself, establish an epistemic
   motive for information gain. In Task 2, expected observation outcomes and their
   uncertainty affect action evaluation.
3. A trajectory that happens to miss an unmodelled rock is not evidence that the rocket
   represented or deliberately avoided it.
4. A computation log, backend agreement test, or successful demonstration supports a
   **specific** claim; it is not a general guarantee of model correctness or
   generalisation.
