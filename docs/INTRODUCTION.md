# Introduction

A relational event dataset records a sequence of interactions. A row might say
that actor A sent a message to actor B at 10:03, followed by B replying at
10:05. A relational event model (REM) uses that order directly. For each
observed event, it compares what happened with the events that were possible at
the same point in the sequence.

This differs from fitting a model to an aggregated network. Counts and final
edges alone cannot show whether a reply followed immediately, whether an actor
became more active over time, or which alternatives were available before an
event occurred. In a REM, those details enter through the event history, risk
set, and time-varying statistics.

REMFlow exposes four functions for fitting and assessing these models:

1. `remify` validates an event table and constructs its risk sets;
2. `remstats` calculates statistics for every observed event and alternative;
3. `remstimate` fits the selected ordinal, exact-time, actor-oriented, or
   duration model;
4. `diagnostics` reports fitted probabilities, ranks, residuals, and related
   checks.

`RelationalEventModel` accepts typed misinformation events and calculates actor
role scores, echo-chamber measures, and next-event probabilities. Its
actor-blocking method removes selected actors from the fitted next-event
distribution and renormalizes the remaining probabilities. These quantities do
not identify causal sources or estimate the causal effect of an intervention.

## Implementation choices

The default NumPy backend runs on the CPU. For model families with a JAX
implementation, `jax:cpu` and `jax:gpu` place likelihood and derivative
calculations on the requested device. Requesting `jax:gpu` without an available
GPU raises `BackendUnavailable`.

Histories, statistics, estimates, and diagnostics are regular typed Python
objects. They retain event and coefficient order, expose documented metadata,
and support serialization where noted in the API reference. Formula strings
are parsed by a restricted parser; arbitrary Python code is never evaluated.

See the [usage guide](USAGE.md) for model-specific examples. The project's
README contains the same summary of combinations that are not yet available.
