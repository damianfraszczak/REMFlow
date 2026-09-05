# Usage guide

This page starts with a pandas event table and adds model components one at a
time. The examples are deliberately small; real analyses should check risk-set
size, convergence, and sensitivity to the chosen effects.

## Input data

REMFlow expects a table with at least two actor columns. The following column
name pairs are recognized:

- `sender`, `receiver`
- `actor1`, `actor2`
- `source`, `target`
- `from`, `to`

`time`, event type, and event weight columns are optional. Their names are
passed to `remify` when they differ from the defaults.

```python
import pandas as pd

events = pd.DataFrame(
    {
        "time": [1, 2, 3, 4],
        "sender": ["A", "B", "A", "C"],
        "receiver": ["B", "A", "C", "A"],
        "channel": ["chat", "chat", "ticket", "chat"],
    }
)
```

## Build an event history

```python
from remflow import remify

history = remify(
    events,
    actors=["A", "B", "C"],
    event_type="channel",
    riskset="full",
    ordinal=True,
)

print(history.summary())
```

`ncores` controls risk-set construction. The default, `ncores=1`, follows the
serial reference path. Larger values use a thread pool without changing event
or risk-set order:

```python
parallel_history = remify(
    events,
    actors=["A", "B", "C"],
    event_type="channel",
    riskset="full",
    ordinal=True,
    ncores=4,
)
```

The thread-pool startup cost can outweigh the benefit for small histories, so
measure this option on the data being analyzed. `ncores` must be a positive
integer.

## Calculate statistics

```python
from remflow import remstats

stats = remstats(
    history,
    tie_effects="~ inertia() + reciprocity() + send() + receive()",
    first=2,
)

print(stats.summary())
```

## Fit the model

```python
from remflow import remstimate

fit = remstimate(history, stats, backend="numpy")
print(fit.summary())
```

`engine="auto"` currently selects SciPy. Use `engine="scipy"` to make that
choice explicit. The separate `backend` argument controls NumPy or JAX
evaluation of the objective and derivatives.

For a large tie risk set, case-control sampling keeps the observed dyad and
draws a reproducible set of alternatives. The resulting inverse-probability
weights are used by both ordinal and exact-time likelihoods:

```python
sampled = remstats(
    history,
    tie_effects="~ inertia() + reciprocity() + otp()",
    first=1,
    sampling=True,
    samp_num=100,
    seed=2026,
)
sampled_fit = remstimate(history, sampled, backend="numpy")
```

## Actor-oriented models

Actor-oriented ordinal models estimate sender rates and receiver choices as
separate likelihood components:

```python
from remflow import aomstats, remify, remstimate

actor_history = remify(
    events,
    actors=["A", "B", "C"],
    model="actor",
    ordinal=True,
)
actor_stats = aomstats(
    reh=actor_history,
    sender_effects="~ 0 + indegreeSender()",
    receiver_effects="~ 0 + inertia() + reciprocity()",
    first=2,
)
actor_fit = remstimate(actor_history, actor_stats, backend="numpy")

print(actor_fit.sender_model.summary())
print(actor_fit.receiver_model.summary())
```

For exact-time actor histories, REMFlow fits a sender-rate model and a
conditional receiver-choice model. Simultaneous event groups are supported,
and the NumPy and JAX CPU paths are checked for numerical agreement.

## Duration events

Duration histories use one dynamic timeline for tie starts and endings. Active
ties are end-process alternatives and are excluded from the start risk set,
including at the exact time at which they end.

```python
from remflow import diagnostics, remify, remstats, remstimate, stack_stats

duration_events = pd.DataFrame(
    {
        "time": [1, 2, 5],
        "actor1": ["A", "B", "A"],
        "actor2": ["B", "C", "C"],
        "end": [6, 7, 8],
    }
)
duration_history = remify(duration_events, duration=True, model="tie")
duration_stats = remstats(
    duration_history,
    start_effects="~ inertia() + activeOutdegreeSender()",
    end_effects="~ inertia() + activeDegreeMin()",
    psi_start=1,
    psi_end=1,
)
duration_design = stack_stats(duration_stats).remstats_stack
duration_fit = remstimate(duration_history, duration_stats, backend="numpy")
duration_diagnostics = diagnostics(duration_fit)
```

Right-censored events stay in the end risk set but have no observed ending.
Completed events contribute to history statistics with
`event_weight * (duration + 1)**psi`; ongoing ties contribute to the binary
`active*` effects. Interval histories use a joint Poisson likelihood with
`log_interevent` as an offset. With `ordinal=True`, estimation instead uses a
conditional likelihood over each start/end risk-set stratum, including tied
cases. The NumPy and JAX CPU implementations are checked against each other.

### Duration moving windows

`remwindow` partitions a duration model by complete timeline strata. A time
point and all of its start/end alternatives always remain in the same window,
including simultaneous starts or endings:

```python
from remflow import remwindow

window_fit = remwindow(
    duration_history,
    duration_stats,
    window_width=3,
    min_events=1,
)
window_diagnostics = diagnostics(window_fit, duration_history, duration_stats)

print(window_fit.windows)
print(window_diagnostics.start["summary"])
print(window_diagnostics.end["summary"])
```

In duration results, `start_event`, `end_event`, and `n_events` refer to ordered
time strata. The `n_strata` column and
`metadata["window_unit"] == "duration_time_strata"` record that unit
explicitly. Diagnostic interpolation follows the actual start/end timeline,
not row positions in the stacked design matrix. The small dataset above emits
a low-events-per-parameter warning; use larger windows for substantive work.

## GPU backend

```python
from remflow import BackendUnavailable, remstimate

try:
    fit = remstimate(
        history,
        stats,
        backend="jax:gpu",
        riskset_chunk_size=50_000,
    )
except BackendUnavailable as exc:
    print(f"GPU backend is not available: {exc}")
```

An explicit GPU request never runs on the CPU silently. Validated accelerator
paths use JAX float64. `riskset_chunk_size` limits device memory use for the
supported JAX tie, actor, and unpenalized ordinal-duration paths. Benchmark
reports separate compilation from steady-state execution.

## High-level relational event model

`RelationalEventModel` is a general convenience facade over `remify`,
`remstats`, and `remstimate`. Three-field tuples represent untyped events; a
fourth field represents the event type. Data frames can instead name a custom
type column with `event_type`.

```python
from remflow import RelationalEventModel

events = [
    (1.0, "A", "B"),
    (2.0, "B", "C"),
    (3.0, "A", "C"),
]

model = RelationalEventModel(
    effects=("reciprocity", "sender_activity"),
    ordinal=True,
).fit(events)

print(model.summary())
print(model.predict_next_events())
```

For typed data-frame input, pass for example `event_type="channel"` and, when
needed, `event_attributes=("sentiment",)`. The facade expands the risk set by
type automatically when multiple types are observed; set
`extend_riskset_by_type` explicitly to override that behavior.

## Misinformation analysis

`MisinformationModel` inherits the general fitting, summary, and prediction
workflow from `RelationalEventModel`. It adds the positional `action` and
`stance` convention and the domain-specific analyses below.

```python
from remflow import MisinformationModel

events = [
    (1.2, "u1", "u5", "retweet", "support"),
    (1.8, "u5", "u8", "reply", "deny"),
    (2.1, "u2", "u5", "mention", "question"),
]

model = MisinformationModel(
    effects=("reciprocity", "sender_activity", "receiver_popularity"),
    ordinal=True,
).fit(events)

print(model.predict_next_events())
print(model.actor_roles())
print(model.detect_sources(top_k=3))

echo = model.echo_chamber_metrics()
print(echo["echo_chamber_score"])
print(echo["trajectory"])

intervention = model.simulate_intervention(blocked_actors=["u5"])
print(intervention["probability_mass_removed"])
```

`stance_similarity` uses the latest observed stance sent by each actor. It does
not infer embeddings from text or update an external language model.
`detect_sources()` ranks actors from observed timing and reach; it is not a
latent-source posterior. `simulate_intervention()` conditions the next-event
distribution on blocked actors rather than simulating an entire cascade.

## Support notes

Tie and actor MLE/HMC, sampled tie likelihoods, and interval or ordinal duration
MLE are implemented. Frequentist frailty, penalties, finite mixtures, Bayesian
shrinkage, moving windows, and non-duration attribution are available primarily
with NumPy. Tie and actor HMC expose posterior summaries and acceptance,
divergence, and energy diagnostics.

The following combinations are not available:

- actor case-control sampling;
- Bayesian frailty and general Bayesian duration inference;
- WAIC for mixtures, random-effects models, and duration fits;
- non-constant mixture concomitant formulas;
- selected extended estimators with a JAX backend;
- duration attribution.

These requests raise an exception instead of changing the requested model. The
project README contains the same compatibility summary; the
[API reference](api.rst) provides function signatures.
