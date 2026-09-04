# Quick start

## Install

REMFlow requires Python 3.11 or newer. The recommended installation uses the
released `remflow` package from PyPI:

```bash
python -m pip install remflow
```

JAX is optional. Install the extra if the analysis will use a JAX backend:

```bash
python -m pip install "remflow[gpu]"
```

The appropriate JAX CUDA build still depends on the operating system, driver,
and CUDA installation. Follow the JAX installation instructions before using
`backend="jax:gpu"`.

A source checkout is needed only for development. In that case, use:

```bash
python -m pip install -e ".[dev,docs]"
```

## Fit a model

Start with a pandas data frame containing an event time, sender, and receiver:

```python
import pandas as pd

from remflow import diagnostics, remify, remstats, remstimate

events = pd.DataFrame(
    {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "sender": ["A", "B", "A", "C", "B"],
        "receiver": ["B", "A", "C", "A", "C"],
    }
)

history = remify(events, actors=["A", "B", "C"], ordinal=False)
statistics = remstats(
    history,
    tie_effects="~ inertia() + reciprocity() + otp()",
    first=2,
)
fit = remstimate(history, statistics, backend="numpy")

print(fit.summary())
print(diagnostics(fit, history, statistics).summary())
```

Here, `ordinal=False` selects an exact-time intensity model. Use
`ordinal=True` for a conditional event-choice model. Public actor and dyad IDs
are 1-based, while the original labels remain available on the history object.

## Select a JAX device

```python
fit = remstimate(history, statistics, backend="jax:gpu")
```

This requires a physical JAX GPU device. REMFlow raises `BackendUnavailable`
if none is visible. Use `backend="jax:cpu"` when the calculation must run with
JAX on the CPU. Reported NumPy/JAX comparisons use float64.

## Where to go next

- The [usage guide](USAGE.md) covers actor models, duration data, sampled risk
  sets, GPU selection, and misinformation examples.
- The [API reference](api.rst) lists public functions and result objects.
- [Performance evaluation](BENCHMARKING.md) documents the benchmark commands.
- The [classroom event study](CLASSROOM_EVENT_STUDY.md) explains the
  tutorial-derived example and its interpretation limits.
