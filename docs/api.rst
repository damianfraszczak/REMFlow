API reference
=============

Install one distribution and import its public objects from ``remflow``.
History preparation, statistic construction, estimation, and diagnostics are
functions in that namespace rather than separate packages.

Typical workflow
----------------

A typical analysis has four stages:

1. :func:`remify` validates and normalizes ordered events and constructs their
   opportunity sets.
2. :func:`remstats` converts a formula or effect builders into time-varying
   statistic matrices.
3. :func:`remstimate` fits the requested ordinal, exact-time, actor-oriented,
   or duration model.
4. :func:`diagnostics` calculates prediction ranks, recall, residuals, and
   plot-ready diagnostic data.

Each stage returns a typed object with documented attributes and
dictionary-style access. Histories and statistic objects also support JSON
serialization.

.. code-block:: python

   import pandas as pd
   from remflow import diagnostics, remify, remstats, remstimate

   events = pd.DataFrame(
       {
           "time": [1.0, 2.0, 3.0],
           "sender": ["A", "B", "A"],
           "receiver": ["B", "A", "C"],
       }
   )
   history = remify(events, actors=["A", "B", "C"], ordinal=False)
   statistics = remstats(history, tie_effects="~ inertia() + reciprocity()", first=2)
   fit = remstimate(history, statistics, backend="numpy")
   report = diagnostics(fit, history, statistics)

Event histories and risk sets
-----------------------------

``remify`` is the public boundary between tabular input and numerical model
state. Public actor, dyad, type, and event identifiers are 1-based; internal
kernels may use zero-based arrays.

.. autofunction:: remflow.remify

.. autoclass:: remflow.EventHistory
   :members: summary, to_dict, to_json, from_json, plot

.. autoclass:: remflow.DurationHistory

Statistics and formulas
-----------------------

Formulas can be supplied as restricted strings or assembled from typed effect
builders. Formula parsing does not evaluate arbitrary Python code. Term order
determines coefficient order.

.. autofunction:: remflow.formula

.. autofunction:: remflow.remstats

.. autoclass:: remflow.Effect
   :members:

.. autoclass:: remflow.Formula
   :members:

.. autoclass:: remflow.RemStats
   :members: summary, to_dict, to_json, from_json, plot, boxplot

.. autoclass:: remflow.AomStats
   :members: summary, to_dict, to_json, from_json

Frequently used effect builders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   remflow.inertia
   remflow.reciprocity
   remflow.send
   remflow.receive
   remflow.tie
   remflow.dyad
   remflow.event
   remflow.same
   remflow.difference
   remflow.average
   remflow.minimum
   remflow.maximum
   remflow.isp
   remflow.itp
   remflow.osp
   remflow.otp
   remflow.psABBA
   remflow.recencySendSender
   remflow.recencyReceiveReceiver

Estimation and results
----------------------

``backend`` selects the array implementation independently of the optimizer.
``numpy`` is the CPU default. For models with a JAX implementation,
``jax:cpu`` and ``jax:gpu`` place likelihood and derivative calculations on the
requested device.

.. autofunction:: remflow.remstimate

.. autofunction:: remflow.fit_rem

.. autoclass:: remflow.RemEstimate
   :members: summary, to_dict, plot

.. autoclass:: remflow.RemEstimateDuration
   :members: summary, to_dict

.. autoclass:: remflow.Diagnostics
   :members: to_dict, plot_data, plot

.. autoclass:: remflow.DurationDiagnostics
   :members: to_dict, plot_data, plot

Model evaluation
~~~~~~~~~~~~~~~~

.. autosummary::

   remflow.diagnostics
   remflow.AIC
   remflow.AICC
   remflow.BIC
   remflow.WAIC
   remflow.bic_table

Extended estimators
~~~~~~~~~~~~~~~~~~~

Availability depends on model family and backend. Unsupported combinations
raise explicit exceptions; the current boundaries are summarized in the main
README.

.. autosummary::

   remflow.frailty_rem
   remflow.remfrailty
   remflow.rempenalty
   remflow.remixture
   remflow.dlcrem
   remflow.remwindow
   remflow.remtribute

Backend selection
-----------------

.. autofunction:: remflow.resolve_backend

.. autofunction:: remflow.available_backends

.. autoexception:: remflow.BackendUnavailable

Misinformation workflow
-----------------------

:class:`RelationalEventModel` provides the high-level typed-event workflow for
next-event ranking, descriptive actor roles, echo-chamber trajectories, and
conditional actor-blocking interventions. These outputs are descriptive and
predictive; they are not causal source-identification results.

.. autoclass:: remflow.RelationalEventModel
   :members: fit, summary, predict_next_events, actor_roles, detect_sources, echo_chamber_metrics, simulate_intervention
