REMFlow documentation
======================

REMFlow fits relational event models to ordered sender-receiver data. It
constructs event histories and risk sets, calculates time-varying network
statistics, and estimates ordinal, exact-time, actor-oriented, and duration
models. The fitted objects provide coefficient estimates, event probabilities,
residuals, ranks, and other model diagnostics.

Installation
============

Install the released ``remflow`` package from PyPI:

.. code-block:: console

   python -m pip install remflow

A source checkout is required only for development. See the
:doc:`QUICK_START` page for optional JAX installation and development setup.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   INTRODUCTION
   QUICK_START
   USAGE

.. toctree::
   :maxdepth: 2
   :caption: Studies and performance

   BENCHMARKING
   CLASSROOM_EVENT_STUDY

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
