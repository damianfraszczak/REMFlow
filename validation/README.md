# Numerical validation and fitting-cost measurements

This directory provides reproducible experiments that evaluate the numerical correctness and computational cost of REMFlow. The study compares Python and R estimates under matched model specifications, independently checks statistic construction, measures fitting time and process memory, and reports automated test results. The recorded experiments were executed on 1 November 2026 using REMFlow 0.1.2.

## What was executed

- Eight exact-time classroom models: four nested specifications on each day, compared with `relevent::rem.dyad` under matched maximum-likelihood specifications.
- One simulated event history fitted in Python and R. Its AB–BA parameter is separated; R reaches the iteration limit. This is a documented limitation, not a successful finite-coefficient equivalence test.
- Independent R construction of all nine risk-set statistic arrays: maximum absolute difference from Python is zero.
- Complete fitting time and Windows process peak working set for NumPy and the JAX CPU selector, plus a separately compiled JAX kernel.
- A focused test suite: 50 passed and four physical-GPU tests skipped; statement coverage is 2,984/6,845 (43.59%). This is not a complete-suite result.

The small, recorded outputs are in [`reference_results/`](reference_results/). Fresh executions write to ignored `results/` and never overwrite the reference results. Large binary design arrays, installed R packages, tutorial copies, document files, and temporary logs are not versioned here.

## Installation

From the repository root, in a Python environment:

```sh
python -m pip install -e ".[dev,gpu]"
python -m pip install -r validation/requirements.txt
```

Install R and `relevent` separately. The recorded comparison used R 4.3.2 and relevent 1.2-1. A newer package version is a new validation condition; record its version rather than assume identical results.

```r
install.packages("relevent", repos="https://cloud.r-project.org")
```

The original Python environment used Python 3.12.0, NumPy 2.3.5, SciPy 1.18.0, JAX/jaxlib 0.7.2, and psutil 7.2.2. The installation commands above resolve the project's supported dependencies and do not promise a bitwise recreation of the original environment.

## Reproduce the cross-language comparison

Run these commands in order from the repository root:

```sh
python validation/compare_export.py
Rscript validation/check_designs.R
Rscript validation/compare_relevent.R
```

`compare_export.py` fits the Python models and exports their event histories and full design arrays. `check_designs.R` independently reconstructs the classroom preprocessing and all covariates, then checks the arrays. `compare_relevent.R` performs fresh R fits through time-varying `CovEvent` terms; it does not reuse cached fitted models.

Both implementations use full directed risk sets without self-events, an empty initial history, exposure from time zero, and every observed event. R receives an additional censoring row one floating-point increment after the last event, because `rem.dyad` treats its final row as the observation endpoint. Classroom fits use BFGS, `reltol=1e-12`, and at most 10,000 iterations. The separated simulation is limited to 200 iterations. Standard errors use observed information; BIC uses the number of actual events.

For the eight classroom fits, the comparison thresholds are absolute differences of 2e-4 for coefficients, 2e-5 for standard errors, and 1e-6 for log-likelihood and BIC. Recorded results satisfy these thresholds. They are not applied to the separated simulation parameter. The simulation's large standard errors and conventional BIC are numerical outputs without the usual regular-model interpretation.

The classroom example uses cumulative contact counts in place of the source tutorial's dyadic recency effects. The simulation changes the tutorial's coefficient assignments and recency definition. The comparison validates these explicitly specified adaptations, not exact reproduction of the tutorial models or identical cross-language random streams. Data provenance and third-party data licensing remain in [`data/classroom_events/`](../data/classroom_events/).

## Reproduce resource measurements (Windows)

Run each command in a separate process:

```sh
python validation/measure_fit.py numpy 20 200
python validation/measure_fit.py jax:cpu 20 200
```

These measurements use OS process high-water marks available through psutil on Windows; the script explicitly rejects other systems. Timing includes event generation, histories, statistics, backend initialization and fitting, but excludes imports. BLAS and OpenMP thread counts are set to one. Other machine processes were active in the original run, so the recorded single-run times are descriptive, not a controlled speed comparison or a maximum capacity estimate.

The standard tie-oriented MLE in the audited source uses NumPy/SciPy for both selectors. Selecting JAX does not prove that the fitted model ran on a GPU. The JAX measurement additionally times a separate exact-time kernel, including lowering, compilation and synchronized first execution; this kernel is not called by the measured fitting routine. No complete GPU fit was measured.

## Reproduce the focused tests

```sh
python -m coverage run --source=src/remflow -m pytest tests/gpu tests/integration tests/test_backends.py tests/test_history.py tests/test_stats.py tests/test_serialization.py tests/test_audit_regressions.py tests/remstimate/test_remstimate2_glm_validation.py -ra
python -m coverage report
```

The complete CI suite is defined in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). The cross-language R comparison is currently a manual validation, not a CI job.
