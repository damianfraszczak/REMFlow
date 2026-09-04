# Classroom relational-event study

This example follows the classroom analysis in Chapter 13, Part 4 of J. A.
Smith's *Network Analysis: Integrating Social Network Theory, Method, and
Application with R*:

<https://inarwhal.github.io/NetworkAnalysisR-book/ch13-Relational-Event-Models-R.html>

The original R Markdown file is linked rather than copied. Provenance, source
URLs, checksums, and reuse terms for the retained data are recorded in
`data/classroom_events/SOURCES.md` and `data/classroom_events/LICENSE.md`.

## Analysis reproduced here

The example uses the same classroom interaction data and follows the relevant
preparation choices from the tutorial. It:

1. loads interaction histories for two dates;
2. removes broadcast interactions where `to_all_col` or `from_all_col` is set;
3. maps `time_estimate_col`, `send_col`, and `receive_col` to the event time,
   sender, and receiver;
4. derives intercept, gender, and teacher actor attributes;
5. builds seating and friendship dyadic covariates;
6. fits baseline, actor-covariate, dyadic-covariate, recency, and participation-
   shift specifications.

## What the comparison means

The tutorial fits an exact-time model with `relevent::rem.dyad()`. The REMFlow
example can fit exact-time and ordinal models to the same event histories and
covariate definitions. It reproduces the study design and workflow, but it does
not claim coefficient-by-coefficient equivalence with the R estimator.

## Run the example

```bash
PYTHONPATH=src python examples/classroom_event_study.py
PYTHONPATH=src python examples/classroom_event_study.py --timing exact
```

In Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
python examples/classroom_event_study.py
```

## Data files

The local copies are stored in `data/classroom_events/`:

- `class_interactions_date1.txt`
- `class_interactions_date2.txt`
- `class_attributes.txt`
- `class_seating_date1.txt`
- `class_seating_date2.txt`
- `class_edgelist_sem2.txt`

They were downloaded from the public `Integrated_Network_Science` repository
used by the tutorial. The files retain the upstream CC BY-NC-ND 4.0 license and
are not covered by REMFlow's MIT license.
