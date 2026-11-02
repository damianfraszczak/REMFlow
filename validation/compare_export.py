"""Export the identical event histories and design arrays for relevent MLE checks."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from examples.classroom_event_study import (  # noqa: E402 — local checkout paths above
    build_classroom_stats,
    build_history,
    load_classroom_data,
    study_model_terms,
)
from examples.exact_time_simulation_study import simulate_study_process  # noqa: E402
from remflow import remify, remstats, remstimate  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
results = []


def export(label, history, stats, n):
    fit = remstimate(history, stats, backend="numpy")
    m, p = len(stats.stats), len(stats.names)
    cov = np.zeros((m + 1, p, n, n))
    for i, x in enumerate(stats.stats):
        risk = history.risksets[i]
        for j, row in enumerate(risk.itertuples()):
            cov[i, :, int(row.sender_id) - 1, int(row.receiver_id) - 1] = x[j]
    cov.ravel(order="F").astype("<f8").tofile(OUT / (label + ".bin"))
    history.events[["time", "sender_id", "receiver_id"]].to_csv(
        OUT / (label + "_events.csv"), index=False
    )
    record = dict(
        label=label,
        n=n,
        m=m,
        p=p,
        names=fit.names,
        coef=fit.coef.tolist(),
        se=fit.se.tolist(),
        loglik=fit.log_likelihood,
        bic=fit.BIC,
        converged=bool(fit.converged),
    )
    results.append(record)
    (OUT / "python_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(label, fit.log_likelihood, fit.BIC, flush=True)


for day in ["date1", "date2"]:
    data = load_classroom_data(day)
    h = build_history(data, ordinal=False)
    for model in ["mod1", "mod2a", "mod3b", "mod4f"]:
        export(
            day + "_" + model, h, build_classroom_stats(h, data, terms=study_model_terms(model)), 18
        )
events = simulate_study_process()
h = remify(events, actors=list(range(1, 26)), riskset="full", ordinal=False)
export("simulation", h, remstats(h, tie_effects="~ psABBA() + recencySendSender()", first=1), 25)
pd.DataFrame([{k: r[k] for k in ["label", "n", "m", "p"]} for r in results]).to_csv(
    OUT / "models.csv", index=False
)
