"""One process per selector; complete API fit and OS peak working set."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
import platform
import sys
import time

if sys.platform != "win32":
    raise SystemExit(
        "This measurement script uses Windows process high-water marks; run on Windows."
    )
import winreg
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from remflow import remify, remstats, remstimate  # noqa: E402 — measure the local checkout

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
selector = sys.argv[1]
proc = psutil.Process()
before = proc.memory_info().rss
start = time.perf_counter()
rng = np.random.default_rng(20261101)
n, m = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 2 else (20, 200)
send = rng.integers(1, n + 1, m)
recv = rng.integers(1, n, m)
recv += recv >= send
events = pd.DataFrame(dict(time=np.cumsum(rng.exponential(1, size=m)), sender=send, receiver=recv))
h = remify(events, actors=list(range(1, n + 1)), riskset="full", ordinal=False)
s = remstats(h, tie_effects="~ inertia() + reciprocity()", first=1)
prepared = time.perf_counter()
f = remstimate(h, s, backend=selector)
end = time.perf_counter()
mem = proc.memory_info()
cpu = winreg.QueryValueEx(
    winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"),
    "ProcessorNameString",
)[0]
result = dict(
    selector=selector,
    actors=n,
    events=m,
    effects=len(s.names),
    tensor_mib=sum(x.nbytes for x in s.stats) / 2**20,
    preparation_seconds=prepared - start,
    fit_seconds=end - prepared,
    total_seconds=end - start,
    baseline_rss_mib=before / 2**20,
    peak_working_set_mib=mem.peak_wset / 2**20,
    peak_private_mib=mem.peak_pagefile / 2**20,
    converged=bool(f.converged),
    coef=f.coef.tolist(),
    loglik=f.log_likelihood,
    platform=platform.platform(),
    python=platform.python_version(),
    numpy=np.__version__,
    scipy=scipy.__version__,
    cpu=cpu.strip(),
    ram_gib=psutil.virtual_memory().total / 2**30,
    metadata=f.metadata,
)
if selector.startswith("jax"):
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(np.stack(s.stats))
    obs = jnp.asarray(s.observed_indices)
    dt = jnp.asarray(np.diff(np.r_[0, events.time.to_numpy()]))
    beta = jnp.asarray(f.coef)

    def objective(b):
        eta = jnp.einsum("mrp,p->mr", x, b)
        return jnp.sum(eta[jnp.arange(m), obs]) - jnp.sum(dt * jnp.sum(jnp.exp(eta), axis=1))

    t = time.perf_counter()
    lowered = jax.jit(jax.value_and_grad(objective)).lower(beta)
    trace_end = time.perf_counter()
    compiled = lowered.compile()
    compile_end = time.perf_counter()
    ll, grad = compiled(beta)
    ll.block_until_ready()
    grad.block_until_ready()
    eval_end = time.perf_counter()
    result["separate_kernel"] = dict(
        trace_seconds=trace_end - t,
        compile_seconds=compile_end - trace_end,
        first_execution_seconds=eval_end - compile_end,
        loglik=float(ll),
        loglik_abs_difference=abs(float(ll) - f.log_likelihood),
        jax=jax.__version__,
        note=(
            "Separate exact-time kernel, not used by remstimate; "
            "excluded from full-fit timing and memory."
        ),
    )
(OUT / ("memory_" + selector.replace(":", "_") + f"_{n}_{m}.json")).write_text(
    json.dumps(result, indent=2, default=str)
)
print(json.dumps(result, indent=2, default=str))
