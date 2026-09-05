"""Benchmark REM likelihood and gradient kernels on NumPy/JAX backends.

This benchmark isolates the compute-heavy part of relational event estimation:
for each observed event, compare the observed dyad against all alternatives in
the risk set using a stable multinomial log-likelihood. This is the part that
is expected to benefit from JAX/GPU on large dense risk-set tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

_cupy_cache = Path(".cache/cupy").resolve()
_cupy_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUPY_CACHE_DIR", str(_cupy_cache))


@dataclass(frozen=True)
class KernelBenchmarkResult:
    backend: str
    device: str
    device_kind: str
    dtype: str
    actors: int
    events: int
    riskset_size: int
    effects: int
    repeats: int
    warmup_seconds: float | None
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    log_likelihood: float
    gradient_norm: float
    reference_log_likelihood: float
    log_likelihood_abs_error: float
    gradient_max_abs_error: float
    parity_passed: bool
    tensor_mib: float
    device_bytes_limit_mib: float | None
    device_memory_after_transfer_mib: float | None
    device_memory_after_warmup_mib: float | None
    device_peak_memory_mib: float | None
    platform: dict[str, Any]
    versions: dict[str, Any]


def directed_riskset_size(actor_count: int) -> int:
    return actor_count * (actor_count - 1)


def make_problem(
    *,
    actor_count: int,
    event_count: int,
    effect_count: int,
    seed: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    riskset_size = directed_riskset_size(actor_count)

    # The first two columns mimic count-like REM statistics; remaining columns
    # are continuous covariates. This keeps the benchmark synthetic but close to
    # typical REM design tensors.
    x = rng.normal(loc=0.0, scale=0.5, size=(event_count, riskset_size, effect_count)).astype(dtype)
    if effect_count >= 1:
        x[:, :, 0] = rng.poisson(0.25, size=(event_count, riskset_size)).astype(dtype)
    if effect_count >= 2:
        x[:, :, 1] = rng.poisson(0.15, size=(event_count, riskset_size)).astype(dtype)

    observed = rng.integers(0, riskset_size, size=event_count, dtype=np.int64)
    beta = rng.normal(loc=0.0, scale=0.2, size=effect_count).astype(dtype)
    return x, observed, beta


def numpy_loglik_grad(
    x: np.ndarray, observed: np.ndarray, beta: np.ndarray
) -> tuple[float, np.ndarray]:
    eta = np.einsum("erf,f->er", x, beta, optimize=True)
    normalizer = logsumexp(eta, axis=1)
    event_ids = np.arange(x.shape[0])
    ll = np.sum(eta[event_ids, observed] - normalizer)
    probs = np.exp(eta - normalizer[:, None])
    expected = np.einsum("er,erf->ef", probs, x, optimize=True)
    grad = np.sum(x[event_ids, observed, :] - expected, axis=0)
    return float(ll), grad


def jax_runner(x_np: np.ndarray, observed_np: np.ndarray, beta_np: np.ndarray, backend: str):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", x_np.dtype == np.float64)
    requested = backend.split(":", 1)[1] if ":" in backend else "default"
    devices = jax.devices()
    if requested == "gpu":
        candidates = [device for device in devices if device.platform == "gpu"]
        if not candidates:
            raise RuntimeError("JAX GPU requested but no GPU device is visible to JAX")
        device = candidates[0]
    elif requested == "cpu":
        candidates = [device for device in devices if device.platform == "cpu"]
        if not candidates:
            raise RuntimeError("JAX CPU requested but no CPU device is visible to JAX")
        device = candidates[0]
    else:
        device = devices[0]

    x = jax.device_put(jnp.asarray(x_np), device)
    observed = jax.device_put(jnp.asarray(observed_np), device)
    beta = jax.device_put(jnp.asarray(beta_np), device)

    def loglik(beta_value):
        eta = jnp.einsum("erf,f->er", x, beta_value)
        event_ids = jnp.arange(x.shape[0])
        return jnp.sum(eta[event_ids, observed] - jax.scipy.special.logsumexp(eta, axis=1))

    value_and_grad = jax.jit(jax.value_and_grad(loglik))

    def run_once() -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(beta)
        value.block_until_ready()
        grad.block_until_ready()
        return float(value), np.asarray(grad)

    return device, run_once


def jax_memory_snapshot(device: Any) -> dict[str, float | None]:
    """Return JAX allocator counters in MiB when the platform exposes them."""

    stats = device.memory_stats()
    if not stats:
        return {
            "limit": None,
            "in_use": None,
            "peak_in_use": None,
        }

    def mib(key: str) -> float | None:
        value = stats.get(key)
        return None if value is None else float(value) / (1024**2)

    return {
        "limit": mib("bytes_limit"),
        "in_use": mib("bytes_in_use"),
        "peak_in_use": mib("peak_bytes_in_use"),
    }


def cupy_runner(x_np: np.ndarray, observed_np: np.ndarray, beta_np: np.ndarray):
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "The optional CuPy benchmark comparator is not installed. "
            "Run `uv sync --extra benchmark-cuda` and verify that an NVIDIA "
            "CUDA 12-compatible driver is available."
        ) from exc

    device_id = 0
    device = cp.cuda.Device(device_id)
    device.use()
    x = cp.asarray(x_np)
    observed = cp.asarray(observed_np)
    beta = cp.asarray(beta_np)

    def run_once() -> tuple[float, np.ndarray]:
        eta = cp.einsum("erf,f->er", x, beta)
        normalizer = cp.log(
            cp.sum(cp.exp(eta - cp.max(eta, axis=1, keepdims=True)), axis=1)
        ) + cp.max(eta, axis=1)
        event_ids = cp.arange(x.shape[0])
        ll = cp.sum(eta[event_ids, observed] - normalizer)
        probs = cp.exp(eta - normalizer[:, None])
        expected = cp.einsum("er,erf->ef", probs, x)
        grad = cp.sum(x[event_ids, observed, :] - expected, axis=0)
        cp.cuda.Stream.null.synchronize()
        return float(ll.get()), cp.asnumpy(grad)

    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return props["name"].decode(), run_once


def time_runner(runner, repeats: int) -> tuple[list[float], float, np.ndarray]:
    durations: list[float] = []
    last_ll = 0.0
    last_grad = np.array([], dtype=float)
    for _ in range(repeats):
        start = time.perf_counter()
        last_ll, last_grad = runner()
        durations.append(time.perf_counter() - start)
    return durations, last_ll, last_grad


def run(args: argparse.Namespace) -> KernelBenchmarkResult:
    dtype = np.float64 if args.float64 else np.float32
    x, observed, beta = make_problem(
        actor_count=args.actors,
        event_count=args.events,
        effect_count=args.effects,
        seed=args.seed,
        dtype=dtype,
    )
    reference_ll, reference_gradient = numpy_loglik_grad(x, observed, beta)

    warmup_seconds: float | None = None
    device_kind = "CPU"
    memory_after_transfer = {"limit": None, "in_use": None, "peak_in_use": None}
    memory_after_warmup = dict(memory_after_transfer)
    memory_after_timing = dict(memory_after_transfer)
    if args.backend == "numpy":
        device = "cpu"

        def runner():
            return numpy_loglik_grad(x, observed, beta)

    elif args.backend in {"jax", "jax:cpu", "jax:gpu"}:
        jax_device, runner = jax_runner(x, observed, beta, args.backend)
        device = jax_device.platform
        device_kind = str(jax_device.device_kind)
        memory_after_transfer = jax_memory_snapshot(jax_device)
        start = time.perf_counter()
        runner()
        warmup_seconds = time.perf_counter() - start
        memory_after_warmup = jax_memory_snapshot(jax_device)
    elif args.backend == "cupy:gpu":
        device, runner = cupy_runner(x, observed, beta)
        device_kind = device
        start = time.perf_counter()
        runner()
        warmup_seconds = time.perf_counter() - start
    else:
        raise ValueError("backend must be numpy, jax, jax:cpu, jax:gpu, or cupy:gpu")

    durations, ll, grad = time_runner(runner, args.repeats)
    if args.backend in {"jax", "jax:cpu", "jax:gpu"}:
        memory_after_timing = jax_memory_snapshot(jax_device)
    ll_error = abs(ll - reference_ll)
    gradient_error = float(np.max(np.abs(grad - reference_gradient)))
    parity_tolerance = 1e-8 if args.float64 else 5e-5
    versions = {"numpy": np.__version__}
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import jax
        import jaxlib

        versions["jax"] = jax.__version__
        versions["jaxlib"] = jaxlib.__version__
        versions["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:
        versions["jax"] = f"unavailable: {exc}"
    try:
        import cupy as cp

        versions["cupy"] = cp.__version__
        versions["cupy_cuda_runtime"] = str(cp.cuda.runtime.runtimeGetVersion())
        versions["cupy_device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception as exc:
        versions["cupy"] = f"unavailable: {exc}"

    return KernelBenchmarkResult(
        backend=args.backend,
        device=device,
        device_kind=device_kind,
        dtype=str(x.dtype),
        actors=args.actors,
        events=args.events,
        riskset_size=directed_riskset_size(args.actors),
        effects=args.effects,
        repeats=args.repeats,
        warmup_seconds=warmup_seconds,
        mean_seconds=float(np.mean(durations)),
        min_seconds=float(np.min(durations)),
        max_seconds=float(np.max(durations)),
        log_likelihood=float(ll),
        gradient_norm=float(np.linalg.norm(grad)),
        reference_log_likelihood=reference_ll,
        log_likelihood_abs_error=float(ll_error),
        gradient_max_abs_error=gradient_error,
        parity_passed=bool(
            ll_error <= parity_tolerance and gradient_error <= parity_tolerance
        ),
        tensor_mib=float(x.nbytes / (1024**2)),
        device_bytes_limit_mib=memory_after_timing["limit"],
        device_memory_after_transfer_mib=memory_after_transfer["in_use"],
        device_memory_after_warmup_mib=memory_after_warmup["in_use"],
        device_peak_memory_mib=memory_after_timing["peak_in_use"],
        platform={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        versions=versions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "jax", "jax:cpu", "jax:gpu", "cupy:gpu"],
    )
    parser.epilog = (
        "cupy:gpu is a feasibility-only comparison and is not a supported REMFlow backend; "
        "JAX is the product accelerator backend."
    )
    parser.add_argument("--actors", type=int, default=25)
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--effects", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--float64", action="store_true", help="Use float64 instead of float32")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        payload = {"backend": args.backend, "error": str(exc)}
        print(json.dumps(payload, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        return 2

    payload = asdict(result)
    print(json.dumps(payload, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
