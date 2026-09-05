# Performance evaluation

REM workloads grow with the number of observed events and the number of
alternatives in each event's risk set. The benchmarks therefore record actor
count, event count, risk-set mode, statistic count, and numerical backend. A
timing without that context is not useful for comparing runs.

## Reference workloads

| Scenario | Actors | Events | Alternatives per event | Dense float64 tensor |
|---|---:|---:|---:|---:|
| Medium | 50 | 1,000 | 2,450 | 74.77 MiB |
| Large | 100 | 1,000 | 9,900 | 302.12 MiB |

## What a result records

Committed benchmark results include:

- CPU and GPU models, core count, and accelerator memory;
- Python, NumPy, SciPy, JAX, and CUDA versions;
- the exact backend selector;
- actor, event, statistic, and risk-set sizes;
- whether JAX compilation is included in the timing;
- peak accelerator memory when it can be measured;
- likelihood and gradient differences from NumPy.

The likelihood benchmark times only the dense numerical kernel. It measures
JAX compilation separately, then times 10 evaluations. Event normalization,
statistic construction, and host-to-device transfer are outside this region.
Use `benchmark_backend.py` when those end-to-end costs matter.

## Reading the numbers

A GPU helps when likelihood and gradient evaluation dominate the workload and
the tensors are large enough to offset compilation and data transfer. NumPy can
be faster for small inputs.

Parallel risk-set construction is a separate measurement. The
`remify(..., ncores>1)` benchmark compares its result with `ncores=1` and checks
that every risk-set frame has the same rows and order. Thread-pool startup may
make the parallel version slower on small histories.

## Running the benchmarks

End-to-end backend runs:

```bash
python benchmarks/benchmark_backend.py --backend numpy --actors 50 --events 2000
python benchmarks/benchmark_backend.py --backend jax:cpu --actors 50 --events 2000
```

Direct likelihood and gradient measurements:

```bash
python benchmarks/benchmark_likelihood_backend.py \
  --float64 --repeats 10 --backend numpy --actors 50 --events 1000
python benchmarks/benchmark_likelihood_backend.py \
  --float64 --repeats 10 --backend jax:gpu --actors 50 --events 1000
python benchmarks/run_gpu_study.py --repeats 10 --include-large
```

`cupy:gpu` is available only as a comparison in the benchmark script. It is not
a REMFlow model backend and is omitted from the default study. To include it:

```bash
uv sync --extra gpu --extra benchmark-cuda
uv run python benchmarks/benchmark_likelihood_backend.py \
  --float64 --repeats 10 --backend cupy:gpu --actors 50 --events 1000
```

The measured RTX 4070 float64 results are reported in the project README. Set
`REMFLOW_REQUIRE_GPU=1` to run the physical-GPU tests; otherwise they skip when
no accelerator is present. A reproducible validation image is defined in
`tools/gpu/Dockerfile`.

`benchmarks/run_gpu_study.py` writes machine-specific results to the ignored
`benchmark-results/gpu/` directory.
