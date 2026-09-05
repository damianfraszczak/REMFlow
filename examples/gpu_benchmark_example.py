"""Run a GPU-oriented REMFlow likelihood benchmark.

This example benchmarks the REM likelihood/gradient kernel. Install its
benchmark-only dependency with ``uv sync --extra benchmark-cuda``. It uses
CuPy CUDA when available and reports an explanatory error when CuPy or a
compatible CUDA GPU is unavailable.
"""

from __future__ import annotations

from argparse import Namespace

from benchmarks.benchmark_likelihood_backend import run


def main() -> None:
    args = Namespace(
        backend="cupy:gpu",
        actors=50,
        events=1000,
        effects=4,
        seed=42,
        repeats=5,
        float64=False,
        output=None,
    )
    result = run(args)
    print(result)


if __name__ == "__main__":
    main()
