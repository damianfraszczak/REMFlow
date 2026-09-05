"""Run a CPU/JAX/GPU benchmark matrix and write a Markdown report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCENARIOS = [
    {"name": "medium", "actors": 50, "events": 1000, "effects": 4},
    {"name": "large", "actors": 100, "events": 1000, "effects": 4},
]


def run_one(output_dir: Path, scenario: dict[str, int | str], backend: str, repeats: int) -> dict:
    output = output_dir / f"{scenario['name']}_{backend.replace(':', '_')}.json"
    command = [
        sys.executable,
        "benchmarks/benchmark_likelihood_backend.py",
        "--backend",
        backend,
        "--actors",
        str(scenario["actors"]),
        "--events",
        str(scenario["events"]),
        "--effects",
        str(scenario["effects"]),
        "--repeats",
        str(repeats),
        "--float64",
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("CUPY_CACHE_DIR", str(Path(".cache/cupy").resolve()))
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    return {
        "backend": backend,
        "scenario": scenario["name"],
        "error": completed.stderr.strip()
        or completed.stdout.strip()
        or f"exit {completed.returncode}",
    }


def write_report(output_dir: Path, results: list[dict]) -> None:
    lines = [
        "# REMFlow CPU/JAX/GPU Kernel Benchmark Results",
        "",
        "The table records the batched REM log-likelihood and gradient kernel.",
        "Pandas-based preprocessing and JAX compilation are outside the timed region.",
        "",
        "| Scenario | Backend | Device | Actors | Events | Risk set | Effects | "
        "Tensor MiB | GPU peak MiB | Warmup s | Mean s | Min s | Max s | LL abs err | "
        "Grad max err | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        if "error" in result:
            scenario = result.get("scenario", "")
            backend = result.get("backend", "")
            lines.append(
                f"| {scenario} | {backend} | n/a | n/a | n/a | n/a | n/a | "
                f"n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {result['error']} |"
            )
            continue
        warmup = result["warmup_seconds"]
        lines.append(
            "| {scenario} | {backend} | {device} | {actors} | {events} | {riskset_size} | "
            "{effects} | {tensor_mib:.1f} | {peak_memory} | {warmup} | {mean_seconds:.4f} | "
            "{min_seconds:.4f} | {max_seconds:.4f} | {log_likelihood_abs_error:.3e} | "
            "{gradient_max_abs_error:.3e} | {status} |".format(
                scenario=_scenario_name(result),
                peak_memory=(
                    "n/a"
                    if result.get("device_peak_memory_mib") is None
                    else f"{result['device_peak_memory_mib']:.1f}"
                ),
                warmup="n/a" if warmup is None else f"{warmup:.4f}",
                status="parity pass" if result["parity_passed"] else "PARITY FAIL",
                **result,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `warmup_seconds` includes JAX compilation for JAX backends.",
            "- `mean_seconds`, `min_seconds`, and `max_seconds` exclude the explicit warmup call.",
            "- GPU results are valid only when `Device` is `gpu`; a CPU-only "
            "JAX run is not a GPU benchmark.",
            "- Every timed result uses float64 and is compared against the NumPy reference; "
            "a failed accuracy check is reported as `PARITY FAIL`.",
            "- `cupy:gpu` is an optional feasibility-only comparator. It is not a "
            "supported REMFlow model backend and is excluded from the default study.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scenario_name(result: dict) -> str:
    for scenario in SCENARIOS:
        if scenario["actors"] == result["actors"] and scenario["events"] == result["events"]:
            return str(scenario["name"])
    return "custom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="benchmark-results/gpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backends", nargs="+", default=["numpy", "jax:cpu", "jax:gpu"]
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include the 100-actor scenario (about 302 MiB input tensor).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS if args.include_large else SCENARIOS[:1]
    results = []
    for scenario in scenarios:
        for backend in args.backends:
            results.append(run_one(output_dir, scenario, backend, args.repeats))
    write_report(output_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
