FROM python:3.12-slim

ARG JAX_VERSION=0.7.2
ARG PIP_TRUSTED_HOST

ENV JAX_ENABLE_X64=true \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --root-user-action=ignore \
    "jax[cuda12]==${JAX_VERSION}" \
    "numpy>=1.26,<2.4" \
    "pandas>=2.1" \
    "scipy>=1.11" \
    "pytest>=8"

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src \
    REMFLOW_REQUIRE_GPU=1

CMD ["python", "-m", "pytest", "-q", "tests/gpu"]
