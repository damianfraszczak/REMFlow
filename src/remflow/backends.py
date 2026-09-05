"""Array backend selection for REMFlow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Protocol

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp


class BackendUnavailable(RuntimeError):
    """Raised when a requested compute backend cannot be used."""


class ArrayBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def float_dtype(self) -> Any: ...

    def asarray(self, value: Any, dtype: Any | None = None) -> Any: ...
    def to_numpy(self, value: Any) -> np.ndarray: ...
    def logsumexp(self, value: Any, axis: int | None = None) -> Any: ...
    def value_and_grad(self, fn: Any) -> Any: ...
    def hessian(self, fn: Any) -> Any: ...
    def jit(self, fn: Any) -> Any: ...
    def scan(self, fn: Any, init: Any, xs: Any) -> Any: ...
    def vmap(self, fn: Any, *args: Any, **kwargs: Any) -> Any: ...

    @property
    def runtime_metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class NumpyBackend:
    name: str = "numpy"
    device: str = "cpu"
    float_dtype: Any = np.float64

    def asarray(self, value: Any, dtype: Any | None = None) -> np.ndarray:
        return np.asarray(value, dtype=dtype or self.float_dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def logsumexp(self, value: Any, axis: int | None = None) -> Any:
        return scipy_logsumexp(value, axis=axis)

    def value_and_grad(self, fn: Any) -> Any:
        raise BackendUnavailable("numpy backend does not provide automatic differentiation")

    def hessian(self, fn: Any) -> Any:
        raise BackendUnavailable("numpy backend does not provide automatic differentiation")

    def jit(self, fn: Any) -> Any:
        return fn

    def scan(
        self,
        fn: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Sequence[Any],
    ) -> tuple[Any, np.ndarray]:
        carry = init
        outputs: list[Any] = []
        for value in xs:
            carry, output = fn(carry, value)
            outputs.append(output)
        return carry, np.asarray(outputs)

    def vmap(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            raise BackendUnavailable("numpy vmap does not support mapping configuration")

        def mapped(values: Sequence[Any]) -> np.ndarray:
            return np.asarray([fn(value) for value in values])

        return mapped

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "device": self.device,
            "precision": "float64",
            "numpy_version": np.__version__,
        }


class JaxBackend:
    """JAX backend with explicit CPU/GPU device selection."""

    name = "jax"
    float_dtype: Any

    def __init__(self, device: str = "auto") -> None:
        try:
            import jax
            import jax.numpy as jnp
            from jax import scipy as jsp
        except Exception as exc:  # pragma: no cover - depends on optional install
            raise BackendUnavailable(
                "JAX backend requested but jax is not installed. Install remflow[gpu]."
            ) from exc

        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call,unused-ignore]
        if device == "gpu":
            candidates = _jax_platform_devices(jax, "gpu")
            if not candidates:
                raise BackendUnavailable("JAX GPU backend requested but no GPU device is available")
            selected = candidates[0]
        elif device == "cpu":
            candidates = _jax_platform_devices(jax, "cpu")
            if not candidates:
                raise BackendUnavailable("JAX CPU backend requested but no CPU device is available")
            selected = candidates[0]
        else:
            selected = jax.devices()[0]

        self._jax = jax
        self._jnp = jnp
        self._jsp = jsp
        self._device = selected
        self.device = selected.platform
        self.float_dtype = jnp.float64

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        enable_x64 = bool(self._jax.config.jax_enable_x64)
        disable_jit = bool(self._jax.config.jax_disable_jit)
        return {
            "backend": self.name,
            "device": self.device,
            "device_id": int(self._device.id),
            "device_kind": str(self._device.device_kind),
            "precision": "float64",
            "jax_version": str(self._jax.__version__),
            "jaxlib_version": version("jaxlib"),
            "jax_enable_x64": enable_x64,
            "jax_disable_jit": disable_jit,
        }

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        return self._jax.device_put(
            self._jnp.asarray(value, dtype=dtype or self.float_dtype), self._device
        )

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def logsumexp(self, value: Any, axis: int | None = None) -> Any:
        return self._jsp.special.logsumexp(value, axis=axis)

    def value_and_grad(self, fn: Any) -> Any:
        return self._jax.value_and_grad(fn)

    def hessian(self, fn: Any) -> Any:
        return self._jax.hessian(fn)

    def jit(self, fn: Any) -> Any:
        return self._jax.jit(fn)

    def scan(self, fn: Any, init: Any, xs: Any) -> Any:
        return self._jax.lax.scan(fn, init, xs)

    def vmap(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return self._jax.vmap(fn, *args, **kwargs)


def resolve_backend(backend: str | ArrayBackend = "numpy") -> ArrayBackend:
    """Resolve a backend specifier.

    Supported strings are `numpy`, `jax`, `jax:cpu`, and `jax:gpu`. A requested
    GPU never falls back to CPU silently.
    """

    if not isinstance(backend, str):
        return backend
    if backend == "numpy" or backend == "auto":
        return NumpyBackend()
    if backend == "jax":
        return JaxBackend("auto")
    if backend == "jax:cpu":
        return JaxBackend("cpu")
    if backend == "jax:gpu":
        return JaxBackend("gpu")
    raise ValueError("backend must be one of: numpy, auto, jax, jax:cpu, jax:gpu")


def available_backends() -> dict[str, Any]:
    """Return backend availability metadata without requiring GPU support."""

    info: dict[str, Any] = {"numpy": {"available": True, "devices": ["cpu"]}}
    try:
        import jax

        devices = [
            device
            for platform in ("cpu", "gpu", "tpu")
            for device in _jax_platform_devices(jax, platform)
        ]
        info["jax"] = {
            "available": True,
            "devices": [device.platform for device in devices],
            "device_kinds": [str(device.device_kind) for device in devices],
        }
    except Exception as exc:  # pragma: no cover - depends on optional install
        info["jax"] = {"available": False, "reason": str(exc)}
    return info


def _jax_platform_devices(jax: Any, platform: str) -> list[Any]:
    """Return devices for one JAX platform without changing the default backend."""

    try:
        return list(jax.devices(platform))
    except RuntimeError:
        return []
