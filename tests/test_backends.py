from remflow import BackendUnavailable, available_backends, resolve_backend


def test_numpy_backend_is_available():
    backend = resolve_backend("numpy")

    assert backend.name == "numpy"
    assert backend.device == "cpu"


def test_backend_inventory_has_numpy():
    info = available_backends()

    assert info["numpy"]["available"] is True


def test_requested_jax_gpu_never_falls_back_to_cpu():
    try:
        backend = resolve_backend("jax:gpu")
    except BackendUnavailable:
        return

    assert backend.device == "gpu"
