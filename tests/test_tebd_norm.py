import numpy as np
from mps.mps_state import MPS
from mps.tebd import tebd_step_ising


def test_tebd_preserves_norm_small_system():
    m = MPS.random_product_state(6, seed=0)
    n0 = m.norm()

    # Evolve for a few steps
    for _ in range(10):
        tebd_step_ising(m, J=1.0, h=0.7, dt=0.05, chi_max=32)

    n1 = m.norm()

    # Should remain ~1 (some numerical drift allowed)
    assert np.isclose(n0, n1, atol=1e-8)
    assert np.isclose(n1, 1.0, atol=1e-8)