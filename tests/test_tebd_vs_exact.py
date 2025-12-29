import numpy as np

from mps.mps_state import MPS
from mps.tebd import tebd_step_ising
from mps.observables import magnetization_z

from exact.exact_diag import (
    tfim_hamiltonian,
    evolve_state_exact,
    basis_state,
    magnetization_z_dense,
)


def test_tebd_matches_exact_small_system():
    # Keep it small so test runs fast
    n = 6
    J = 1.0
    h = 0.7
    dt = 0.02
    steps = 20
    chi = 64
    state = "0" * n

    # TEBD
    mps = MPS.product_state(state)
    tebd_vals = []
    for _ in range(steps + 1):
        tebd_vals.append(magnetization_z(mps))
        tebd_step_ising(mps, J=J, h=h, dt=dt, chi_max=chi)

    tebd_vals = np.array(tebd_vals)

    # Exact
    psi0 = basis_state(state)
    H = tfim_hamiltonian(n, J=J, h=h)
    psis = evolve_state_exact(psi0, H, dt=dt, steps=steps)
    exact_vals = np.array([magnetization_z_dense(psi, n) for psi in psis])

    # Compare (allow small numerical error + trotter error)
    max_err = np.max(np.abs(tebd_vals - exact_vals))
    assert max_err < 5e-3