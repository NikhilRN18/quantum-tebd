import numpy as np
from mps.mps_state import MPS
from mps.tebd import apply_two_site_gate


def kron(*ops):
    out = np.array([[1.0]], dtype=np.complex128)
    for op in ops:
        out = np.kron(out, op)
    return out


def test_identity_gate_does_not_change_state():
    # Start with |0101>
    m = MPS.product_state("0101")
    psi_before = m.to_dense()

    I4 = np.eye(4, dtype=np.complex128)
    trunc = apply_two_site_gate(m, i=1, U=I4, chi_max=10)

    psi_after = m.to_dense()

    assert trunc == 0.0
    assert np.allclose(psi_before, psi_after)


def test_cnot_gate_changes_state_correctly():
    # CNOT on qubits (1 -> 2) in a 3-qubit state
    # We'll apply it to sites i=1 and i+1=2 (0-indexed).
    #
    # Start with |010> (control=1, target=0) -> should become |011>
    m = MPS.product_state("010")
    psi_before = m.to_dense()

    # CNOT matrix (control is first qubit of the pair, target is second)
    CNOT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )

    trunc = apply_two_site_gate(m, i=1, U=CNOT, chi_max=10)
    psi_after = m.to_dense()

    # Exact dense calculation:
    I2 = np.eye(2, dtype=np.complex128)
    U_full = kron(I2, CNOT)  # CNOT acts on last two qubits
    expected = U_full @ psi_before

    assert trunc == 0.0
    assert np.allclose(psi_after, expected)


def test_truncation_limits_bond_dimension():
    # Apply a gate that creates entanglement on |00> -> Bell-ish
    # We'll create (|00> + |11>)/sqrt(2) using a simple gate:
    #
    # H on first qubit + CNOT => Bell state, but here we can just do it
    # in 2 steps with 2-site gates for test simplicity.
    #
    # We'll do: apply (H ⊗ I) as a 2-site "gate", then apply CNOT.
    m = MPS.product_state("00")

    H = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    # (H ⊗ I) as 4x4
    HI = np.kron(H, I2)

    # CNOT
    CNOT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )

    apply_two_site_gate(m, i=0, U=HI, chi_max=10)
    apply_two_site_gate(m, i=0, U=CNOT, chi_max=10)

    # Now the state should be entangled, so bond dim should be 2
    dims_before = m.bond_dims()
    assert dims_before == [1, 2, 1]

    # Now apply identity but force chi_max=1 (truncate hard)
    I4 = np.eye(4, dtype=np.complex128)
    trunc = apply_two_site_gate(m, i=0, U=I4, chi_max=1)

    dims_after = m.bond_dims()
    assert dims_after == [1, 1, 1]
    assert trunc > 0.0