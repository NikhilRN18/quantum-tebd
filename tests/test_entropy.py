import numpy as np
from mps.mps_state import MPS
from mps.tebd import apply_two_site_gate


def test_bell_state_entropy_is_ln2():
    # Start with |00>
    m = MPS.product_state("00")

    # Build Bell state using (H ⊗ I) then CNOT
    H = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    HI = np.kron(H, I2)

    CNOT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )

    _, ent1 = apply_two_site_gate(m, i=0, U=HI, chi_max=10)
    _, ent2 = apply_two_site_gate(m, i=0, U=CNOT, chi_max=10)

    # After CNOT we should have a Bell state.
    # Entropy should be ln(2).
    assert np.isclose(ent2, np.log(2), atol=1e-6)