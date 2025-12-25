import numpy as np
from mps.mps_state import MPS


def test_product_state_dense():
    m = MPS.product_state("010")
    psi = m.to_dense()

    expected = np.zeros(8, dtype=np.complex128)
    expected[2] = 1.0

    assert np.allclose(psi, expected)


def test_product_state_norm():
    m = MPS.product_state("00000")
    assert np.isclose(m.norm(), 1.0)


def test_random_product_state_norm():
    m = MPS.random_product_state(10, seed=123)
    assert np.isclose(m.norm(), 1.0, atol=1e-12)


def test_bond_dims_for_product_state():
    m = MPS.product_state("0011")
    # For product states all bonds are dimension 1
    assert m.bond_dims() == [1, 1, 1, 1, 1]