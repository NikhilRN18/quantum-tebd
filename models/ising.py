import numpy as np
from scipy.linalg import expm


def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    

def ising_two_site_gate(J, dt):
    """
    Two-site gate for the ZZ interaction:
        h_zz = -J * (Z ⊗ Z)
        U_zz = exp(-i dt h_zz)
    Returns 4x4 complex matrix.
    """
    Z = pauli_z()
    ZZ = np.kron(Z, Z)
    h_zz = -J * ZZ
    U = expm(-1j * dt * h_zz)
    return U


def ising_one_site_gate(h, dt):
    """
    One-site gate for transverse field term:
        h_x = -h * X
        U_x = exp(-i dt h_x) = exp(+i dt h X)
    Returns 2x2 complex matrix.
    """
    X = pauli_x()
    h_x = -h * X
    U = expm(-1j * dt * h_x)
    return U