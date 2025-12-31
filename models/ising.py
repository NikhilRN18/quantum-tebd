import numpy as np
from scipy.linalg import expm


def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    

def ising_two_site_gate(J, dt):
    Z = pauli_z()
    ZZ = np.kron(Z, Z)
    h_zz = -J * ZZ
    U = expm(-1j * dt * h_zz)
    return U


def ising_one_site_gate(h, dt):
    X = pauli_x()
    h_x = -h * X
    U = expm(-1j * dt * h_x)
    return U