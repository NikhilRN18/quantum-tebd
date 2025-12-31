import numpy as np
from mps.truncation import svd_truncate
from models.ising import ising_one_site_gate, ising_two_site_gate
from mps.entropy import entanglement_entropy_from_singular_values


def apply_two_site_gate(mps, i, U, chi_max=None):
    """
    Apply a two-site gate U to sites i and i+1 of an MPS.
    """
    if i < 0 or i >= mps.n - 1:
        raise ValueError("i must satisfy 0 <= i < n-1")

    A = mps.tensors[i]
    B = mps.tensors[i + 1]

    bl, d1, bm = A.shape
    bm2, d2, br = B.shape

    if bm != bm2:
        raise ValueError("Bond mismatch between tensors at i and i+1")
    if d1 != 2 or d2 != 2:
        raise ValueError("This implementation assumes qubits (physical dim 2)")

    # reshape U to rank-4 form: (s1', s2', s1, s2)
    if U.shape == (4, 4):
        U4 = U.reshape(2, 2, 2, 2)
    elif U.shape == (2, 2, 2, 2):
        U4 = U
    else:
        raise ValueError("U must have shape (4,4) or (2,2,2,2)")
    theta = np.tensordot(A, B, axes=(2, 0))

    theta = np.tensordot(U4, theta, axes=([2, 3], [1, 2]))
    theta = np.transpose(theta, (2, 0, 1, 3))


    theta_mat = theta.reshape(bl * 2, 2 * br)

    Umat, S, Vh, trunc_error = svd_truncate(theta_mat, chi_max)

    chi_new = len(S)

    # left tensor A': reshape Umat into (bl, 2, chi_new)
    A_new = Umat.reshape(bl, 2, chi_new)

    # right tensor B': (chi_new, 2, br)
    SVh = (S[:, None] * Vh)
    B_new = SVh.reshape(chi_new, 2, br)

    mps.tensors[i] = A_new
    mps.tensors[i + 1] = B_new

    ent = entanglement_entropy_from_singular_values(S)
    return trunc_error, ent

def tebd_step_ising(mps, J, h, dt, chi_max=None):
    """
    Perform one Strang-split TEBD timestep for the transverse-field Ising model.
    """
    total_trunc = 0.0
    bond_entropy = [0.0] * (mps.n - 1)
    # 1) half-step on-site X gates
    Ux_half = ising_one_site_gate(h=h, dt=dt / 2.0)
    for i in range(mps.n):
        mps.apply_one_site_gate(i, Ux_half)

    # 2) full-step ZZ gates on even bonds
    Uzz = ising_two_site_gate(J=J, dt=dt)
    for i in range(0, mps.n - 1, 2):
        trunc, ent = apply_two_site_gate(mps, i, Uzz, chi_max=chi_max)
        total_trunc += trunc
        bond_entropy[i] = ent   

    # 3) full-step ZZ gates on odd bonds
    for i in range(1, mps.n - 1, 2):
        trunc, ent = apply_two_site_gate(mps, i, Uzz, chi_max=chi_max)
        total_trunc += trunc
        bond_entropy[i] = ent
    # 4) half-step on-site X gates again
    for i in range(mps.n):
        mps.apply_one_site_gate(i, Ux_half)

    return total_trunc, bond_entropy