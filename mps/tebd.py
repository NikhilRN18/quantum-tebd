import numpy as np
from mps.truncation import svd_truncate


def apply_two_site_gate(mps, i, U, chi_max=None):
    """
    Apply a two-site gate U to sites i and i+1 of an MPS.

    mps.tensors[i] has shape (bl, 2, bm)
    mps.tensors[i+1] has shape (bm, 2, br)
    U should be shape (4, 4) or (2,2,2,2).

    Returns:
        trunc_error (float)
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

    # 1) Build theta = contract A and B over bm
    # A: (bl, s1, bm)
    # B: (bm, s2, br)
    # theta: (bl, s1, s2, br)
    theta = np.tensordot(A, B, axes=(2, 0))

    # 2) Apply gate on physical indices (s1, s2)
    # U4: (s1', s2', s1, s2)
    # theta: (bl, s1, s2, br)
    # result: (bl, s1', s2', br)
    theta = np.tensordot(U4, theta, axes=([2, 3], [1, 2]))
    theta = np.transpose(theta, (2, 0, 1, 3))

    # 3) Reshape theta into matrix for SVD:
    # left = (bl * s1'), right = (s2' * br)
    theta_mat = theta.reshape(bl * 2, 2 * br)

    Umat, S, Vh, trunc_error = svd_truncate(theta_mat, chi_max)

    # 4) Put back into MPS tensors
    chi_new = len(S)

    # left tensor A': reshape Umat into (bl, 2, chi_new)
    A_new = Umat.reshape(bl, 2, chi_new)

    # right tensor B': (chi_new, 2, br)
    # We absorb S into Vh to keep things simple:
    SVh = (S[:, None] * Vh)
    B_new = SVh.reshape(chi_new, 2, br)

    mps.tensors[i] = A_new
    mps.tensors[i + 1] = B_new

    return trunc_error