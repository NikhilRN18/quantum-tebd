import numpy as np
from scipy.linalg import expm


def pauli_x():
    return np.array([[0, 1],
                     [1, 0]], dtype=np.complex128)


def pauli_z():
    return np.array([[1, 0],
                     [0, -1]], dtype=np.complex128)


def kron_all(ops):
    out = np.array([[1.0]], dtype=np.complex128)
    for op in ops:
        out = np.kron(out, op)
    return out


def op_on_site(op, site, n):
    """
    Returns the full 2^n x 2^n operator that applies 'op' on 'site' and identity elsewhere.
    site is 0-indexed.
    """
    I = np.eye(2, dtype=np.complex128)
    ops = [I] * n
    ops[site] = op
    return kron_all(ops)


def op_on_two_sites(op2, site, n):
    """
    Returns full operator for a 2-qubit operator op2 acting on (site, site+1).
    op2 must be 4x4 in the basis |00>,|01>,|10>,|11>.
    """
    if site < 0 or site >= n - 1:
        raise ValueError("site must satisfy 0 <= site < n-1")

    I = np.eye(2, dtype=np.complex128)

    # Build using kron blocks:
    # (I ⊗ ... ⊗ op2 ⊗ ... ⊗ I)
    left_ops = [I] * site
    right_ops = [I] * (n - site - 2)

    out = kron_all(left_ops) if left_ops else np.array([[1.0]], dtype=np.complex128)
    out = np.kron(out, op2)
    out = np.kron(out, kron_all(right_ops)) if right_ops else out

    return out


def tfim_hamiltonian(n, J, h):
    """
    Build the transverse-field Ising Hamiltonian (open boundary):
        H = -J sum_{i=0}^{n-2} Z_i Z_{i+1} - h sum_{i=0}^{n-1} X_i
    Returns a dense 2^n x 2^n matrix.
    """
    X = pauli_x()
    Z = pauli_z()
    I2 = np.eye(2, dtype=np.complex128)

    # Two-site operator ZZ
    ZZ = np.kron(Z, Z)

    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Interaction terms
    for i in range(n - 1):
        H += -J * op_on_two_sites(ZZ, i, n)

    # Field terms
    for i in range(n):
        H += -h * op_on_site(X, i, n)

    return H


def evolve_state_exact(psi0, H, dt, steps):
    """
    Evolve psi using exact matrix exponential:
        psi(t+dt) = exp(-iH dt) psi(t)
    Returns list of psi at each step including initial (length steps+1).
    """
    U = expm(-1j * dt * H)
    psis = [psi0]
    psi = psi0.copy()

    for _ in range(steps):
        psi = U @ psi
        psis.append(psi)

    return psis


def basis_state(bitstring):
    """
    Returns a dense computational basis vector |bitstring> of length 2^n.
    bitstring: string like "0101"
    """
    if isinstance(bitstring, str):
        bits = [int(c) for c in bitstring]
    else:
        bits = list(bitstring)

    n = len(bits)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b

    psi = np.zeros(2**n, dtype=np.complex128)
    psi[idx] = 1.0
    return psi


def magnetization_z_dense(psi, n):
    """
    Average <Z> across sites for dense state psi.
    """
    Z = pauli_z()
    total = 0.0 + 0.0j
    for i in range(n):
        Zi = op_on_site(Z, i, n)
        total += psi.conj().T @ (Zi @ psi)
    return (total / n).real