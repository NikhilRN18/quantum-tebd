import numpy as np
from models.ising import pauli_z


def magnetization_z(mps):
    n = mps.n
    psi = mps.to_dense()
    Z = pauli_z()

    total = 0.0 + 0.0j

    # Build Z_i operator by kron products (slow, but fine for first version)
    I2 = np.eye(2, dtype=np.complex128)

    for i in range(n):
        ops = []
        for j in range(n):
            ops.append(Z if j == i else I2)

        Zi = ops[0]
        for op in ops[1:]:
            Zi = np.kron(Zi, op)

        total += (psi.conj().T @ (Zi @ psi))

    return (total / n).real