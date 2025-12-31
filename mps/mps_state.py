import numpy as np


class MPS:
    """
    A simple Matrix Product State (MPS) for a 1D chain.

    We store tensors A[i] with shape:
        (bond_left, physical_dim, bond_right)

    For spin-1/2 systems, physical_dim = 2.
    """

    def __init__(self, tensors):
        self.tensors = tensors 

    @property
    def n(self):
        return len(self.tensors)

    def bond_dims(self):
        dims = [self.tensors[0].shape[0]]
        for A in self.tensors:
            dims.append(A.shape[2])
        return dims

    def check_shapes(self):
        if self.n == 0:
            raise ValueError("MPS must have at least 1 site.")

        for i, A in enumerate(self.tensors):
            if A.ndim != 3:
                raise ValueError(f"Tensor at site {i} is not rank-3.")
            bl, d, br = A.shape
            if d != 2:
                raise ValueError(f"Tensor at site {i} has physical_dim={d}, expected 2.")

        # bond matching
        for i in range(self.n - 1):
            right_dim = self.tensors[i].shape[2]
            left_dim_next = self.tensors[i + 1].shape[0]
            if right_dim != left_dim_next:
                raise ValueError(
                    f"Bond mismatch between sites {i} and {i+1}: "
                    f"{right_dim} != {left_dim_next}"
                )

        if self.tensors[0].shape[0] != 1:
            raise ValueError("Left boundary bond dimension must be 1.")
        if self.tensors[-1].shape[2] != 1:
            raise ValueError("Right boundary bond dimension must be 1.")

    @staticmethod
    def product_state(bitstring):
        """
        Create an MPS for a computational basis product state.
        """
        if isinstance(bitstring, str):
            bitstring = [int(ch) for ch in bitstring]

        tensors = []
        for b in bitstring:
            A = np.zeros((1, 2, 1), dtype=np.complex128)
            A[0, b, 0] = 1.0
            tensors.append(A)
        return MPS(tensors)

    @staticmethod
    def random_product_state(n, seed=None):
        """
        Create a random product state MPS.
        Each site is a random normalized 2-vector.
        """
        rng = np.random.default_rng(seed)
        tensors = []
        for _ in range(n):
            v = rng.normal(size=2) + 1j * rng.normal(size=2)
            v = v / np.linalg.norm(v)
            A = np.zeros((1, 2, 1), dtype=np.complex128)
            A[0, 0, 0] = v[0]
            A[0, 1, 0] = v[1]
            tensors.append(A)
        return MPS(tensors)

    def to_dense(self):
        self.check_shapes()

        A0 = self.tensors[0]
        psi = A0[0, :, :]  # shape (2, b1)

        for i in range(1, self.n):
            A = self.tensors[i]  # (bi,2,bi+1)

            psi = np.tensordot(psi, A, axes=(1, 0))  # (2^i, 2, bi+1)
            psi = psi.reshape(-1, A.shape[2])  # (2^(i+1), bi+1)

        psi = psi[:, 0]
        return psi

    def norm(self):
        psi = self.to_dense()
        return np.linalg.norm(psi)
    
    def apply_one_site_gate(self, i, G):
        A = self.tensors[i] 
        if G.shape != (2, 2):
            raise ValueError("One-site gate must be 2x2")

        A_new = np.tensordot(G, A, axes=(1, 1))  
        A_new = np.transpose(A_new, (1, 0, 2))   
        self.tensors[i] = A_new