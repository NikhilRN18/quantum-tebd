import argparse
import numpy as np
import matplotlib.pyplot as plt

from mps.mps_state import MPS
from mps.tebd import tebd_step_ising
from mps.observables import magnetization_z

from exact.exact_diag import (
    tfim_hamiltonian,
    evolve_state_exact,
    basis_state,
    magnetization_z_dense,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--state", type=str, default="00000000")
    args = parser.parse_args()

    if len(args.state) != args.n:
        raise ValueError("bitstring length must match n")

    # --- TEBD ---
    mps = MPS.product_state(args.state)
    tebd_times = []
    tebd_mags = []
    truncs = []

    t = 0.0
    for step in range(args.steps + 1):
        tebd_times.append(t)
        tebd_mags.append(magnetization_z(mps))

        if step < args.steps:
            trunc = tebd_step_ising(mps, J=args.J, h=args.h, dt=args.dt, chi_max=args.chi)
            truncs.append(trunc)
            t += args.dt

    # --- Exact ---
    psi0 = basis_state(args.state)
    H = tfim_hamiltonian(args.n, J=args.J, h=args.h)
    psis = evolve_state_exact(psi0, H, dt=args.dt, steps=args.steps)

    exact_mags = [magnetization_z_dense(psi, args.n) for psi in psis]

    # --- Compare ---
    tebd_mags = np.array(tebd_mags)
    exact_mags = np.array(exact_mags)
    err = np.abs(tebd_mags - exact_mags)

    print("Max |<Z>_TEBD - <Z>_exact| =", float(np.max(err)))
    print("Sum trunc errors =", float(np.sum(truncs)))

    plt.figure()
    plt.plot(tebd_times, exact_mags, label="Exact (ED)")
    plt.plot(tebd_times, tebd_mags, "--", label="TEBD (MPS)")
    plt.xlabel("time")
    plt.ylabel("average <Z>")
    plt.title(f"TFIM: n={args.n}, J={args.J}, h={args.h}, dt={args.dt}, chi={args.chi}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tebd_times, err)
    plt.xlabel("time")
    plt.ylabel("|error|")
    plt.title("Absolute error in <Z>")
    plt.show()


if __name__ == "__main__":
    main()