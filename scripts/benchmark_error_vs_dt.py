import argparse
import time
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


def run_tebd(n, J, h, dt, steps, chi, state):
    mps = MPS.product_state(state)

    mags = []
    truncs = []

    t0 = time.perf_counter()
    for step in range(steps + 1):
        mags.append(magnetization_z(mps))
        if step < steps:
            out = tebd_step_ising(mps, J=J, h=h, dt=dt, chi_max=chi)
            trunc = out[0] if isinstance(out, tuple) else out
            truncs.append(trunc)
    t1 = time.perf_counter()

    return np.array(mags), float(np.sum(truncs)), (t1 - t0)


def run_exact(n, J, h, dt, steps, state):
    psi0 = basis_state(state)
    H = tfim_hamiltonian(n, J=J, h=h)
    psis = evolve_state_exact(psi0, H, dt=dt, steps=steps)
    mags = np.array([magnetization_z_dense(psi, n) for psi in psis])
    return mags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--state", type=str, default="00000000")

    parser.add_argument("--dts", type=str, default="0.10,0.05,0.02,0.01")
    args = parser.parse_args()

    if len(args.state) != args.n:
        raise ValueError("bitstring length must match n")

    dts = [float(x.strip()) for x in args.dts.split(",")]

    errors = []
    trunc_sums = []
    runtimes = []

    for dt in dts:
        print(f"\nRunning exact baseline for dt={dt}...")
        exact_mags = run_exact(args.n, args.J, args.h, dt, args.steps, args.state)

        print(f"Running TEBD for dt={dt} (chi={args.chi})...")
        tebd_mags, trunc_sum, runtime = run_tebd(args.n, args.J, args.h, dt, args.steps, args.chi, args.state)

        err = float(np.max(np.abs(tebd_mags - exact_mags)))

        print(f"  max error = {err:.6f}")
        print(f"  sum trunc errors = {trunc_sum:.6e}")
        print(f"  runtime = {runtime:.3f}s")

        errors.append(err)
        trunc_sums.append(trunc_sum)
        runtimes.append(runtime)

    # --- Plot: error vs dt ---
    plt.figure()
    plt.plot(dts, errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time step dt")
    plt.ylabel("max |<Z>_TEBD - <Z>_exact|")
    plt.title(f"Error vs dt (n={args.n}, chi={args.chi}, steps={args.steps})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    # --- Plot: truncation vs dt ---
    plt.figure()
    plt.plot(dts, trunc_sums, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time step dt")
    plt.ylabel("sum truncation errors")
    plt.title("Truncation vs dt")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    # --- Plot: runtime vs dt ---
    plt.figure()
    plt.plot(dts, runtimes, marker="o")
    plt.xscale("log")
    plt.xlabel("time step dt")
    plt.ylabel("runtime (seconds)")
    plt.title("Runtime vs dt")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()