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
import os

def maybe_savefig(save, outdir, filename):
    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

def run_tebd(n, J, h, dt, steps, chi, state):
    mps = MPS.product_state(state)

    mags = []
    truncs = []
    t0 = time.perf_counter()

    for step in range(steps + 1):
        mags.append(magnetization_z(mps))
        if step < steps:
            out = tebd_step_ising(mps, J=J, h=h, dt=dt, chi_max=chi)
            # tebd_step_ising may return either trunc or (trunc, entropy)
            if isinstance(out, tuple):
                trunc = out[0]
            else:
                trunc = out
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
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--state", type=str, default="00000000")

    parser.add_argument("--chis", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--save", action="store_true", help="save plots to files")
    parser.add_argument("--outdir", type=str, default="figures", help="output directory for plots")
    parser.add_argument("--no_show", action="store_true", help="do not display plots")
    args = parser.parse_args()

    if len(args.state) != args.n:
        raise ValueError("bitstring length must match n")

    chis = [int(x.strip()) for x in args.chis.split(",")]

    print("Running exact diagonalization baseline...")
    exact_mags = run_exact(args.n, args.J, args.h, args.dt, args.steps, args.state)

    errors = []
    trunc_sums = []
    runtimes = []

    for chi in chis:
        print(f"\nRunning TEBD with chi={chi}...")
        tebd_mags, trunc_sum, runtime = run_tebd(
            args.n, args.J, args.h, args.dt, args.steps, chi, args.state
        )

        err = float(np.max(np.abs(tebd_mags - exact_mags)))

        print(f"  max error = {err:.6f}")
        print(f"  sum trunc errors = {trunc_sum:.6e}")
        print(f"  runtime = {runtime:.3f}s")

        errors.append(err)
        trunc_sums.append(trunc_sum)
        runtimes.append(runtime)

    # --- Plot: error vs chi ---
    plt.figure()
    plt.plot(chis, errors, marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("bond dimension chi")
    plt.ylabel("max |<Z>_TEBD - <Z>_exact|")
    plt.title(f"Error vs chi (n={args.n}, dt={args.dt}, steps={args.steps})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    maybe_savefig(args.save, args.outdir, "error_vs_chi.png")
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # --- Plot: truncation sum vs chi ---
    plt.figure()
    plt.plot(chis, trunc_sums, marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("bond dimension chi")
    plt.ylabel("sum truncation errors")
    plt.title("Truncation vs chi")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    maybe_savefig(args.save, args.outdir, "trunc_vs_chi.png")
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # --- Plot: runtime vs chi ---
    plt.figure()
    plt.plot(chis, runtimes, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("bond dimension chi")
    plt.ylabel("runtime (seconds)")
    plt.title("Runtime vs chi")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    maybe_savefig(args.save, args.outdir, "runtime_vs_chi.png")
    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()