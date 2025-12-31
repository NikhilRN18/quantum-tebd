import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from mps.mps_state import MPS
from mps.tebd import tebd_step_ising

from exact.exact_diag import tfim_hamiltonian, evolve_state_exact, basis_state

import os


def maybe_savefig(save, outdir, filename):
    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

def time_tebd(n, J, h, dt, steps, chi, state):
    mps = MPS.product_state(state)

    t0 = time.perf_counter()
    for _ in range(steps):
        out = tebd_step_ising(mps, J=J, h=h, dt=dt, chi_max=chi)
        # tebd_step_ising may return trunc or (trunc, entropy)
        _ = out[0] if isinstance(out, tuple) else out
    t1 = time.perf_counter()

    return t1 - t0


def time_exact(n, J, h, dt, steps, state):
    psi0 = basis_state(state)

    t0 = time.perf_counter()
    H = tfim_hamiltonian(n, J=J, h=h)
    _ = evolve_state_exact(psi0, H, dt=dt, steps=steps)
    t1 = time.perf_counter()

    return t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--chi", type=int, default=32)

    parser.add_argument("--ns", type=str, default="6,8,10,12,14")
    parser.add_argument("--max_exact_n", type=int, default=14)
    parser.add_argument("--save", action="store_true", help="save plots to files")
    parser.add_argument("--outdir", type=str, default="figures", help="output directory for plots")
    parser.add_argument("--no_show", action="store_true", help="do not display plots")

    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.ns.split(",")]

    tebd_times = []
    exact_times = []

    for n in ns:
        state = "0" * n

        print(f"\nN={n}: timing TEBD...")
        t_tebd = time_tebd(n, args.J, args.h, args.dt, args.steps, args.chi, state)
        tebd_times.append(t_tebd)
        print(f"  TEBD runtime:  {t_tebd:.4f} s")

        if n <= args.max_exact_n:
            print(f"N={n}: timing Exact Diagonalization...")
            t_exact = time_exact(n, args.J, args.h, args.dt, args.steps, state)
            exact_times.append(t_exact)
            print(f"  Exact runtime: {t_exact:.4f} s")
        else:
            exact_times.append(np.nan)
            print(f"  Exact runtime: skipped (n > {args.max_exact_n})")

    # --- Plot ---
    plt.figure()
    plt.plot(ns, tebd_times, marker="o", label="TEBD (MPS)")

    exact_mask = [not np.isnan(x) for x in exact_times]
    ns_exact = [n for n, ok in zip(ns, exact_mask) if ok]
    exact_vals = [x for x in exact_times if not np.isnan(x)]
    plt.plot(ns_exact, exact_vals, marker="o", label="Exact (ED)")

    plt.yscale("log")
    plt.xlabel("system size N")
    plt.ylabel("runtime (seconds, log scale)")
    plt.title(f"Runtime scaling vs N (dt={args.dt}, steps={args.steps}, chi={args.chi})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    maybe_savefig(args.save, args.outdir, "runtime_vs_n.png")

    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()