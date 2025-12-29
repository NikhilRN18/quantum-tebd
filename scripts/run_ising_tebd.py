import argparse
import numpy as np
import matplotlib.pyplot as plt

from mps.mps_state import MPS
from mps.tebd import tebd_step_ising
from mps.observables import magnetization_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="Ising coupling")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field")
    parser.add_argument("--dt", type=float, default=0.05, help="time step")
    parser.add_argument("--steps", type=int, default=50, help="number of steps")
    parser.add_argument("--chi", type=int, default=32, help="max bond dimension")
    parser.add_argument("--state", type=str, default="0000000000", help="initial bitstring (length n)")
    args = parser.parse_args()

    if len(args.state) != args.n:
        raise ValueError("Initial state bitstring length must equal n")

    mps = MPS.product_state(args.state)
    mps.check_shapes()

    times = []
    mags = []
    truncs = []

    t = 0.0
    for step in range(args.steps + 1):
        times.append(t)
        mags.append(magnetization_z(mps))

        if step < args.steps:
            trunc = tebd_step_ising(mps, J=args.J, h=args.h, dt=args.dt, chi_max=args.chi)
            truncs.append(trunc)
            t += args.dt

    print("Final <Z> =", mags[-1])

    plt.figure()
    plt.plot(times, mags)
    plt.xlabel("time")
    plt.ylabel("average <Z>")
    plt.title(f"TFIM TEBD: n={args.n}, J={args.J}, h={args.h}, dt={args.dt}, chi={args.chi}")
    plt.show()


if __name__ == "__main__":
    main()