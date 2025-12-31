import subprocess
import sys


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    py = sys.executable  # uses the current venv python

    # Benchmark 1: error vs chi
    run([
        py, "scripts/benchmark_error_vs_chi.py",
        "--n", "8",
        "--dt", "0.05",
        "--steps", "60",
        "--chis", "2,4,8,16,32,64",
        "--save",
        "--no_show",
        "--outdir", "figures"
    ])

    # Benchmark 2: error vs dt
    run([
        py, "scripts/benchmark_error_vs_dt.py",
        "--n", "8",
        "--chi", "64",
        "--steps", "60",
        "--dts", "0.10,0.05,0.02,0.01",
        "--save",
        "--no_show",
        "--outdir", "figures"
    ])

    # Benchmark 3: runtime vs n (cap exact to avoid very long runs)
    run([
        py, "scripts/benchmark_runtime_vs_n.py",
        "--ns", "6,8,10,12,14",
        "--max_exact_n", "12",
        "--steps", "50",
        "--dt", "0.05",
        "--chi", "32",
        "--save",
        "--no_show",
        "--outdir", "figures"
    ])

    print("\nAll benchmarks finished! Check the figures/ directory.")


if __name__ == "__main__":
    main()