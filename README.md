# Quantum TEBD Simulator (MPS-based)

A classical simulator for 1D quantum many-body systems using **Matrix Product States (MPS)** and **Time-Evolving Block Decimation (TEBD)**.

This repository implements TEBD from scratch (tensor contractions, SVD-based refactorization, truncation strategies), validates the simulator against exact diagonalization (ED) for small system sizes, and benchmarks the accuracy–cost tradeoffs that arise from finite bond dimension \(\chi\) and finite Trotter time step \(\Delta t\). It also includes entanglement entropy tracking (per bond) to diagnose when and where the simulation becomes truncation-limited.

---

## Overview

The project has three core components:

1. **MPS representation**  
   Efficient storage of a quantum state on a 1D chain as a product of local tensors, avoiding the exponential cost of a full state vector when entanglement is limited.

2. **TEBD time evolution**  
   Simulation of unitary time evolution under a local Hamiltonian using a second-order (Strang) Suzuki–Trotter decomposition into local gates. Each two-site gate update uses SVD to restore the MPS factorization and truncates the bond dimension to control computational cost.

3. **Validation and benchmarking**  
   For small \(N\), ED provides a ground-truth baseline. We use it to quantify:
   - **Trotter error** as a function of \(\Delta t\)
   - **Truncation error** as a function of \(\chi\)
   - Runtime scaling versus \(N\) for TEBD compared to ED  
   We additionally measure **entanglement entropy** to explain why and when truncation becomes necessary.

---

## Physics Model

We simulate the 1D Transverse Field Ising Model (TFIM) with open boundary conditions:

\[
H = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} \;-\; h \sum_{i=0}^{N-1} X_i
\]

where:
- \(J\) sets the nearest-neighbor interaction strength
- \(h\) sets the transverse field strength
- \(Z\), \(X\) are Pauli operators

Time evolution is governed by:

\[
|\psi(t+\Delta t)\rangle = e^{-iH\Delta t}|\psi(t)\rangle
\]

In TEBD, we approximate \(e^{-iH\Delta t}\) using a **second-order Trotter scheme** and apply local gates sequentially.

---

## Algorithm Details

### Matrix Product States (MPS)

An \(N\)-qubit state has \(2^N\) amplitudes in the computational basis, which becomes infeasible beyond modest \(N\). An MPS expresses the same state as a chain of tensors:

- Each site tensor \(A^{[i]}\) has shape \((\chi_{i-1}, 2, \chi_i)\)
- The bond dimensions \(\chi_i\) control representational power
- The maximum bond dimension \(\chi\) is the primary accuracy–cost knob

MPS is efficient for many 1D systems because low-energy states and short-time evolutions often obey an entanglement area law, meaning the required \(\chi\) grows slowly with \(N\) and time (until entanglement becomes large).

### TEBD

TEBD exploits locality: the TFIM Hamiltonian decomposes into nearest-neighbor interaction terms plus on-site field terms. Using Strang splitting:

\[
e^{-iH\Delta t} \approx
e^{-iH_X\Delta t/2}
\left(
\prod_{\text{even } i} e^{-ih_{i,i+1}\Delta t}
\right)
\left(
\prod_{\text{odd } i} e^{-ih_{i,i+1}\Delta t}
\right)
e^{-iH_X\Delta t/2}
\]

where:
- \(H_X = -h \sum_i X_i\) (one-site gates)
- \(h_{i,i+1} = -J Z_i Z_{i+1}\) (two-site gates)

Each two-site update:
1. contracts two neighboring tensors into a rank-4 object \(\Theta\)
2. applies the two-site unitary gate on physical indices
3. reshapes \(\Theta\) into a matrix and performs SVD
4. truncates to \(\chi_{\max}\) to control cost
5. records truncation error and entanglement entropy from singular values

---

## Entanglement Entropy Tracking

Given the singular values \(s_k\) from the SVD on a bond, we compute the Von Neumann entanglement entropy:

\[
p_k = \frac{s_k^2}{\sum_j s_j^2}, \qquad
S = -\sum_k p_k \log p_k
\]

Entropy is tracked:
- per bond (cut between \(i\) and \(i+1\))
- over time during TEBD evolution

This provides a diagnostic for:
- where entanglement is concentrated
- when the simulation becomes truncation-limited
- why increasing \(\chi\) improves accuracy (until truncation disappears)

---

## Repository Structure

```
quantum-tebd/
├── mps/                # MPS core, TEBD, entropy, observables
├── models/             # TFIM/Ising model gate generation
├── exact/              # Exact diagonalization reference implementation
├── scripts/            # Simulation, validation, and benchmarking scripts
├── tests/              # Unit tests (numerics and physics correctness)
├── figures/            # Generated benchmark plots
└── README.md
```

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you have a `pyproject.toml` with editable install support:

```bash
pip install -e .
```

---

## Usage

### Run a TFIM TEBD simulation

```bash
PYTHONPATH=. python scripts/run_ising_tebd.py \
  --n 10 --J 1.0 --h 1.0 --dt 0.05 --steps 100 --chi 32 \
  --state 0000000000
```

This script produces:
- magnetization \(\langle Z \rangle\) vs time
- entanglement entropy vs time (max and middle bond)
- entanglement entropy heatmap (bond index × time)

### Compare TEBD vs Exact (ED)

```bash
PYTHONPATH=. python scripts/compare_tebd_vs_exact.py \
  --n 8 --dt 0.05 --chi 64
```

This produces:
- TEBD vs ED magnetization curves
- absolute error over time

---

## Testing

Run the full test suite:

```bash
pytest -q
```

Tests include:
- MPS normalization and shape checks
- gate application correctness (including comparisons to dense evolution for small cases)
- TEBD norm preservation
- TEBD vs ED agreement within tolerance
- entanglement entropy correctness (Bell state entropy \(\approx \ln(2)\))

---

## Results and Benchmarks

All benchmarks can be reproduced via:

```bash
PYTHONPATH=. python scripts/run_all_benchmarks.py
```

This regenerates all plots in `figures/`.

### Benchmark 1: Error vs Bond Dimension \(\chi\)

**Files**
- `figures/error_vs_chi.png`
- `figures/trunc_vs_chi.png`
- `figures/runtime_vs_chi.png`

**Method**
- Fix system size \(N=8\), \(\Delta t=0.05\), steps=60, \(J=1\), \(h=1\)
- Compute exact magnetization curve using ED
- Run TEBD for a sweep of \(\chi\) values
- Measure max absolute error:
  \[
  \max_t \left|\langle Z \rangle_{\text{TEBD}}(t) - \langle Z \rangle_{\text{ED}}(t)\right|
  \]
- Track total truncation error (sum over TEBD steps)

**Observed behavior (representative run)**
- \(\chi=2\): max error \(\approx 2.545\times 10^{-1}\), truncation sum \(\approx 1.455\times 10^{-1}\)
- \(\chi=4\): max error \(\approx 4.996\times 10^{-2}\), truncation sum \(\approx 5.946\times 10^{-2}\)
- \(\chi=8\): max error \(\approx 5.945\times 10^{-3}\), truncation sum \(\approx 3.214\times 10^{-3}\)
- \(\chi\ge 16\): truncation sum \(\approx 0\), error saturates \(\approx 1.795\times 10^{-3}\)

**Interpretation**
- For small \(\chi\), truncation dominates: the MPS cannot represent the entanglement produced by time evolution, so singular values are discarded and the state deviates from the exact result.
- As \(\chi\) increases, truncation decreases rapidly; once truncation error reaches zero, increasing \(\chi\) no longer improves accuracy.
- The plateau at \(\chi \ge 16\) indicates the remaining error is primarily **Trotter error** rather than MPS truncation.

### Benchmark 2: Error vs Time Step \(\Delta t\) (Trotter Error)

**Files**
- `figures/error_vs_dt.png`
- `figures/runtime_vs_dt.png`

**Method**
- Fix \(N=8\), steps=60, \(\chi=64\) (chosen so truncation \(\approx 0\))
- Sweep \(\Delta t\in \{0.10, 0.05, 0.02, 0.01\}\)
- Compare TEBD against ED as above

**Observed behavior (representative run)**
- \(\Delta t=0.10\): error \(\approx 7.157\times 10^{-3}\)
- \(\Delta t=0.05\): error \(\approx 1.795\times 10^{-3}\)
- \(\Delta t=0.02\): error \(\approx 2.869\times 10^{-4}\)
- \(\Delta t=0.01\): error \(\approx 6.80\times 10^{-5}\)

**Interpretation**
- Error decreases rapidly as \(\Delta t\) decreases, consistent with a second-order Strang splitting scheme.
- Because truncation \(\approx 0\) across all runs, this benchmark isolates the error source to the Trotter approximation alone.
- In practice, the best \(\Delta t\) depends on the desired accuracy and runtime budget, since smaller \(\Delta t\) requires more steps to reach the same physical time.

### Benchmark 3: Runtime Scaling vs System Size \(N\) (TEBD vs ED)

**File**
- `figures/runtime_vs_n.png`

**Method**
- Fix \(\Delta t=0.05\), steps=50, \(\chi=32\), \(J=1\), \(h=1\)
- Measure wall-clock runtime for:
  - TEBD evolution
  - Exact diagonalization evolution (up to \(N\) where feasible)
- Plot runtime on a log scale versus \(N\)

**Observed behavior (representative run)**
- \(N=10\): TEBD \(\approx 0.0377\) s, ED \(\approx 0.3180\) s
- \(N=12\): TEBD \(\approx 0.0960\) s, ED \(\approx 15.10\) s
- \(N=14\): TEBD \(\approx 0.1721\) s, ED \(\approx 1055.93\) s

**Interpretation**
- ED scales exponentially with \(N\) because it operates on \(2^N \times 2^N\) matrices.
- TEBD scales far more gently because it operates locally and its complexity is primarily driven by \(\chi\) rather than \(2^N\).
- This benchmark demonstrates the regime where tensor-network methods become essential: beyond modest \(N\), ED becomes infeasible while TEBD continues to run.

### Entanglement Diagnostics

The TEBD simulation script produces:
- max bond entropy vs time
- middle-bond entropy vs time
- entropy heatmap across bonds over time

Key observed patterns:
- Starting from a product state, entanglement grows rapidly under TFIM evolution.
- The entropy heatmap shows the strongest entanglement developing in the middle bonds (for open boundary conditions).
- Entropy saturates near \(\ln(\chi)\), which directly demonstrates the representational limit imposed by finite bond dimension.
- This provides a concrete diagnostic linking simulation accuracy to the entanglement structure of the evolving state.

## Limitations

- Current observable computation converts the MPS to a dense state (`to_dense()`) for simplicity in small-\(N\) validation workflows. This is not scalable for large \(N\).
- Exact diagonalization is only used for small \(N\) due to exponential memory/time cost.
- The simulator currently targets open boundary conditions and qubit physical dimension \(d=2\).

