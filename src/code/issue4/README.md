# NCCL Bitwise Reproducibility Diagnostics

A reproducibility and forensic toolkit for NCCL AllReduce and ReduceScatter, validated on a single-node 4× NVIDIA RTX PRO 6000 Blackwell PCIe system.

## Key Findings

- No same-configuration run-to-run bitwise divergence was observed across the measured AllReduce / ReduceScatter matrix.
- A controlled 1-ULP perturbation was localized exactly to its run / call / rank / element.
- Ring and Tree were individually reproducible but produced different outputs for identical inputs, showing reduction-path numerical sensitivity, not nondeterminism.
- System NCCL 2.25.1 failed for `nccl-tests` on this Blackwell stack; rebuilding `nccl-tests` against PyTorch wheel NCCL 2.27.3 succeeded.

See `EXPERIMENT.md` for methodology and `results/RESULTS_INDEX.md` for evidence.

## Quick Start

Run from `hpn/src/code`:

```bash
python3 -m issue4.run_cpu_tests

python3 -m issue4.diagnose \
  --collective all_reduce \
  --dtype float32 \
  --size-bytes 1MiB \
  --runs 5 \
  --calls 20 \
  --nproc-per-node 4

python3 -m issue4.diagnose \
  --collective reduce_scatter \
  --dtype float32 \
  --size-bytes 1MiB \
  --runs 5 \
  --calls 20 \
  --nproc-per-node 4
```

Default baseline runs sanitize common NCCL override variables before launching workers: `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_P2P_DISABLE`, and `NCCL_NVLS_ENABLE`.

## What the Tool Checks

`diagnose.py` launches each independent run as a fresh `torchrun` subprocess.  Each worker creates a new NCCL process group, executes the same collective sequence, writes raw output bytes, exits, and is compared against run 0.

The comparator checks raw bytes, not tolerance.  It reports first differing element / byte / bit, changed element / byte / bit counts, max absolute error, max relative error, and ULP distance.

## Main Files

- `diagnose.py` — independent-run orchestrator.
- `worker.py` — one NCCL distributed execution.
- `data_generator.py` — deterministic rank-distinct tensor generator.
- `comparator.py` — bitwise and ULP comparator.
- `sweep.py` — algorithm / protocol / size matrix runner.
- `reproduce_case.py` — controlled injection and Ring-vs-Tree cases.
- `compare_experiments.py` — cross-experiment comparison.
- `make_figures.py` — regenerates SVG figures from canonical CSV/JSON evidence.
- `run_cpu_tests.py`, `tests/` — dependency-free CPU tests.
- `EXPERIMENT.md` — methodology, results, limitations, and acceptance mapping.
- `results/` — compact reviewer-facing evidence and figures.

## Canonical Evidence

The PR keeps compact evidence files only.  Timestamped raw experiment directories, `.bin` captures, verbose logs, and per-run metadata remain on the experiment host and are ignored by `results/.gitignore`.

- `results/same_config_matrix.csv`
- `results/size_sweep.csv`
- `results/algorithm_protocol_matrix.csv`
- `results/controlled_injection.json`
- `results/ring_vs_tree.json`
- `results/nccl_tests_performance.csv`
- `results/nccl_tests_compatibility_summary.json`
- `results/nccl_trace_selection_summary.json`
- `results/environment_summary.json`

Regenerate figures from committed evidence:

```bash
python3 -m issue4.make_figures
```

## Terminology

- Same-configuration reproducibility: same input, rank mapping, world size, software stack, topology, NCCL config, and collective call sequence across independent fresh launches.
- Cross-configuration numerical sensitivity: outputs differ after intentionally changing an algorithm or protocol path.
- Controlled injection: detector self-test using an artificial perturbation.

Ring-vs-Tree differences are not reported as same-configuration nondeterminism.  Controlled injection is not reported as NCCL nondeterminism.  A measured negative result is not a proof that NCCL is deterministic in general.

## Performance Data

Performance numbers in the report come from `nccl-tests` linked against PyTorch wheel NCCL 2.27.3.  Python capture timings are not used as NCCL kernel performance measurements.
