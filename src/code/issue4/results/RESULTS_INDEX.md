# ISSUE4 Results Index

This directory contains compact reviewer-facing evidence for the ISSUE4 NCCL bitwise reproducibility diagnostics run on 2026-08-09 UTC.

Raw `.bin` captures, verbose logs, timestamped experiment directories, and per-run metadata remain on the experiment host.  They are excluded from the PR by `results/.gitignore` to keep the review surface small.  The files below are the canonical evidence for the submitted report.

## Canonical Evidence Files

| File | Purpose |
|---|---|
| `environment_summary.json` | Hardware, topology, software versions, commits, and scope limit |
| `same_config_matrix.csv` | Reviewer summary for collective × algorithm × protocol same-configuration results |
| `algorithm_protocol_matrix.csv` | Detailed algorithm/protocol same-configuration matrix |
| `size_sweep.csv` | Default AllReduce / ReduceScatter message-size sweep from 1 KiB through 128 MiB |
| `controlled_injection.json` | Controlled 1-ULP detector validation |
| `ring_vs_tree.json` | Ring/Tree same-configuration checks and Ring-vs-Tree cross-configuration comparison |
| `nccl_tests_performance.csv` | `nccl-tests` latency/algbw/busbw data linked against PyTorch NCCL 2.27.3 |
| `nccl_tests_compatibility_summary.json` | System NCCL 2.25.1 failure vs PyTorch NCCL 2.27.3 successful rebuild |
| `nccl_trace_selection_summary.json` | NCCL algorithm/protocol selection lines from TRACE/TUNING logs |
| `figures/*.svg` | Reviewer figures generated from the CSV/JSON files above |

## Core Finding Mapping

| Finding | Experiment / config | Compact evidence | Result | Interpretation |
|---|---|---|---|---|
| Same-config AllReduce baseline | FP32, default NCCL config, 4 GPUs, fresh launches | `size_sweep.csv`, `algorithm_protocol_matrix.csv` | 0 divergent comparisons across retained AllReduce matrix | No same-configuration run-to-run divergence observed in measured range |
| Same-config ReduceScatter baseline | FP32, default/Ring/protocol configs, 4 GPUs, fresh launches | `size_sweep.csv`, `algorithm_protocol_matrix.csv` | 0 divergent comparisons where supported | No same-configuration run-to-run divergence observed in measured range |
| ReduceScatter Tree handling | FP32, `NCCL_ALGO=Tree`, 1 MiB, 4 GPUs | `algorithm_protocol_matrix.csv` | unsupported | NCCL reported no available algorithm/protocol for this forced configuration |
| Protocol sweep | default, Simple, LL, LL128 at 1 MiB | `algorithm_protocol_matrix.csv` | 0 divergent comparisons | Protocol overrides did not produce same-config divergence in measured runs |
| Message-size sweep | 1 KiB, 64 KiB, 1 MiB, 16 MiB, 128 MiB | `size_sweep.csv` | 0 divergent comparisons | Same-config results stayed bitwise identical across measured sizes |
| Controlled injection | 1-ULP perturbation at run 1 / call 1 / rank 0 / element 5 | `controlled_injection.json` | exact localization; max ordered ULP = 1 | Detector validation only, not NCCL nondeterminism |
| Ring vs Tree | AllReduce FP32 1 MiB; Ring and Tree checked independently before cross-compare | `ring_vs_tree.json` | Ring 0/20 divergent, Tree 0/20 divergent, Ring-vs-Tree 20/20 different | Cross-configuration numerical sensitivity, not same-configuration nondeterminism |
| NCCL path selection | default and forced TRACE/TUNING probes | `nccl_trace_selection_summary.json` | default Ring/Simple; forced Tree/LL128 selected | Algorithm/protocol findings are tied to observed NCCL selections |
| Blackwell compatibility | `nccl-tests` with system NCCL 2.25.1 vs PyTorch NCCL 2.27.3 | `nccl_tests_compatibility_summary.json` | 2.25.1 failed; 2.27.3 succeeded | Records observed compatibility split without unverified root-cause claims |
| Performance baseline | `nccl-tests` with PyTorch NCCL 2.27.3 | `nccl_tests_performance.csv` | latency/algbw/busbw retained | Performance numbers are from `nccl-tests`, not Python capture timing |

## Figures

- `figures/reproducibility_matrix.svg` — collective × algorithm × protocol reproducibility matrix.
- `figures/nccl_tests_performance.svg` — message size vs `nccl-tests` bus bandwidth.
- `figures/ring_vs_tree.svg` — Ring/Tree same-config stability vs Ring-vs-Tree cross-config difference.

Regenerate figures from committed evidence:

```bash
python3 -m issue4.make_figures
```

## Scope

The evidence is scoped to the recorded 4× NVIDIA RTX PRO 6000 Blackwell Server Edition PCIe single-node system and software stack.  It does not generalize to NVLink/NVSwitch systems, multi-node RDMA, other GPU counts, or other NCCL/PyTorch/driver versions.
