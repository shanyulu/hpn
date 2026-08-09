# Tencent HPN 2026 Issue6

## What This Implements

This submission optimizes ByteDance Flux `moe_ag_scatter` for the official shape `<Group,M,N,K> = <4,256,4096,7168>` on 4 GPUs. The final path keeps Flux's real V2 AGScatter execution and adds a finer consumer tile candidate (`tile_shape=(32,128,32)`) plus SM scheduling margin (`sm_margin=32`) so grouped GEMM can consume rank-ready token chunks with less coarse waiting than the original `tile_M=128` path.

## Result Snapshot

| Item | Value |
|---|---:|
| Hardware | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, PCIe, no NVLink |
| Official shape | `<4,256,4096,7168>` |
| Baseline median latency | 0.342304 ms |
| Optimized median latency | 0.323568 ms |
| Latency improvement | 5.47% |
| Throughput improvement | 5.79% |
| Derived overlap efficiency | 19.02% |
| Profiler comm+compute overlap | 82.53% |
| Profiler comm+GEMM-only overlap | 69.69% |
| Peak allocated memory delta | +4,608 bytes |
| Correctness | PASS, BF16 allclose, no NaN/Inf |

The profiler overlap number above uses communication P2P memcpy activity overlapped by compute-side kernels (`SCATTER_PREP + GROUPED_GEMM`). GEMM-only overlap is reported separately because Nsight Systems shows the PCIe P2P copies also overlap with scatter/workspace prep.

## One-Command Reproduction

From this repository after applying `optimized/flux_issue6.patch` to Flux and rebuilding Flux:

```bash
python3 src/code/issue6/baseline.py --variant optimized --output-dir /root/tencent-hpn-issue6/artifacts/issue6_repro_optimized
python3 src/code/issue6/overlap_metrics.py --input-dir /root/tencent-hpn-issue6/artifacts/issue6_repro_optimized --variant optimized --output /root/tencent-hpn-issue6/artifacts/issue6_repro_optimized/overlap.csv
```

## Links

- Methodology: `EXPERIMENT.md`
- Evidence index: `results/RESULTS_INDEX.md`
- Flux patch: `optimized/flux_issue6.patch`
- Compact benchmark table: `results/benchmark.csv`
- Compact profiler table: `results/profiling_summary.csv`
