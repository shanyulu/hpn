# Tencent HPN 2026 Issue6

## What This Implements

This submission optimizes ByteDance Flux `moe_ag_scatter` for the official shape `<Group,M,N,K> = <4,256,4096,7168>` on 4 GPUs. The final path keeps Flux's real V2 AGScatter execution, uses the tuned `tile_shape=(32,128,32)` grouped-GEMM consumer, and fuses the official `topk=1`, `EP=1` dispatch-prep path so grouped GEMM starts earlier after existing P2P rank-ready communication.

## Result Snapshot

| Item | Value |
|---|---:|
| Hardware | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, PCIe, no NVLink |
| Official shape | `<4,256,4096,7168>` |
| Baseline median latency | 0.341744 ms |
| Optimized median latency | 0.321904 ms |
| Latency improvement | 5.81% |
| Throughput improvement | 6.16% |
| Derived overlap efficiency | 14.13% |
| Profiler comm+GEMM-only overlap | 82.44% |
| Profiler comm+compute-side overlap | 90.55% |
| Peak allocated memory delta | +3,584 bytes |
| Correctness | PASS, BF16 allclose, no NaN/Inf |

Profiler communication/GEMM overlap is measured directly from Nsight Systems iteration-level CUDA timelines: P2P memcpy activity overlapped by grouped-GEMM kernel activity, divided by P2P communication active time. The compute-side metric (`SCATTER_PREP + GROUPED_GEMM`) is reported separately and is not used to claim pure communication/GEMM overlap.

## One-Command Reproduction

From this repository after applying `optimized/flux_issue6.patch` to Flux and rebuilding Flux:

```bash
python3 src/code/issue6/baseline.py --variant optimized --output-dir /root/tencent-hpn-issue6/artifacts/issue6_repro_optimized
python3 src/code/issue6/overlap_metrics.py --input-dir /root/tencent-hpn-issue6/artifacts/issue6_repro_optimized --variant optimized --output /root/tencent-hpn-issue6/artifacts/issue6_repro_optimized/overlap.csv
python3 src/code/issue6/profile.py --variant optimized --sm-margin 0 --output /root/tencent-hpn-issue6/artifacts/profiling/raw/issue6_final
```

## Links

- Methodology: `EXPERIMENT.md`
- Evidence index: `results/RESULTS_INDEX.md`
- Flux patch: `optimized/flux_issue6.patch`
- Compact benchmark table: `results/benchmark.csv`
- Compact profiler table: `results/profiling_summary.csv`
- Iteration-level profiler table: `results/profiling_iterations.csv`
