# Issue6 Results Index

## Claims

| Claim | Experiment | Command / Config | Raw Artifact | Compact Evidence | Interpretation |
|---|---|---|---|---|---|
| Official shape runs on 4 GPUs | Baseline + optimized fused | `benchmark.py --B 1 --S 256 --H 7168 --ffn-hidden-size 16384 --G 4 --topk 1` | `/root/tencent-hpn-issue6/artifacts/final_overlap_audit/final_5run/` | `benchmark.csv` | Both baseline and optimized run the exact official shape. |
| Correctness is preserved | Torch reference vs Flux fused output | BF16 `atol=1e-2`, `rtol=1.5e-2` | Optimized summary JSON | `correctness.json` | All 4 ranks pass allclose; no NaN/Inf. |
| Latency improves | Five independent launches, each 100 measured iterations and 20 warmup | Median of launch medians | Raw summary JSONs | `benchmark.csv` | 0.341744 ms to 0.321904 ms, 5.81% faster. |
| Peak memory does not materially regress | CUDA peak memory stats | Same official benchmark | Raw summary JSONs | `memory.csv` | Peak allocated memory increases by 3,584 bytes vs B0 and decreases by 1,536 bytes vs prior B4. |
| Communication/GEMM overlap passes 80% | Nsight Systems iteration-level CUDA trace | `profile.py --variant optimized --sm-margin 0` | `/root/tencent-hpn-issue6/artifacts/final_overlap_audit/raw/final_sm0_measured_loop.nsys-rep` | `profiling_iterations.csv`, `profiling_summary.csv` | P2P communication overlaps grouped GEMM by 82.44% median, up from B0 73.55% and B4 71.47%. |
| Compute-side overlap is reported separately | Nsight Systems iteration-level CUDA trace | Same profile | Same raw profile | `profiling_summary.csv` | P2P communication overlaps scatter-prep plus grouped GEMM by 90.55%; this is auxiliary, not the primary comm/GEMM metric. |
| Derived overlap is reported separately | Component timing benchmark | compute-only, comm-only, fused | Raw summary JSONs | `overlap.csv` | Derived final overlap is 14.13%; it is not used as the timeline overlap claim. |
| Fine-grained tile candidate is the winning hparam | Tile ablation | tuning ranks at `sm_margin=32` | `/root/tencent-hpn-issue6/artifacts/issue6_benchmark_tile_rank_sweep/` | `ablation.csv` | `tile_shape=(32,128,32)` beats original 128-row and 64-row candidates. |
| Fused dispatch prep improves pure overlap | Fused topk1/EP1 dispatch-prep ablation | No extra P2P copies | `/root/tencent-hpn-issue6/artifacts/final_overlap_audit/` | `ablation.csv` | Prep kernels drop from 4 to 2, first-GEMM delay drops to 13.87 us, and comm/GEMM overlap reaches 82.44%. |
| Split-level AG readiness was evaluated and rejected | Split=2 experiment | row-aligned two-way signal split | `/root/tencent-hpn-issue6/artifacts/issue6_benchmark_official_final_split2_sm16/` | `ablation.csv` | Correct but slower, so not final. |
| SM120 compatibility is real GPU execution | Build + smoke tests | `./build.sh --arch 120 --sm-cores '108;132' --nvshmem --no_test --jobs 16` | `/root/tencent-hpn-issue6/artifacts/compatibility/` | `environment.json` | Compiles/runs real CUDA kernels; no CPU fallback. |

## Core Table

| Method | Latency ms | Comm ms | Compute ms | Derived overlap | Profiler comm+GEMM overlap | Profiler comm+compute overlap | Throughput tokens/s | Peak alloc bytes | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 Original | 0.341744 | 0.135440 | 0.222688 | 12.10% | 73.55% | 86.50% | 749,099 | 260,061,184 | PASS |
| B4 Tile32 current | 0.323568 | 0.135280 | 0.214016 | 19.02% | 71.47% | 84.63% | 791,178 | 260,066,304 | PASS |
| B8 Fused-prep final | 0.321904 | 0.134976 | 0.206000 | 14.13% | 82.44% | 90.55% | 795,268 | 260,064,768 | PASS |

## Files

- `benchmark.csv`: latency and throughput by method/component.
- `ablation.csv`: tile and rejected split ablations.
- `overlap.csv`: derived overlap calculation.
- `profiling_summary.csv`: Nsight Systems timeline overlap summary.
- `profiling_iterations.csv`: per-rank, per-iteration timeline overlap rows.
- `correctness.json`: per-rank correctness checks.
- `memory.csv`: peak allocation/reservation and delta.
- `environment.json`: hardware/software/repo environment.
- `figures/latency.svg`: generated from `benchmark.csv`.
