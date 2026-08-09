# Issue6 Results Index

## Claims

| Claim | Experiment | Command / Config | Raw Artifact | Compact Evidence | Interpretation |
|---|---|---|---|---|---|
| Official shape runs on 4 GPUs | Baseline + optimized fused | `benchmark.py --B 1 --S 256 --H 7168 --ffn-hidden-size 16384 --G 4 --topk 1` | `/root/tencent-hpn-issue6/artifacts/issue6_benchmark_official_full/`, `/root/tencent-hpn-issue6/artifacts/issue6_benchmark_official_final_current/` | `benchmark.csv` | Both baseline and optimized run the exact official shape. |
| Correctness is preserved | Torch reference vs Flux fused output | BF16 `atol=1e-2`, `rtol=1.5e-2` | Optimized summary JSON | `correctness.json` | All 4 ranks pass allclose; no NaN/Inf. |
| Latency improves | 100 measured iterations, 20 warmup | Median distributed max across ranks | Raw summary JSONs | `benchmark.csv` | 0.342304 ms to 0.323568 ms, 5.47% faster. |
| Peak memory does not materially regress | CUDA peak memory stats | Same official benchmark | Raw summary JSONs | `memory.csv` | Peak allocated memory increases by 4,608 bytes. |
| Communication overlap is visible in profiler | Nsight Systems CUDA/NVTX trace | `profile.py --variant optimized --sm-margin 32` | `/root/tencent-hpn-issue6/artifacts/profiling/raw/optimized_sm32_measured_loop.nsys-rep` | `profiling_summary.csv` | P2P communication overlaps 82.53% with compute-side kernels and 69.69% with GEMM-only kernels. |
| Derived overlap is reported separately | Component timing benchmark | compute-only, comm-only, fused | Raw summary JSONs | `overlap.csv` | Derived optimized overlap is 19.02%; this includes launch/prep/resource effects beyond raw timeline overlap. |
| Fine-grained tile candidate is the winning hparam | Tile ablation | tuning ranks at `sm_margin=32` | `/root/tencent-hpn-issue6/artifacts/issue6_benchmark_tile_rank_sweep/` | `ablation.csv` | `tile_shape=(32,128,32)` beats original 128-row and 64-row candidates. |
| Split-level AG readiness was evaluated and rejected | Split=2 experiment | row-aligned two-way signal split | `/root/tencent-hpn-issue6/artifacts/issue6_benchmark_official_final_split2_sm16/` | `ablation.csv` | Correct but slower, so not final. |
| SM120 compatibility is real GPU execution | Build + smoke tests | `./build.sh --arch 120 --sm-cores '108;132' --nvshmem --no_test --jobs 16` | `/root/tencent-hpn-issue6/artifacts/compatibility/` | `environment.json` | Compiles/runs real CUDA kernels; no CPU fallback. |

## Core Table

| Method | Latency ms | Comm ms | Compute ms | Derived overlap | Profiler comm+compute overlap | Throughput tokens/s | Peak alloc bytes | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0 Original | 0.342304 | 0.135440 | 0.222688 | 11.68% | 84.72% | 747,873 | 260,061,696 | PASS |
| B4 Final | 0.323568 | 0.135280 | 0.214016 | 19.02% | 82.53% | 791,178 | 260,066,304 | PASS |

## Files

- `benchmark.csv`: latency and throughput by method/component.
- `ablation.csv`: tile and rejected split ablations.
- `overlap.csv`: derived overlap calculation.
- `profiling_summary.csv`: Nsight Systems timeline overlap summary.
- `correctness.json`: per-rank correctness checks.
- `memory.csv`: peak allocation/reservation and delta.
- `environment.json`: hardware/software/repo environment.
- `figures/latency.svg`: generated from `benchmark.csv`.
