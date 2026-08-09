# Issue6 Experiment Method

## Environment

Environment facts are recorded in `results/environment.json` and raw host probes under `/root/tencent-hpn-issue6/artifacts/environment/`. The tested system has 4 RTX PRO 6000 Blackwell Server Edition GPUs, compute capability 12.0, PCIe/NUMA topology, no NVLink/NVSwitch, PyTorch 2.8.0+cu128, CUDA toolkit 12.8, and NCCL 2.27.3.

## Compatibility Gate

Flux upstream does not natively register SM120 in its documented paths. The compatibility patch compiles real `sm_120` CUDA cubins and maps runtime selection to the SM80 V2 path with `FLUX_SM120_LOGICAL_ARCH=80`. The SM90 logical path compiled but failed CUTLASS initialization, so it is not used for the submitted result.

## Baseline

Baseline uses the original Flux V2 AGScatter hparams for BF16/BF16 grouped GEMM: `tile_shape=(128,128,32)`, `warp_shape=(64,64,32)`, StreamK, raster heuristic. It runs the official shape with 4 GPUs, `topk=1`, `dist=uniform`, `stable_index=true`, `warmup=20`, and `iters=100`.

## Optimization

The first optimization adds targeted V2 hparams for the official MoE shape:

- Original: `tile_shape=(128,128,32)`
- Candidate: `tile_shape=(64,128,32)`
- Candidate: `tile_shape=(32,128,32)`
- Candidate: `tile_shape=(16,128,32)`

The tuned consumer is `tile_shape=(32,128,32)`, `warp_shape=(32,64,32)`, StreamK, raster along N. This reduces consumer tile granularity from a tile that spans the full 64-row expert to a 32-row tile that aligns closer to the approximately 16-row source-rank chunks in the official uniform routing case.

The final overlap optimization addresses the dispatch-prep gap before grouped GEMM. For the official `topk=1`, `EP=1` path, it fuses gather-index inversion and AG scatter sort into one GPU kernel, keeps `sorted_scatter_index` in EP-local output coordinates, and points grouped-GEMM output pointers at the full EP output base. This removes the separate per-expert scatter-index normalization kernel. The final path keeps the same ring1d P2P communication count and uses `sm_margin=0`.

The resulting pre-GEMM dispatch path is reduced from four kernels to two kernels:

- Original/B4: gather-index inversion, AG scatter sort, scatter-index normalization, workspace preparation
- Final/B8: fused topk1 EP-local dispatch sort, workspace preparation

## Overlap Methodology

Derived overlap is calculated from median component timings:

```text
Tc = compute-only latency
Tm = communication-only latency
Tf = fused latency
Tserial = Tc + Tm
Tideal = max(Tc, Tm)
Derived overlap = (Tserial - Tf) / (Tserial - Tideal)
```

Profiler overlap is calculated from Nsight Systems SQLite exports inside the `ISSUE6_MEASURED_LOOP` NVTX range. Communication activity is P2P memcpy on this PCIe host. The primary profiler metric is grouped GEMM only:

```text
comm_gemm_overlap_pct = overlap(COMM, GROUPED_GEMM) / active(COMM) * 100
```

The auxiliary compute-side metric is reported separately:

```text
comm_compute_side_overlap_pct = overlap(COMM, SCATTER_PREP + GROUPED_GEMM) / active(COMM) * 100
```

`results/profiling_iterations.csv` stores per-rank, per-iteration rows. `results/profiling_summary.csv` stores medians and distribution summaries. CPU NVTX range duration is not treated as CUDA active time.

## Final Validation

Final B0 and B8 fused latency was measured across five independent launches with 20 warmup iterations and 100 measured iterations per launch. Main latency and throughput numbers use the median of per-launch medians. The final B8 median is 0.321904 ms, versus B0 0.341744 ms and the previous B4 PR state 0.323568 ms.

## Correctness

The benchmark constructs a Torch reference path using Flux's test utilities for communication, scatter, and GEMM. BF16 comparisons use `atol=1e-2` and `rtol=1.5e-2`. Final optimized output passes allclose on all 4 ranks with no NaN or Inf.

## NCU Limitation

Nsight Compute was invoked against the final CUTLASS grouped GEMM kernel. The driver rejected performance-counter access on every GPU with `ERR_NVGPUCTRPERM`. Nsight Systems still records launch geometry, register count, shared memory, P2P copies, and kernel timing; counter-level metrics such as SM active and tensor utilization require enabling NVIDIA performance counters on the host.

## Rejected Ablation

A row-aligned two-way AG signal split was implemented and tested. It preserved correctness but increased P2P memcpy launch count and regressed official-shape latency to 0.346608 ms, so it is not part of the final patch.
