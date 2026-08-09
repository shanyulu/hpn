# Issue6 Experiment Method

## Environment

Environment facts are recorded in `results/environment.json` and raw host probes under `/root/tencent-hpn-issue6/artifacts/environment/`. The tested system has 4 RTX PRO 6000 Blackwell Server Edition GPUs, compute capability 12.0, PCIe/NUMA topology, no NVLink/NVSwitch, PyTorch 2.8.0+cu128, CUDA toolkit 12.8, and NCCL 2.27.3.

## Compatibility Gate

Flux upstream does not natively register SM120 in its documented paths. The compatibility patch compiles real `sm_120` CUDA cubins and maps runtime selection to the SM80 V2 path with `FLUX_SM120_LOGICAL_ARCH=80`. The SM90 logical path compiled but failed CUTLASS initialization, so it is not used for the submitted result.

## Baseline

Baseline uses the original Flux V2 AGScatter hparams for BF16/BF16 grouped GEMM: `tile_shape=(128,128,32)`, `warp_shape=(64,64,32)`, StreamK, raster heuristic. It runs the official shape with 4 GPUs, `topk=1`, `dist=uniform`, `stable_index=true`, `warmup=20`, and `iters=100`.

## Optimization

The implementation adds targeted V2 hparams for the official MoE shape:

- Original: `tile_shape=(128,128,32)`
- Candidate: `tile_shape=(64,128,32)`
- Candidate: `tile_shape=(32,128,32)`
- Candidate: `tile_shape=(16,128,32)`

The final tuned candidate is `tile_shape=(32,128,32)`, `warp_shape=(32,64,32)`, StreamK, raster along N, with `sm_margin=32`. This reduces the consumer tile granularity from a tile that spans the full 64-row expert to a 32-row tile that aligns closer to the approximately 16-row source-rank chunks in the official uniform routing case.

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

Profiler overlap is calculated from Nsight Systems SQLite exports inside the `ISSUE6_MEASURED_LOOP` NVTX range. Communication activity is P2P memcpy on this PCIe host. Compute-side activity is reported two ways: grouped GEMM only, and grouped GEMM plus scatter/workspace prep.

## Correctness

The benchmark constructs a Torch reference path using Flux's test utilities for communication, scatter, and GEMM. BF16 comparisons use `atol=1e-2` and `rtol=1.5e-2`. Final optimized output passes allclose on all 4 ranks with no NaN or Inf.

## NCU Limitation

Nsight Compute was invoked against the final CUTLASS grouped GEMM kernel. The driver rejected performance-counter access on every GPU with `ERR_NVGPUCTRPERM`. Nsight Systems still records launch geometry, register count, shared memory, P2P copies, and kernel timing; counter-level metrics such as SM active and tensor utilization require enabling NVIDIA performance counters on the host.

## Rejected Ablation

A row-aligned two-way AG signal split was implemented and tested. It preserved correctness but increased P2P memcpy launch count and regressed official-shape latency to 0.346608 ms, so it is not part of the final patch.
