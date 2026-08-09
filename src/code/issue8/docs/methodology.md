# Methodology

## Same Input Contract

Each case is identified by `case_id`, seed, token count, hidden size, top-k, expert count, world size, and duplicate bucket. `routing.build_case` creates one activation tensor, one `topk_idx`, and one weight tensor. All A/B/C runs consume the same routing case; only the DeepEP layout and reduction flags vary.

## Duplicate Definitions

The decision variable is destination-rank duplication:

`rank_duplicate_ratio = 1 - mean_t(|unique(topk_idx[t] // experts_per_rank)| / K)`.

It is zero when every top-k selection of a token targets a different rank. It is `(K-1)/K` when they all target one rank. `expert_duplicate_ratio` is also recorded, but is not used to claim local-reduction savings because different experts can still share one destination rank.

## Traffic Evidence

- L1 is `logical_payload_tokens * hidden * dtype_bytes`, derived from the source-mapped return layout. It excludes protocol and fabric overhead.
- L2 is reserved for counts obtained from a real DeepEP metadata/runtime observation. The helper is present, but no L2 value is emitted until a run exposes the required count.
- L3 is hardware-counter evidence from Nsight/NCCL/fabric counters. It is never implied by L1 or L2.

## Timing

The real benchmark warms the exact combine path, barriers ranks before every sample, surrounds the asynchronous call with CUDA Events, synchronizes on the end Event, gathers all rank durations, and retains the maximum duration as the distributed critical path. It reports median, p95, min, max, and population standard deviation. Run A, B, and C in three separate `torchrun` launches.

## Precision

The benchmark uses DeepEP's deterministic generated BF16 source values and forms a deterministic FP32 full-return reduction after that input quantization. It reports absolute, mean absolute, RMSE, relative L2, guarded maximum relative error, mismatches, NaNs, and Infs. This isolates combine/reduction rounding; it does not claim to measure BF16 input quantization error.

## Current Environment Boundary

The recorded development host has four H800 PCIe GPUs with P2P access but no NVLink edges. Its current DeepEP V2 multi-rank initialization reports NCCL Gin unavailable; disabling Gin hangs. Therefore no repository result calls its logical bytes or its one-rank validation a four-rank measured result. Use a supported NCCL Gin topology for the real benchmark commands.
