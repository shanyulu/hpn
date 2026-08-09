# Methodology

## Same Input

Each case records `case_id`, seed, tokens, hidden size, top-k, experts, world
size, and duplicate bucket. `routing.build_case` creates one activation,
`topk_idx`, and weight tensor. A/B/C reuse that case; only the source-mapped
layout and reduction flags change.

## Duplicate Metric

`rank_duplicate_ratio = 1 - mean_t(|unique(topk_idx[t] // experts_per_rank)| / K)`.

This measures destination-rank collisions that Mode B can merge. Expert-ID
duplication is retained as a separate diagnostic and is not treated as a local
reduction opportunity.

## Evidence Classes

| Class | Meaning in this submission |
| --- | --- |
| Measured | One-rank `ElasticBuffer` execution and the committed precision summary. |
| Analytical | L1 logical return payload bytes: payload units times hidden size and dtype bytes. |
| Source-derived | A/B/C trigger and reduction semantics in `docs/source_map.md`. |
| Blocked | Multi-rank V2 completion time on the tested host; see `docs/feasibility.md`. |

L1 excludes protocol overhead, fabric counters, and bandwidth. L2/L3 values are
not emitted without runtime metadata or hardware-counter evidence.

## Completion Time

On a host where multi-rank combine completes, `benchmark.py` warms the path,
barriers ranks, records CUDA Events, synchronizes the end Event, gathers all
rank durations, and retains the per-iteration maximum as the critical path.
Only a record with `GPU_event_max_rank_completion` and world size greater than
one can produce a measured completion-time recommendation. One-rank event data
is never promoted to multi-rank communication time.

The tested 4 x H800 PCIe host has no such record. Its completion time is `N/A`,
not zero or a model estimate.

## Precision

The benchmark forms a deterministic FP32 full-return reduction after BF16 input
quantization. It reports max/mean absolute error, RMSE, relative L2, guarded
maximum relative error, mismatch count, NaNs, and Infs. That reference isolates
combine/reduction rounding and does not claim to include BF16 input
quantization error.
