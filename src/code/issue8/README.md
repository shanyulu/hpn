# Issue8: DeepEP Combine Reduction Trade-offs

## What

This submission maps and compares three DeepEP V2 combine configurations.

| Mode | Configuration | Source-derived reduction path |
| --- | --- | --- |
| A | `do_expand=False` | direct return path, then standard epilogue |
| B | `do_expand=True`, `allow_multiple_reduction=True` | mergeable replicas reduce in the combine kernel |
| C | `do_expand=True`, `allow_multiple_reduction=False` | replicas return independently; epilogue reduces all slots |

The source proof is in [docs/source_map.md](docs/source_map.md).

## Evidence

### Measured

- DeepEP main `01dc3aaac82068020353dce2c302e38153c0bfaa` built against and loaded NCCL 2.30.4.
- A, B, and C each executed through the one-rank `ElasticBuffer.dispatch` /
  `ElasticBuffer.combine` paths.
- [results/one_rank_execution.json](results/one_rank_execution.json) records
  the same-input executed BF16 precision summary: all three paths had
  `max_abs_error=0.00390625`, `relative_l2_error=0.001704840688034892`, and no
  NaN or Inf for the recorded case. One-rank execution is not presented as
  multi-rank communication evidence.

### Analytical

- `results/summary.csv` and `results/summary.json` contain 45 deterministic
  same-input cases with L1 logical return-payload accounting only.
- L1 is source-derived logical payload bytes. It excludes protocol overhead and
  is not a hardware traffic counter or bandwidth measurement.
- No analytical or theoretical latency is emitted by this submission.

### Unavailable

Multi-rank DeepEP V2 combine completion latency is reported as `N/A` on the
tested 4 x H800 PCIe host. The environment was audited against Gin, the
`EP_DISABLE_GIN=1` fallback, and the upstream `hybrid-ep` PCIe path. None
provided a semantically valid executable multi-rank A/B/C combine path. No
modeled latency is substituted for measured latency.

[docs/feasibility.md](docs/feasibility.md) contains the concise 2-rank/4-rank
blocker evidence and provenance.

## Method

`routing.build_case` makes one fixed-seed activation/routing case. All A/B/C
runs reuse it; only DeepEP layout and reduction flags change. The decision
variable is destination-rank duplication, not merely repeated expert IDs:

`rank_duplicate_ratio = 1 - mean_t(|unique(topk_idx[t] // experts_per_rank)| / K)`.

When a valid multi-rank path completes, `benchmark.py` uses CUDA Events after
warmup and records the maximum rank completion duration per iteration only after
all ranks complete. It intentionally does not fall back to a Python or
closed-form latency estimate. The expanded-send test uses unit top-k weights
because current upstream Mode C asserts null `topk_weights`; see the source map
for the boundary.

Full methodology: [docs/methodology.md](docs/methodology.md).

## Results And Decision

[results/decision_table.md](results/decision_table.md) separates source-derived
payload guidance from completion-time selection. It has no multi-rank latency
threshold on this host. In particular, no recommendation in this checkout
claims a measured latency advantage. Message-size and precision-requirement
crossover thresholds are `unresolved`: neither L1 payload bytes nor one-rank
precision is used to infer them.

## Reproduce

From this directory, with a matching upstream DeepEP installation available to
Python:

```bash
python run_sweep.py --analytical-only --output-dir results
python tests/test_issue8.py
```

On a different host where all three modes complete across ranks, use three
independent native DeepEP process launches per mode, then merge their raw
critical-path samples:

```bash
for run in 1 2 3; do
  python benchmark.py --num-processes 4 --mode A --output results/a-${run}.json
  python benchmark.py --num-processes 4 --mode B --output results/b-${run}.json
  python benchmark.py --num-processes 4 --mode C --output results/c-${run}.json
done
python merge_measurements.py --inputs results/a-*.json results/b-*.json results/c-*.json \
  --output results/measured_summary.csv
```

Only records with `GPU_event_max_rank_completion` evidence are eligible for a
measured completion-time recommendation.

## Provenance

| Item | Value |
| --- | --- |
| hpn base | `fcb16fe5a2942da749543b0e206707c7f3faba54` |
| DeepEP | `01dc3aaac82068020353dce2c302e38153c0bfaa` |
| GPU/topology | 4 x NVIDIA H800 PCIe; no NVLink |
| NVIDIA driver | 595.71.05 |
| CUDA toolkit / PyTorch CUDA | 12.8.93 / 12.8 |
| PyTorch | 2.8.0+cu128 |
| DeepEP NCCL compile/runtime | 2.30.4 / 2.30.4 |

## Limits

- No L3 hardware traffic counter, PCIe bandwidth, NVLink bandwidth, or RDMA
  bandwidth is claimed.
- No 2-rank or 4-rank combine latency is claimed for this host.
- Weighted Mode C needs pre-weighted slots and a separately validated FP32
  reference.
