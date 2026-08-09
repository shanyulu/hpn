# Issue8: DeepEP Combine Reduction Trade-offs

## What

This submission compares the three real DeepEP V2 combine configurations:

| Mode | Configuration | Reduction location |
| --- | --- | --- |
| A | `do_expand=False` | direct return followed by the standard epilogue |
| B | `do_expand=True`, `allow_multiple_reduction=True` | mergeable replicas reduced in the combine kernel |
| C | `do_expand=True`, `allow_multiple_reduction=False` | each replica returns independently; epilogue reduces all slots |

The exact source-level mapping is in [docs/source_map.md](docs/source_map.md).

## Why

Mode B can shrink the return payload when multiple top-k selections target the
same destination rank, but adds a local floating-point reduction. Mode C keeps
those replicas separate until the epilogue, changing both traffic and rounding.
The relevant routing variable is destination-rank duplication, not merely
duplicate expert ids.

## Current Evidence Status

The committed `results/summary.csv` contains 45 deterministic same-input cases
and **L1 analytical logical payload bytes only**. It contains no measured
latency or hardware traffic claim.

Development validation built DeepEP commit
`01dc3aaac82068020353dce2c302e38153c0bfaa` and passed its one-rank elastic
smoke test. The available four-GPU H800 PCIe host has P2P enabled but no NVLink
edges. DeepEP V2 multi-rank initialization fails because NCCL Gin is
unavailable; disabling Gin then hangs. Consequently this checkout cannot
honestly provide four-rank DeepEP completion timings. The real benchmark writes
`status: blocked` rather than a latency number on that failure.

This is an environmental blocker, not an estimate. A supported Hopper/NVLink or
properly configured NCCL Gin environment is required to fill the measured rows.

## Measurement Method

`benchmark.py` calls `ElasticBuffer.dispatch` and `ElasticBuffer.combine`; it
does not model the kernel in Python. It uses a fixed seed and reuses a single
routing case per A/B/C comparison. The benchmark uses unit top-k weights because
current upstream expanded-send asserts `topk_weights == nullptr`; see the source
map for the required weighted extension.

For a successful multi-rank run it records:

- L1 logical return payload bytes, separately labeled from transport counters.
- CUDA Event completion duration after warmup, using the maximum duration across
  ranks as each iteration's distributed critical path; median, p95, min, max,
  and standard deviation are reported.
- Error against a deterministic FP32 full-return reduction after BF16 input
  quantization: max/mean absolute error, RMSE, relative L2, guarded maximum
  relative error, mismatch count, NaNs, and Infs.

Details and limitations are in [docs/methodology.md](docs/methodology.md).

## Reproduce

The scripts have no checkout-specific paths. Install/build the upstream DeepEP
matching the source map, make it importable, then run from this directory.

```bash
python run_sweep.py --analytical-only --output-dir results
python tests/test_issue8.py
```

On a supported four-rank DeepEP V2 topology, run each mode as an independent
process launch. DeepEP's upstream `init_dist` owns its multiprocessing contract,
so invoke this script directly rather than through `torchrun`. Repeat every
command at least three times with distinct output files before merging the
results into a decision table.

```bash
for run in 1 2 3; do
  python benchmark.py --num-processes 4 \
    --mode A --output results/measured-a-run${run}.json
  python benchmark.py --num-processes 4 \
    --mode B --output results/measured-b-run${run}.json
  python benchmark.py --num-processes 4 \
    --mode C --output results/measured-c-run${run}.json
done
python merge_measurements.py --inputs \
  results/measured-a-run1.json results/measured-a-run2.json results/measured-a-run3.json \
  results/measured-b-run1.json results/measured-b-run2.json results/measured-b-run3.json \
  results/measured-c-run1.json results/measured-c-run2.json results/measured-c-run3.json \
  --output results/measured_summary.csv
```

`merge_measurements.py` requires the raw per-iteration critical-path samples
from each independent launch, recomputes aggregate statistics, and regenerates
the decision table. It only makes recommendations among modes that satisfy the
recorded maximum-absolute-error limits.

## Decision Table

[results/decision_table.md](results/decision_table.md) intentionally remains
pending until a supported multi-rank sweep writes actual critical-path samples.
`report.py` rejects non-measured rows as a source of recommendations, so it
cannot manufacture duplicate-rate or message-size thresholds from byte formulas.

## Environment Recorded During Development

| Component | Value |
| --- | --- |
| GPU | 4 x NVIDIA H800 PCIe, SM90 |
| GPU topology | GPU0-GPU1 and GPU2-GPU3 PIX; cross-pair NODE; no NVLink |
| Driver | 595.71.05 |
| CUDA toolkit / PyTorch CUDA | 12.8 / 12.8 |
| PyTorch | 2.8.0+cu128 |
| DeepEP | `01dc3aaac82068020353dce2c302e38153c0bfaa` |
| NCCL seen by PyTorch | 2.27.3 |
| Profiler | Nsight Systems and Nsight Compute unavailable |

The installed external NCCL package was upgraded to 2.30.4 to build DeepEP, but
the PyTorch runtime still reports NCCL 2.27.3. This is another reason the host
is not presented as a validated DeepEP V2 multi-rank target.

## Limitations

- No L3 hardware counter claim is made without Nsight/NCCL/fabric evidence.
- No PCIe result is labeled NVLink or RDMA.
- The committed initial matrix is not a latency benchmark and cannot choose a
  mode by itself.
- The current implementation covers the unweighted expanded-send path required
  by upstream's current assertion. Weighted Mode C needs pre-weighted slot data
  and a dedicated FP32 validation.
