# Multi-GPU Feasibility

## Tested Host

| Item | Value |
| --- | --- |
| GPU | 4 x NVIDIA H800 PCIe, SM90 |
| Topology | GPU0-GPU1 and GPU2-GPU3 PIX; cross-pair NODE; no NVLink |
| NVIDIA driver | 595.71.05 |
| CUDA toolkit | 12.8.93 |
| PyTorch | 2.8.0+cu128, CUDA 12.8 |
| DeepEP main | `01dc3aaac82068020353dce2c302e38153c0bfaa` |
| DeepEP NCCL compile/runtime | 2.30.4 / 2.30.4 |
| hpn base | `fcb16fe5a2942da749543b0e206707c7f3faba54` |

DeepEP debug output and `ncclGetVersion` both reported NCCL 2.30.4. PyTorch's
own NCCL build-version report is older, but DeepEP was rebuilt against and
loaded the same 2.30.4 pip NCCL library.

## Results

| Path | 2-rank | 4-rank | Reached combine? | Result |
| --- | --- | --- | --- | --- |
| Main + Gin | `nccl.cu:87`: `NCCL GIN is unavailable` | same | No | `ElasticBuffer` construction rejects unavailable Gin before dispatch/combine. |
| Main + `EP_DISABLE_GIN=1` | process group/domain query pass; `ElasticBuffer` construction times out | same | No | All ranks reach the constructor but never return from it; dispatch, combine, and JIT are not reached. |
| `hybrid-ep` PCIe | not executable | not executable | No | CUDA 12.8 build stops in `csrc/kernels/internode_ll.cu:658`: `cudaLaunchAttributeNvlinkUtilCentricScheduling` is unavailable. Its legacy `Buffer` PCIe path also does not preserve V2 Issue8 A/B/C semantics. |

The main non-Gin probes used `EP_DISABLE_GIN=1`, `EP_BUFFER_DEBUG=1`, native
DeepEP multiprocessing, and hard 90 s (2-rank) / 120 s (4-rank) limits. They
completed NCCL process-group setup and reported physical/logical domains before
stalling in `ElasticBuffer` construction.

`hybrid-ep` was inspected in a separate upstream checkout. Its PCIe path is
`Buffer(..., allow_nvlink_for_normal_mode=False)` to `dispatch_pcie` /
`combine_pcie`; it has no `use_expanded_layout`, `allow_multiple_reduction`, or
V2 `combine_reduce_epilogue` A/B/C trigger pair.

## Consequence

Multi-rank DeepEP V2 combine completion latency is `N/A` on this tested host.
Gin, the documented `EP_DISABLE_GIN=1` fallback, and the upstream `hybrid-ep`
PCIe path were audited. None provided a semantically valid executable
multi-rank A/B/C combine path. No modeled latency is substituted for measured
latency.

This finding does not claim that DeepEP V2 is unsupported on every PCIe system;
it describes the tested topology, toolchain, and upstream commits above.
