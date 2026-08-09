# DeepEP V2 Source Map

This map is based on DeepEP commit `01dc3aaac82068020353dce2c302e38153c0bfaa`. It maps the Issue8 user-level modes to the current upstream implementation.

| Mode | ElasticBuffer configuration | Runtime/JIT arguments | Main combine kernel | Epilogue | Returned payload layout |
| --- | --- | --- | --- | --- | --- |
| A | `dispatch(..., do_expand=False)` | `use_expanded_layout=false`; `allow_multiple_reduction` is irrelevant to this direct branch | `combine_impl<..., false, ...>` uses the non-expanded `no_local_reduce` direct load/store branch | `combine_reduce_epilogue_impl<false, ...>` deduplicates destination rank/scale-rank slots | one logical returned token per source token and destination rank |
| B | `ElasticBuffer(..., allow_multiple_reduction=True)` and `dispatch(..., do_expand=True)` | `use_expanded_layout=true`, `allow_multiple_reduction=true` | `combine_impl<..., true, true, ...>` invokes `combine_reduce` for a mergeable multi-slot group | `combine_reduce_epilogue_impl<true, true, ...>` uses deduplicated rank/scale-rank slots | one logical returned token per mergeable rank/scale-rank group |
| C | `ElasticBuffer(..., allow_multiple_reduction=False)` and `dispatch(..., do_expand=True)` | `use_expanded_layout=true`, `allow_multiple_reduction=false`; `kDoExpandedSend=true` | `combine_impl<..., true, false, ...>` sends every valid top-k slot independently | `combine_reduce_epilogue_impl<true, false, ...>` sets `should_deduplicate=false` and reduces all valid top-k slots | one logical returned token per valid top-k replica |

## Call Chain

1. `deep_ep/buffers/elastic.py::ElasticBuffer.dispatch` exposes `do_expand` and returns an `EPHandle` containing `do_expand`, `topk_idx`, `recv_src_metadata`, and receive prefix sums.
2. `deep_ep/buffers/elastic.py::ElasticBuffer.combine` passes those handle fields to the C++ runtime.
3. `csrc/elastic/buffer.hpp::ElasticBuffer::combine` passes `use_expanded_layout` from the handle and its `allow_multiple_reduction` member to `launch_combine` and `launch_combine_reduce_epilogue`.
4. `csrc/kernels/elastic/combine.hpp::CombineRuntime::generate_impl` specializes `combine_impl` with both booleans. `CombineReduceEpilogueRuntime::generate_impl` specializes `combine_reduce_epilogue_impl` with the same pair.

## Reduction and Metadata Facts

- `deep_ep/include/deep_ep/impls/combine.cuh::combine_impl` defines `kDoExpandedSend = !kAllowMultipleReduction && kUseExpandedLayout` and reads `src_metadata` with stride `2 + kNumTopk`.
- Metadata columns after the first two hold expanded slot indices. The benchmark uses those slots to construct the upstream-compatible expanded combine input.
- `deep_ep/include/deep_ep/impls/combine_utils.cuh::combine_reduce` accumulates general BF16 reductions in `float2` before rounding back to BF16. It has a special BF16 add path for small no-bias reductions.
- `deep_ep/include/deep_ep/impls/dispatch.cuh::dispatch` maps experts to rank with `dst_expert_idx / kNumExpertsPerRank` and deduplicates destination ranks for non-expanded routing. Therefore rank collision, not just repeated expert id, is the relevant local-merge variable.
- `combine.cuh` selects direct symmetric remote write when `gin.is_nvlink_accessible`; otherwise it uses a local send buffer plus `gin.put`. Transport selection is a runtime property, so this repository never labels a logical byte count as NVLink/RDMA traffic.

## Weight Limitation

Current expanded-send code asserts that `topk_weights == nullptr`. The supplied real-path benchmark intentionally uses unit weights for all three modes. A weighted extension must pre-apply each slot's weight before Mode C combine and validate its FP32 reference separately.
