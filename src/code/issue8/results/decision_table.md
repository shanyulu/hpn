# Decision Table

Only rows with `GPU_event_max_rank_completion` latency are allowed to produce a recommendation.
Logical payload estimates are never promoted to measured latency.

| Duplicate bucket | Message bucket | Precision requirement | Recommended mode | Basis |
| --- | --- | --- | --- | --- |
| pending | pending | pending | pending | No supported multi-rank measurement is present. |
