# Decision Table

The tested host has no measured multi-rank DeepEP V2 combine completion time.
No completion-time threshold or latency advantage is inferred below.

| Evidence | Duplicate bucket | Message bucket | Precision requirement | Recommended mode | Basis |
| --- | --- | --- | --- | --- | --- |
| source_derived_and_L1_analytical | zero | any | no host-validated latency threshold | A | direct non-expanded source path; this is not a measured latency recommendation |
| source_derived_and_L1_analytical | nonzero; expanded layout required | any | logical return-payload priority | B | merges destination-rank collisions before return; no multi-rank latency validation |
| source_derived | any | any | independent replica return semantics required | C | all valid top-k slots reach the epilogue; no host-validated precision or latency ranking |
