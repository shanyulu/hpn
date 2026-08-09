import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overlap_metrics import derived_overlap


def test_derived_overlap_formula():
    row = derived_overlap(compute_ms=2.0, comm_ms=1.0, fused_ms=2.2)
    assert round(row["derived_overlap_pct"], 6) == 80.0
