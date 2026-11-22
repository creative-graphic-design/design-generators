# BBox property-based regression pilot

This minimal pilot for [Issue #560](https://github.com/creative-graphic-design/design-generators/issues/560) exercises **the actual Python/PyTorch implementation** in `lib/laygen/src/laygen/common/bbox.py`. It intentionally does not introduce Lean or claim formal correctness.

## Verification and bounds

- Hypothesis generates valid, normalized, finite `xywh` and `ltwh` boxes (1–8 boxes per case; 60 cases per property) and checks float64 round trips within `atol=1e-12`.
- Property checks run against the real imported `laygen.common.bbox` functions, not duplicated example implementations.
- The mutation smoke check temporarily changes the real `xywh_to_ltrb` operation from `x - w / 2` to `x - w / 3` in the ephemeral checkout. It requires the property to **fail with a Hypothesis counterexample** and verifies restored source passes afterward.
- CI runs on this pilot's files and **every PR changing `bbox.py`**, and the pilot jobs do not alter existing required checks.

Clamping, quantization, NaN/Inf inputs, invalid canvas dimensions, and GPU behavior are out of scope. The tested finite inputs do **not** imply universal correctness.

## Reproduce

From the repository root:

```bash
uv run --locked --package laygen --extra torch --with hypothesis==6.131.0 --with pytest==8.4.2 \
  python -m pytest verification/bbox/test_bbox_properties.py -q

uv run --locked --package laygen --extra torch --with hypothesis==6.131.0 --with pytest==8.4.2 \
  python verification/bbox/check_actual_mutation.py
```

The mutation is temporary; the script uses `finally` to restore the original `bbox.py` contents. The temporary change is **never committed**.

## Tool-selection decision

The earlier [Lean + Hypothesis pilot PR #561](https://github.com/creative-graphic-design/design-generators/pull/561) proved three properties of a separate integer-coordinate algebra model. Those Lean theorems had **no machine-checked link to the Python code**; they stayed green while an actual Python mutation was caught by Hypothesis. This experiment provides positive defect-detection evidence for Hypothesis but **no demonstrated incremental defect detection from Lean**. Lean remains a future candidate for a specific unbounded, safety-relevant property if an explicit implementation-conformance mechanism is established.

The original paper / vendor numerical parity and staged S0–S5 training reproduction are separate acceptance requirements, not replaced by this test.
