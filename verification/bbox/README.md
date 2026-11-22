# Bounding-box verification pilot

This is a small, isolated proof-and-test experiment for [Issue #560](https://github.com/creative-graphic-design/design-generators/issues/560). It does not change production `laygen` behavior, workspace dependencies, existing CI requirements, or merge/publication policy.

## What is verified

| Level | Subject | Claim |
| --- | --- | --- |
| Lean 4 | `BBoxContracts.lean` using exact `Int` arithmetic | Converting integer-centered `xywh` into doubled-unit `ltrb` and back is an identity; doubled edge differences reproduce twice the width and height. |
| Hypothesis + PyTorch | `laygen.common.bbox` | For bounded valid normalized boxes, actual float64 implementations of `xywh_to_ltrb` / `ltrb_to_xywh` and `ltwh_to_xywh` / `xywh_to_ltwh` round-trip within `atol=1e-12`. |
| Negative control | Deliberately incorrect /3 half-width formula | A known incorrect formula is rejected on a concrete counterexample. |

The Lean contract maps `xywhToLTRB2` to `xywh_to_ltrb` (with integer inputs and doubled output coordinates), and `ltrb2ToXYWH` to `ltrb_to_xywh`. The Hypothesis test independently exercises the Python functions. **A Lean proof of integer algebra is not a proof of the floating-point PyTorch implementation.** Clamping, quantization, invalid canvases, NaN/Inf, overflow and GPU determinism are deliberately out of scope.

## Commands

From the repository root:

```bash
uv run --locked --package laygen --extra torch --with hypothesis==6.131.0 --with pytest==8.4.2 \
  python -m pytest verification/bbox/test_bbox_properties.py -q
```

For Lean (Lean v4.32.1 and its included Lake are installed via [elan](https://github.com/leanprover/elan)):

```bash
cd verification/bbox
lake build
```

A dedicated GitHub Actions workflow runs both checks on this pilot's changes, without adding any mandatory gate to the existing main CI. The Hypothesis suite uses a fixed execution profile (60 examples per property, `derandomize=True`) and does not require GPU or vendor assets. Full model and training reproducibility still require the existing vendor-parity and S0–S5 protocols.

## Incremental value / actual-code mutation experiment

Run the baseline Hypothesis property, temporarily change **the actual** `xywh_to_ltrb` implementation from `x - w / 2` to `x - w / 3`, rerun the same property in a fresh interpreter, and restore the file. The experiment **passes only if** the original implementation passes, the mutant produces a Hypothesis counterexample, and the restored implementation passes again.

```bash
uv run --locked --package laygen --extra torch --with hypothesis==6.131.0 --with pytest==8.4.2 \\
  python verification/bbox/check_actual_mutation.py
```

Lean's separately authored `BBoxContracts.lean` is *not* derived from that Python function; its compilation does not observe the mutation. Consequently, a passing Lean proof adds **no demonstrated protection against Python code drift** in this pilot. Do not require Lean in production CI based on these results alone. Review tool choice and proof-to-implementation linkage before making a merge decision; see #560.
