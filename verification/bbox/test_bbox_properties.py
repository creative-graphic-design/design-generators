"""Bounded floating-point checks against the actual laygen bbox helpers."""

import torch
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import DrawFn
from laygen.common.bbox import (
    ltrb_to_xywh,
    ltwh_to_xywh,
    xywh_to_ltrb,
    xywh_to_ltwh,
)


Box4 = tuple[float, float, float, float]


@st.composite
def valid_xywh(draw: DrawFn) -> Box4:
    """Generate a nonempty normalized box contained within the unit canvas."""
    x = draw(st.floats(min_value=0.25, max_value=0.75, allow_nan=False))
    y = draw(st.floats(min_value=0.25, max_value=0.75, allow_nan=False))
    w = draw(
        st.floats(min_value=0.0, max_value=2 * min(x, 1 - x), allow_nan=False)
    )
    h = draw(
        st.floats(min_value=0.0, max_value=2 * min(y, 1 - y), allow_nan=False)
    )
    return (x, y, w, h)


@st.composite
def valid_ltwh(draw: DrawFn) -> Box4:
    """Generate a left/top box with edges inside the unit canvas."""
    left = draw(st.floats(min_value=0.0, max_value=0.8, allow_nan=False))
    top = draw(st.floats(min_value=0.0, max_value=0.8, allow_nan=False))
    w = draw(st.floats(min_value=0.0, max_value=1.0 - left, allow_nan=False))
    h = draw(st.floats(min_value=0.0, max_value=1.0 - top, allow_nan=False))
    return (left, top, w, h)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(st.lists(valid_xywh(), min_size=1, max_size=8))
def test_xywh_ltrb_roundtrip(boxes: list[Box4]) -> None:
    """Round trips should agree up to float64 operation rounding."""
    tensor = torch.tensor(boxes, dtype=torch.float64)
    recovered = ltrb_to_xywh(xywh_to_ltrb(tensor))
    assert recovered.shape == tensor.shape
    torch.testing.assert_close(recovered, tensor, rtol=0, atol=1e-12)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(st.lists(valid_ltwh(), min_size=1, max_size=8))
def test_ltwh_xywh_roundtrip(boxes: list[Box4]) -> None:
    """The independent left/top conversion should have the same property."""
    tensor = torch.tensor(boxes, dtype=torch.float64)
    recovered = xywh_to_ltwh(ltwh_to_xywh(tensor))
    assert recovered.shape == tensor.shape
    torch.testing.assert_close(recovered, tensor, rtol=0, atol=1e-12)


def test_deliberately_wrong_half_width_is_detected() -> None:
    """Negative control: a /3 instead of /2 offset does not preserve boxes."""
    tensor = torch.tensor([[0.5, 0.5, 0.2, 0.4]], dtype=torch.float64)
    x, y, w, h = tensor.unbind(-1)
    mutant = torch.stack((x - w / 3, y - h / 2, x + w / 2, y + h / 2), -1)
    assert not torch.allclose(ltrb_to_xywh(mutant), tensor)
