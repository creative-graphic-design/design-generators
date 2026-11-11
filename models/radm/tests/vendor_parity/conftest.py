"""Scope the pinned Detectron2 Pillow alias to vendor-parity imports."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


_DETECTRON2_TEST_MODULES = {
    "test_radm_300_lockstep",
    "test_radm_evaluation",
    "test_radm_parity",
    "test_s0_radm_topology",
    "test_s1_radm_training",
    "test_s4_radm_loader_stream",
}


@pytest.fixture(autouse=True, scope="module")
def legacy_pillow_compat_for_detectron2(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if (
        request.module.__name__.rsplit(".", maxsplit=1)[-1]
        not in _DETECTRON2_TEST_MODULES
    ):
        yield
        return

    from reference_adapter import _legacy_pillow_compat as legacy_pillow_compat

    with legacy_pillow_compat():
        yield
