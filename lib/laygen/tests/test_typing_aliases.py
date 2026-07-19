import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from laygen.common.testing import assert_layout_output_schema
from laygen.common.typing import (
    NumpyLayoutBBoxes,
    NumpyLayoutLabels,
    NumpyLayoutMask,
    TorchLayoutBBoxes,
    TorchLayoutLabels,
    TorchLayoutMask,
)
from laygen.modeling_outputs import LayoutGenerationOutput


def test_torch_aliases_accept_torch_values():
    def accept_layout(
        bbox: TorchLayoutBBoxes,
        labels: TorchLayoutLabels,
        mask: TorchLayoutMask,
    ) -> tuple[TorchLayoutBBoxes, TorchLayoutLabels, TorchLayoutMask]:
        return bbox, labels, mask

    torch_values = accept_layout(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert torch_values[0].shape == (1, 2, 4)


def test_numpy_aliases_accept_numpy_values():
    bbox: NumpyLayoutBBoxes = np.zeros((1, 1, 4), dtype=np.float32)
    labels: NumpyLayoutLabels = np.zeros((1, 1), dtype=np.int64)
    mask: NumpyLayoutMask = np.ones((1, 1), dtype=bool)

    output = LayoutGenerationOutput(
        bbox=bbox,
        labels=labels,
        mask=mask,
        id2label={0: "text"},
    )

    assert output.to_tuple()[0].shape == (1, 1, 4)


def test_modeling_output_shape_validation_is_schema_assertion_responsibility():
    output = LayoutGenerationOutput(
        bbox=np.zeros((1, 2, 5), dtype=np.float32),
        labels=np.zeros((1, 2), dtype=np.int64),
        mask=np.ones((1, 2), dtype=bool),
        id2label={0: "text"},
    )

    assert output["bbox"].shape == (1, 2, 5)
    with pytest.raises(AssertionError):
        assert_layout_output_schema(output)


def test_typing_aliases_and_output_import_without_torch():
    code = textwrap.dedent(
        """
        import builtins
        import importlib.util
        import os
        import sys

        import numpy as np

        os.environ["USE_TORCH"] = "0"
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__

        def find_spec_without_torch(name, package=None):
            if name == "torch" or name.startswith("torch."):
                return None
            return original_find_spec(name, package)

        def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("blocked torch import")
            return original_import(name, globals, locals, fromlist, level)

        importlib.util.find_spec = find_spec_without_torch
        builtins.__import__ = import_without_torch

        from laygen.common.typing import NumpyLayoutBBoxes
        from laygen.modeling_outputs import LayoutGenerationOutput

        bbox: NumpyLayoutBBoxes = np.zeros((1, 1, 4), dtype=np.float32)
        numpy_bbox: NumpyLayoutBBoxes = bbox
        output = LayoutGenerationOutput(
            bbox=numpy_bbox,
            labels=np.zeros((1, 1), dtype=np.int64),
            mask=np.ones((1, 1), dtype=bool),
            id2label={0: "text"},
        )
        assert output["bbox"].shape == (1, 1, 4)
        assert output.to_tuple()[0].shape == (1, 1, 4)
        assert "torch" not in sys.modules
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)
