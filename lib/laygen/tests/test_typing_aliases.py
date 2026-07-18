import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from laygen.common.testing import install_jaxtyping_runtime_hook
from laygen.common.typing import (
    LayoutArray,
    LayoutBBoxes,
    LayoutLabels,
    LayoutMask,
    NumpyArray,
    NumpyLayoutBBoxes,
    NumpyLayoutLabels,
    NumpyLayoutMask,
)
from laygen.modeling_outputs import LayoutGenerationOutput


def test_shared_aliases_accept_numpy_and_torch_values():
    def accept_layout(
        bbox: LayoutBBoxes,
        labels: LayoutLabels,
        mask: LayoutMask,
    ) -> tuple[LayoutBBoxes, LayoutLabels, LayoutMask]:
        return bbox, labels, mask

    numpy_values = accept_layout(
        np.zeros((1, 2, 4), dtype=np.float32),
        np.zeros((1, 2), dtype=np.int64),
        np.ones((1, 2), dtype=bool),
    )
    torch_values = accept_layout(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert numpy_values[0].shape == (1, 2, 4)
    assert torch_values[0].shape == (1, 2, 4)


def test_numpy_aliases_accept_numpy_values():
    array: NumpyArray = np.zeros((1,), dtype=np.float32)
    bbox: NumpyLayoutBBoxes = np.zeros((1, 1, 4), dtype=np.float32)
    labels: NumpyLayoutLabels = np.zeros((1, 1), dtype=np.int64)
    mask: NumpyLayoutMask = np.ones((1, 1), dtype=bool)
    layout_array: LayoutArray = array

    output = LayoutGenerationOutput(
        bbox=bbox,
        labels=labels,
        mask=mask,
        id2label={0: "text"},
    )

    assert layout_array.shape == (1,)
    assert output.to_tuple()[0].shape == (1, 1, 4)


def test_torch_typing_aliases_import_with_torch():
    from laygen.common.torch_typing import (
        TorchLayoutBBoxes,
        TorchLayoutLabels,
        TorchLayoutMask,
        TorchLogOneHot,
        TorchPayload,
        TorchTensor,
        TorchTokenIds,
        TorchTokenLogits,
    )

    payload: TorchPayload = None
    tensor: TorchTensor = torch.zeros(1)
    bbox: TorchLayoutBBoxes = torch.zeros(1, 2, 4)
    labels: TorchLayoutLabels = torch.zeros(1, 2, dtype=torch.long)
    mask: TorchLayoutMask = torch.ones(1, 2, dtype=torch.bool)
    token_ids: TorchTokenIds = torch.zeros(1, 3, dtype=torch.long)
    token_logits: TorchTokenLogits = torch.zeros(1, 3, 5)
    log_onehot: TorchLogOneHot = torch.zeros(1, 5, 3)

    assert payload is None
    assert tensor.shape == (1,)
    assert bbox.shape == (1, 2, 4)
    assert labels.shape == mask.shape == (1, 2)
    assert token_ids.shape == (1, 3)
    assert token_logits.shape == (1, 3, 5)
    assert log_onehot.shape == (1, 5, 3)


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

        from laygen.common.typing import LayoutBBoxes, NumpyLayoutBBoxes
        from laygen.modeling_outputs import LayoutGenerationOutput

        bbox: LayoutBBoxes = np.zeros((1, 1, 4), dtype=np.float32)
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


def test_runtime_hook_rejects_wrong_shape_in_probe(tmp_path, monkeypatch):
    module_path = tmp_path / "shape_probe.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from laygen.common.typing import LayoutBBoxes

            def accept_bbox(bbox: LayoutBBoxes) -> LayoutBBoxes:
                return bbox
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with install_jaxtyping_runtime_hook(["shape_probe"]):
        from shape_probe import accept_bbox

    assert accept_bbox(np.zeros((1, 2, 4), dtype=np.float32)).shape == (1, 2, 4)
    assert accept_bbox(torch.zeros(1, 2, 4)).shape == (1, 2, 4)
    with pytest.raises(Exception, match="accept_bbox"):
        accept_bbox(np.zeros((1, 2, 5), dtype=np.float32))


def test_runtime_hook_rejects_wrong_shape_in_modeling_output():
    code = textwrap.dedent(
        """
        import numpy as np

        from laygen.common.testing import install_jaxtyping_runtime_hook

        with install_jaxtyping_runtime_hook(["laygen.modeling_outputs"]):
            from laygen.modeling_outputs import LayoutGenerationOutput

        LayoutGenerationOutput(
            bbox=np.zeros((1, 2, 5), dtype=np.float32),
            labels=np.zeros((1, 2), dtype=np.int64),
            mask=np.ones((1, 2), dtype=bool),
            id2label={0: "text"},
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env={**os.environ, "USE_TORCH": "0"},
        text=True,
    )

    assert result.returncode != 0
    assert "LayoutGenerationOutput.__init__" in result.stderr
