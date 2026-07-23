import subprocess
import sys
import textwrap


def _assert_probe_rejected(code: str, expected: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected in result.stderr


def _assert_probe_passes(code: str) -> None:
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


def test_modeling_hook_liveness_rejects_bbox_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutganpp.modeling_layoutganpp"], "beartype.beartype"):
            from layoutganpp.modeling_layoutganpp import LayoutGANPPModelOutput

        LayoutGANPPModelOutput(bbox=torch.zeros(1, 2, 4, dtype=torch.long))
        """,
        "LayoutGANPPModelOutput.__init__",
    )


def test_modeling_hook_liveness_rejects_bbox_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutganpp.modeling_layoutganpp"], "beartype.beartype"):
            from layoutganpp.modeling_layoutganpp import LayoutGANPPModelOutput

        LayoutGANPPModelOutput(bbox=torch.zeros(1, 2))
        """,
        "LayoutGANPPModelOutput.__init__",
    )


def test_modeling_hook_liveness_rejects_latent_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutganpp.modeling_layoutganpp"], "beartype.beartype"):
            from layoutganpp import LayoutGANPPConfig, LayoutGANPPModel

        model = LayoutGANPPModel(LayoutGANPPConfig(num_labels=3, latent_size=4, d_model=8, nhead=2, num_layers=1))
        model(latents=torch.zeros(1, 3, 4), labels=torch.zeros(1, 2, dtype=torch.long))
        """,
        "LayoutGANPPModel.forward",
    )


def test_modeling_without_hook_accepts_output_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layoutganpp.modeling_layoutganpp import LayoutGANPPModelOutput

        assert LayoutGANPPModelOutput(bbox=torch.zeros(1, 2, 4, dtype=torch.long)).bbox.shape == (1, 2, 4)
        assert LayoutGANPPModelOutput(bbox=torch.zeros(1, 2)).bbox.shape == (1, 2)
        """
    )
