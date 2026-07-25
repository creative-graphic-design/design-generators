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


def test_transformer_hook_liveness_rejects_logits_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["layoutdiffusion.modeling_layoutdiffusion"], "beartype.beartype"
        ):
            from layoutdiffusion.modeling_layoutdiffusion import (
                LayoutDiffusionTransformerOutput,
            )

        LayoutDiffusionTransformerOutput(logits=torch.zeros(1, 2, 3, dtype=torch.long))
        """,
        "LayoutDiffusionTransformerOutput.__init__",
    )


def test_transformer_hook_liveness_rejects_logits_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["layoutdiffusion.modeling_layoutdiffusion"], "beartype.beartype"
        ):
            from layoutdiffusion.modeling_layoutdiffusion import (
                LayoutDiffusionTransformerOutput,
            )

        LayoutDiffusionTransformerOutput(logits=torch.zeros(1, 2))
        """,
        "LayoutDiffusionTransformerOutput.__init__",
    )


def test_transformer_hook_liveness_rejects_condition_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["layoutdiffusion.modeling_layoutdiffusion"], "beartype.beartype"
        ):
            from layoutdiffusion import LayoutDiffusionTransformer

        model = LayoutDiffusionTransformer(vocab_size=16, hidden_size=32, num_channels=8, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)
        model(torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, dtype=torch.long), condition_ids=torch.zeros(1, 6, dtype=torch.long))
        """,
        "LayoutDiffusionTransformer.forward",
    )


def test_transformer_without_hook_accepts_output_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layoutdiffusion.modeling_layoutdiffusion import LayoutDiffusionTransformerOutput

        assert LayoutDiffusionTransformerOutput(logits=torch.zeros(1, 2, 3, dtype=torch.long)).logits.shape == (1, 2, 3)
        assert LayoutDiffusionTransformerOutput(logits=torch.zeros(1, 2)).logits.shape == (1, 2)
        """
    )
