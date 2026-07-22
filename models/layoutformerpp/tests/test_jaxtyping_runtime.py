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


def test_modeling_hook_liveness_rejects_input_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutformerpp.modeling_layoutformerpp"], "beartype.beartype"):
            from layoutformerpp.modeling_layoutformerpp import PositionalEncoding

        PositionalEncoding(d_model=8)(torch.zeros(2, 1, 8, dtype=torch.long))
        """,
        "PositionalEncoding.forward",
    )


def test_modeling_hook_liveness_rejects_input_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutformerpp.modeling_layoutformerpp"], "beartype.beartype"):
            from layoutformerpp import LayoutFormerPPConfig, LayoutFormerPPForConditionalGeneration

        model = LayoutFormerPPForConditionalGeneration(LayoutFormerPPConfig(vocab_size=16, d_model=8, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2, dim_feedforward=16))
        model.encode(torch.zeros(1, 2, 1, dtype=torch.long), torch.zeros(1, 2, dtype=torch.bool))
        """,
        "LayoutFormerPPForConditionalGeneration.encode",
    )


def test_modeling_hook_liveness_rejects_attention_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutformerpp.modeling_layoutformerpp"], "beartype.beartype"):
            from layoutformerpp import LayoutFormerPPConfig, LayoutFormerPPForConditionalGeneration

        model = LayoutFormerPPForConditionalGeneration(LayoutFormerPPConfig(vocab_size=16, d_model=8, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2, dim_feedforward=16))
        model.encode(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 3, dtype=torch.bool))
        """,
        "LayoutFormerPPForConditionalGeneration.encode",
    )


def test_modeling_without_hook_accepts_annotation_only_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layoutformerpp.modeling_layoutformerpp import PositionalEncoding

        pos = PositionalEncoding(d_model=8)
        assert pos(torch.zeros(2, 1, 8, dtype=torch.long)).shape == (2, 1, 8)
        assert pos(torch.zeros(2, 8)).shape == (2, 2, 8)
        """
    )
