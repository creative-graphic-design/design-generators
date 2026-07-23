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


def test_modeling_hook_liveness_rejects_hidden_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["flex_dm.modeling_flex_dm"], "beartype.beartype"):
            from flex_dm.modeling_flex_dm import FlexDmMultiHeadSelfAttention

        attention = FlexDmMultiHeadSelfAttention(hidden_size=8, num_heads=2)
        attention._split(torch.zeros(1, 2, 8, dtype=torch.long))
        """,
        "FlexDmMultiHeadSelfAttention._split",
    )


def test_modeling_hook_liveness_rejects_hidden_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["flex_dm.modeling_flex_dm"], "beartype.beartype"):
            from flex_dm.modeling_flex_dm import FlexDmDecoder
            from flex_dm.testing import tiny_config

        decoder = FlexDmDecoder(tiny_config())
        decoder(torch.zeros(1, 2, 1, 16))
        """,
        "FlexDmDecoder.forward",
    )


def test_modeling_hook_liveness_rejects_mask_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["flex_dm.modeling_flex_dm"], "beartype.beartype"):
            from flex_dm.modeling_flex_dm import FlexDmMultiHeadSelfAttention

        attention = FlexDmMultiHeadSelfAttention(hidden_size=8, num_heads=2)
        attention(torch.zeros(1, 2, 8), torch.ones(1, 3, dtype=torch.bool))
        """,
        "FlexDmMultiHeadSelfAttention.forward",
    )


def test_modeling_without_hook_accepts_annotation_only_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from flex_dm.modeling_flex_dm import FlexDmDecoder, FlexDmMultiHeadSelfAttention
        from flex_dm.testing import tiny_config

        attention = FlexDmMultiHeadSelfAttention(hidden_size=8, num_heads=2)
        assert attention._split(torch.zeros(1, 2, 8, dtype=torch.long)).shape == (1, 2, 2, 4)

        decoder = FlexDmDecoder(tiny_config())
        outputs = decoder(torch.zeros(1, 2, 1, 16))
        assert outputs["left"].shape == (1, 2, 1, 64)
        """
    )
