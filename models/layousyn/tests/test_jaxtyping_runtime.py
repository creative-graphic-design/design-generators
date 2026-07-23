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


def test_modeling_hook_liveness_rejects_scalar_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layousyn.modeling_layousyn"], "beartype.beartype"):
            from layousyn.modeling_layousyn import ScalarEmbedder

        ScalarEmbedder(hidden_size=4, frequency_embedding_size=4)(torch.tensor([True]))
        """,
        "ScalarEmbedder.forward",
    )


def test_modeling_hook_liveness_rejects_scalar_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layousyn.modeling_layousyn"], "beartype.beartype"):
            from layousyn.modeling_layousyn import ScalarEmbedder

        ScalarEmbedder(hidden_size=4, frequency_embedding_size=4)(torch.zeros(1, 1))
        """,
        "ScalarEmbedder.forward",
    )


def test_modeling_hook_liveness_rejects_forward_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layousyn.modeling_layousyn"], "beartype.beartype"):
            from layousyn import LayouSynDiTModel

        model = LayouSynDiTModel(
            in_channels=4,
            max_in_len=2,
            concept_in_channels=8,
            y_in_channels=None,
            max_y_len=None,
            hidden_size=8,
            depth=1,
            num_heads=2,
            is_unconditional=True,
        )
        model(
            sample=torch.zeros(1, 2, 4),
            timestep=torch.tensor([1]),
            x_padding_mask=torch.ones(1, 3, dtype=torch.bool),
            aspect_ratio=torch.ones(1),
            concept_embeds=torch.zeros(1, 2, 8),
        )
        """,
        "LayouSynDiTModel.forward",
    )


def test_modeling_without_hook_accepts_annotation_only_scalar_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layousyn.modeling_layousyn import ScalarEmbedder

        embedder = ScalarEmbedder(hidden_size=4, frequency_embedding_size=4)
        assert embedder(torch.tensor([True])).shape == (1, 4)
        assert embedder(torch.zeros(1, 1)).shape == (1, 1, 4)
        """
    )
