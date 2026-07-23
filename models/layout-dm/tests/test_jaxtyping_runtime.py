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


def test_denoiser_hook_liveness_rejects_logits_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_dm.denoiser"], "beartype.beartype"):
            from layout_dm.denoiser import LayoutDMDenoiserOutput

        LayoutDMDenoiserOutput(logits=torch.zeros(1, 2, 3, dtype=torch.long))
        """,
        "LayoutDMDenoiserOutput.__init__",
    )


def test_denoiser_hook_liveness_rejects_logits_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_dm.denoiser"], "beartype.beartype"):
            from layout_dm.denoiser import LayoutDMDenoiserOutput

        LayoutDMDenoiserOutput(logits=torch.zeros(1, 2))
        """,
        "LayoutDMDenoiserOutput.__init__",
    )


def test_denoiser_hook_liveness_rejects_timestep_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_dm.denoiser"], "beartype.beartype"):
            from layout_dm import LayoutDMDenoiser

        model = LayoutDMDenoiser(vocab_size=8, max_token_length=4, hidden_size=8, num_attention_heads=2, num_hidden_layers=1, intermediate_size=16)
        model(input_ids=torch.zeros(1, 4, dtype=torch.long), timesteps=torch.zeros(2, dtype=torch.long))
        """,
        "LayoutDMDenoiser.forward",
    )


def test_denoiser_without_hook_accepts_output_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layout_dm.denoiser import LayoutDMDenoiserOutput

        assert LayoutDMDenoiserOutput(logits=torch.zeros(1, 2, 3, dtype=torch.long)).logits.shape == (1, 2, 3)
        assert LayoutDMDenoiserOutput(logits=torch.zeros(1, 2)).logits.shape == (1, 2)
        """
    )
