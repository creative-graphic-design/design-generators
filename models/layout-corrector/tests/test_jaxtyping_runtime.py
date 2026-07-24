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


def test_corrector_hook_liveness_rejects_output_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["layout_corrector.modeling_layout_corrector"], "beartype.beartype"
        ):
            from layout_corrector.modeling_layout_corrector import LayoutCorrectorOutput

        LayoutCorrectorOutput(logits=torch.zeros(1, 2, dtype=torch.long))
        """,
        "LayoutCorrectorOutput.__init__",
    )


def test_corrector_hook_liveness_rejects_output_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["layout_corrector.modeling_layout_corrector"], "beartype.beartype"
        ):
            from layout_corrector.modeling_layout_corrector import LayoutCorrectorOutput

        LayoutCorrectorOutput(logits=torch.zeros(1, 2, 1))
        """,
        "LayoutCorrectorOutput.__init__",
    )


def test_corrector_hook_liveness_rejects_padding_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["layout_corrector.modeling_layout_corrector"], "beartype.beartype"
        ):
            from layout_corrector.modeling_layout_corrector import LayoutCorrectorModel

        model = LayoutCorrectorModel(
            dataset_name="publaynet",
            vocab_size=8,
            max_seq_length=2,
            num_attributes_per_element=2,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=16,
        )
        model(
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            timesteps=torch.tensor([1]),
            padding_mask=torch.ones(1, 5, dtype=torch.bool),
        )
        """,
        "LayoutCorrectorModel.forward",
    )


def test_corrector_without_hook_accepts_output_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layout_corrector.modeling_layout_corrector import LayoutCorrectorOutput

        assert LayoutCorrectorOutput(logits=torch.zeros(1, 2, dtype=torch.long)).logits.shape == (1, 2)
        assert LayoutCorrectorOutput(logits=torch.zeros(1, 2, 1)).logits.shape == (1, 2, 1)
        """
    )
