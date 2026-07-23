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


def test_modeling_hook_liveness_rejects_sample_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["lace.modeling_lace"], "beartype.beartype"):
            from lace import LaceTransformerModel

        model = LaceTransformerModel(seq_dim=10, max_seq_length=5, num_layers=1, dim_transformer=16, nhead=2, dim_feedforward=32)
        model(sample=torch.zeros(1, 5, 10, 1), timestep=torch.tensor([1]))
        """,
        "LaceTransformerModel.forward",
    )


def test_modeling_hook_liveness_rejects_timestep_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["lace.modeling_lace"], "beartype.beartype"):
            from lace import LaceTransformerModel

        model = LaceTransformerModel(seq_dim=10, max_seq_length=5, num_layers=1, dim_transformer=16, nhead=2, dim_feedforward=32)
        model(sample=torch.zeros(1, 5, 10), timestep=torch.tensor([1.0]))
        """,
        "LaceTransformerModel.forward",
    )


def test_modeling_hook_liveness_rejects_mask_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["lace.modeling_lace"], "beartype.beartype"):
            from lace import LaceTransformerModel

        model = LaceTransformerModel(seq_dim=10, max_seq_length=5, num_layers=1, dim_transformer=16, nhead=2, dim_feedforward=32)
        model(
            sample=torch.zeros(1, 5, 10),
            timestep=torch.tensor([1]),
            attention_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        """,
        "LaceTransformerModel.forward",
    )


def test_processor_without_hook_accepts_annotation_only_dtype_mismatch() -> None:
    _assert_probe_passes(
        """
        import torch

        from lace import LaceProcessor

        processor = LaceProcessor.from_dataset("publaynet")
        bbox, labels, mask = processor.pad(
            torch.zeros(1, 1, 4, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            max_seq_length=1,
        )
        assert bbox.dtype == torch.int64
        assert labels.shape == mask.shape == (1, 1)
        """
    )
