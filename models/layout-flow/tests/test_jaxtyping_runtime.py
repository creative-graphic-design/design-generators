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


def test_modeling_hook_liveness_rejects_sample_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_flow.modeling_layout_flow"], "beartype.beartype"):
            from layout_flow.modeling_layout_flow import LayoutFlowModelOutput

        LayoutFlowModelOutput(sample=torch.zeros(1, 2, 5, dtype=torch.long))
        """,
        "LayoutFlowModelOutput.__init__",
    )


def test_modeling_hook_liveness_rejects_sample_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_flow.modeling_layout_flow"], "beartype.beartype"):
            from layout_flow.modeling_layout_flow import LayoutFlowModelOutput

        LayoutFlowModelOutput(sample=torch.zeros(1, 2))
        """,
        "LayoutFlowModelOutput.__init__",
    )


def test_modeling_hook_liveness_rejects_condition_mask_shape_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_flow.modeling_layout_flow"], "beartype.beartype"):
            from layout_flow import LayoutFlowTransformerModel

        model = LayoutFlowTransformerModel(num_labels=6, latent_dim=8, d_model=16, nhead=4, dim_feedforward=32, num_layers=1)
        model(sample=torch.zeros(1, 2, 7), timestep=torch.tensor(0.0), cond_mask=torch.zeros(1, 3, 7, dtype=torch.bool))
        """,
        "LayoutFlowTransformerModel.forward",
    )


def test_modeling_without_hook_accepts_output_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layout_flow.modeling_layout_flow import LayoutFlowModelOutput

        assert LayoutFlowModelOutput(sample=torch.zeros(1, 2, 5, dtype=torch.long)).sample.shape == (1, 2, 5)
        assert LayoutFlowModelOutput(sample=torch.zeros(1, 2)).sample.shape == (1, 2)
        """
    )


def test_processor_hook_liveness_rejects_label_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_flow.processing_layout_flow"], "beartype.beartype"):
            from layout_flow.configuration_layout_flow import LayoutFlowConfig
            from layout_flow.processing_layout_flow import LayoutFlowProcessor

        processor = LayoutFlowProcessor(LayoutFlowConfig(max_length=2))
        processor.encode_labels(torch.zeros(1, 2, dtype=torch.float32))
        """,
        "LayoutFlowProcessor.encode_labels",
    )


def test_scheduler_hook_liveness_rejects_state_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_flow.scheduling_layout_flow"], "beartype.beartype"):
            from layout_flow.scheduling_layout_flow import LayoutFlowEulerScheduler

        scheduler = LayoutFlowEulerScheduler(num_inference_steps=2)
        scheduler.step(torch.zeros(1, 2), torch.tensor(0.0), torch.zeros(1, 2, 3))
        """,
        "LayoutFlowEulerScheduler.step",
    )


def test_training_loss_hook_liveness_rejects_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_flow.training.losses"], "beartype.beartype"):
            from layout_flow.training.losses import layout_flow_losses

        layout_flow_losses(torch.ones(1, 2, 3), torch.ones(1, 2, 3), torch.ones(1, 2))
        """,
        "layout_flow_losses",
    )
