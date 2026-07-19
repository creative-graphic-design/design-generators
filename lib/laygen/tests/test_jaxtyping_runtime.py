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


def test_bbox_hook_liveness_rejects_bad_box_shape() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["laygen.common.bbox"], "beartype.beartype"):
            from laygen.common.bbox import xywh_to_ltrb

        xywh_to_ltrb(torch.zeros(1, 2, 5))
        """,
        "xywh_to_ltrb",
    )


def test_discrete_hook_liveness_rejects_bad_scores_shape() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["laygen.common.discrete"], "beartype.beartype"):
            from laygen.common.discrete import batch_topk_mask

        batch_topk_mask(torch.zeros(1, 2, 3), torch.tensor([1]))
        """,
        "batch_topk_mask",
    )


def test_scheduler_hook_liveness_rejects_bad_extract_shape() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["laygen.schedulers.continuous"], "beartype.beartype"):
            import laygen.schedulers.continuous as continuous

        continuous._betas_for_alpha_bar = lambda *args, **kwargs: torch.zeros(2, 2)
        continuous.get_layoutdiffusion_beta_schedule("sqrt", 4)
        """,
        "get_layoutdiffusion_beta_schedule",
    )


def test_nn_blocks_hook_liveness_rejects_bad_hidden_shape() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["laygen.nn"], "beartype.beartype"):
            from laygen.nn.blocks import TimestepTransformerEncoderLayer

        layer = TimestepTransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            timestep_type=None,
        )
        layer(torch.zeros(2, 8))
        """,
        "TimestepTransformerEncoderLayer.forward",
    )


def test_nn_embeddings_hook_liveness_rejects_bad_timestep_shape() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["laygen.nn"], "beartype.beartype"):
            from laygen.nn.embeddings import SinusoidalPosEmb

        SinusoidalPosEmb(num_steps=10, dim=8)(torch.zeros(1, 1, dtype=torch.long))
        """,
        "SinusoidalPosEmb.forward",
    )


def test_nn_adaptive_norm_path_hook_liveness_rejects_bad_timestep_shape() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["laygen.nn"], "beartype.beartype"):
            from laygen.nn.blocks import TimestepTransformerEncoderLayer

        layer = TimestepTransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            timestep_type="adalayernorm_abs",
        )
        layer(torch.zeros(1, 2, 8), timestep=torch.zeros(1, 1, dtype=torch.long))
        """,
        "TimestepTransformerEncoderLayer.forward",
    )
