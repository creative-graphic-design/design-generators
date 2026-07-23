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


def test_modeling_hook_liveness_rejects_bad_latent_dtype() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["coarse_to_fine.modeling_coarse_to_fine"], "beartype.beartype"
        ):
            from coarse_to_fine import CoarseToFineConfig
            from coarse_to_fine.modeling_coarse_to_fine import VAE

        vae = VAE(CoarseToFineConfig(d_z=4, d_model=8))
        vae.inference(torch.zeros(1, 2, 4, dtype=torch.long), batch_size=2, device=torch.device("cpu"))
        """,
        "VAE.inference",
    )


def test_modeling_hook_liveness_rejects_bad_latent_rank() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["coarse_to_fine.modeling_coarse_to_fine"], "beartype.beartype"
        ):
            from coarse_to_fine import CoarseToFineConfig
            from coarse_to_fine.modeling_coarse_to_fine import VAE

        vae = VAE(CoarseToFineConfig(d_z=4, d_model=8))
        vae.inference(torch.zeros(1, 2, 4, 1), batch_size=2, device=torch.device("cpu"))
        """,
        "VAE.inference",
    )


def test_modeling_hook_liveness_rejects_bad_latent_shape() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["coarse_to_fine.modeling_coarse_to_fine"], "beartype.beartype"
        ):
            from coarse_to_fine import CoarseToFineConfig
            from coarse_to_fine.modeling_coarse_to_fine import VAE

        vae = VAE(CoarseToFineConfig(d_z=4, d_model=8))
        vae.inference(torch.zeros(2, 2, 4), batch_size=2, device=torch.device("cpu"))
        """,
        "VAE.inference",
    )


def test_modeling_without_hook_accepts_latent_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from coarse_to_fine import CoarseToFineConfig
        from coarse_to_fine.modeling_coarse_to_fine import VAE

        vae = VAE(CoarseToFineConfig(d_z=4, d_model=8))
        assert vae.inference(torch.zeros(1, 2, 4, dtype=torch.long), batch_size=2, device=torch.device("cpu")).shape == (1, 2, 4)
        assert vae.inference(torch.zeros(1, 2, 4, 1), batch_size=2, device=torch.device("cpu")).shape == (1, 2, 4, 1)
        assert vae.inference(torch.zeros(2, 2, 4), batch_size=2, device=torch.device("cpu")).shape == (2, 2, 4)
        """
    )
