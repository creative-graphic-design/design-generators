import subprocess
import sys
import textwrap


def test_runtime_import_hook_validates_canvas_vae_shapes():
    script = r"""
    import torch
    from jaxtyping import install_import_hook

    with install_import_hook("canvas_vae", "beartype.beartype"):
        import canvas_vae.metrics as metrics
        import canvas_vae.modeling_canvas_vae as modeling
        import canvas_vae.pipeline_canvas_vae as pipeline
        from canvas_vae.configuration_canvas_vae import CanvasVAEConfig

    config = CanvasVAEConfig(
        vocabularies={"component": ["[UNK]", ""], "icon": ["[UNK]"], "text_button": ["[UNK]"]},
        max_length=4,
        num_bins=8,
        latent_dim=16,
        num_heads=2,
    )
    model = modeling.CanvasVAEModel(config).train()
    element_ids = torch.zeros(2, 3, 8, dtype=torch.long)
    output = model(torch.tensor([3, 1]), element_ids)
    assert output.loss.ndim == 0
    out = pipeline.CanvasVAEPipeline(model=model)(batch_size=2, seed=0)
    assert out.bbox.shape[-1] == 4

    try:
        metrics.bleu1(torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 4, dtype=torch.bool),
                      torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool), 2)
    except Exception:
        pass
    else:
        raise AssertionError("jaxtyping did not reject mismatched mask shape")
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
