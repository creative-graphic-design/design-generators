import subprocess
import sys
import textwrap


def test_runtime_import_hook_validates_layoutvae_shapes():
    script = r"""
    from typing import cast

    import torch
    from jaxtyping import install_import_hook
    from laygen.modeling_outputs import LayoutGenerationOutput

    with install_import_hook("layoutvae", "beartype.beartype"):
        import layoutvae.modeling_layoutvae as modeling
        import layoutvae.pipeline_layoutvae as pipeline
        import layoutvae.processing_layoutvae as processing

    processor = processing.LayoutVAEProcessor()
    encoded = processor(["text", "figure"])
    label_set = encoded["label_set"]
    assert tuple(label_set.shape) == (1, 6)

    model = modeling.LayoutVAEModel(modeling.LayoutVAEConfig())
    class_counts = torch.tensor([[7, 1, 0, 0, 0, 1]], dtype=torch.float32)
    output = model(label_set, class_counts=class_counts)
    assert tuple(output.bbox.shape) == (1, 9, 4)
    assert tuple(output.labels.shape) == (1, 9)
    assert tuple(output.mask.shape) == (1, 9)

    tuple_output = model(label_set, class_counts=class_counts, return_dict=False)
    assert tuple(tuple_output[1].shape) == (1, 9, 4)

    decoded = processor.batch_decode(output.bbox, output.labels, output.mask)
    assert decoded[0][0]["label"] == "figure"

    generated = cast(
        LayoutGenerationOutput,
        pipeline.LayoutVAEPipeline(model=model, processor=processor)(
            labels=["text", "figure"],
            class_counts=class_counts,
        ),
    )
    assert tuple(generated.bbox.shape) == (1, 9, 4)

    try:
        processor.batch_decode(
            torch.zeros(1, 1, 5), torch.zeros(1, 1, dtype=torch.long)
        )
    except Exception:
        pass
    else:
        raise AssertionError("jaxtyping did not reject bbox with last dimension 5")
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
