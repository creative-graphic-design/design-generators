import subprocess
import sys
import textwrap


def test_runtime_import_hook_validates_layout_detr_shapes():
    script = r"""
    import torch
    from PIL import Image
    from jaxtyping import install_import_hook

    with install_import_hook("layout_detr", "beartype.beartype"):
        import layout_detr.modeling_layout_detr as modeling
        import layout_detr.pipeline_layout_detr as pipeline
        import layout_detr.processing_layout_detr as processing

    config = modeling.LayoutDetrConfig(
        background_size=8,
        hidden_dim=16,
        bert_f_dim=16,
        max_text_length=4,
        text_vocab_size=64,
    )
    model = modeling.LayoutDetrForConditionalGeneration(config)
    processor = processing.LayoutDetrProcessor(config=config)
    encoded = processor(
        images=Image.new("RGB", (8, 8), "white"),
        texts=["Sale", "Shop"],
        labels=["header", "button"],
    )
    assert tuple(encoded["pixel_values"].shape) == (1, 3, 8, 8)
    assert tuple(encoded["input_ids"].shape) == (1, 9, 4)
    assert tuple(encoded["layout_mask"].shape) == (1, 9)

    latents = torch.zeros(1, 9, config.z_dim)
    output = model(
        pixel_values=encoded["pixel_values"],
        input_ids=encoded["input_ids"],
        text_attention_mask=encoded["text_attention_mask"],
        bbox_labels=encoded["bbox_labels"],
        layout_mask=encoded["layout_mask"],
        latents=latents,
        text_lengths=encoded["text_lengths"],
    )
    assert tuple(output.bbox.shape) == (1, 9, 4)
    assert tuple(output.labels.shape) == (1, 9)
    assert tuple(output.mask.shape) == (1, 9)

    generated = pipeline.LayoutDetrPipeline(
        model=model,
        processor=processor,
        config=config,
    )(
        Image.new("RGB", (8, 8), "white"),
        texts=["Sale"],
        labels=["header"],
        latents=torch.zeros(1, 9, config.z_dim),
    )
    assert tuple(generated.bbox.shape) == (1, 9, 4)

    try:
        modeling.LayoutDetrModelOutput(
            bbox=torch.zeros(1, 2, 5),
            labels=torch.zeros(1, 2, dtype=torch.long),
            mask=torch.ones(1, 2, dtype=torch.bool),
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
