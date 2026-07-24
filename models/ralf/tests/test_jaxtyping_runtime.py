import subprocess
import sys
import textwrap


def test_runtime_import_hook_validates_ralf_shapes():
    script = r"""
    import torch
    from jaxtyping import install_import_hook

    with install_import_hook("ralf", "beartype.beartype"):
        import ralf.configuration_ralf as configuration
        import ralf.image_processing_ralf as image_processing
        import ralf.modeling_ralf as modeling
        import ralf.processing_ralf as processing

    config = configuration.RalfConfig(max_seq_length=2, image_size=(8, 8))
    processor = processing.RalfProcessor.from_config(config)

    image = image_processing._image_to_tensor(torch.zeros(3, 8, 8), channels=3)
    assert tuple(image.shape) == (3, 8, 8)

    labels = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    bbox = torch.tensor(
        [
            [[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]],
            [[0.3, 0.3, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True], [True, False]])
    encoded = processor(
        images=[torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)],
        condition_type="label",
        labels=labels,
        bbox=bbox,
        mask=mask,
    )
    assert tuple(encoded["input_ids"].shape) == (2, 11)
    assert tuple(encoded["attention_mask"].shape) == (2, 11)
    assert tuple(encoded["constraint_bbox"].shape) == (2, 2, 4)

    output = processor.post_process_layouts(encoded["input_ids"])
    assert tuple(output.bbox.shape) == (2, 2, 4)
    assert tuple(output.labels.shape) == (2, 2)
    assert tuple(output.mask.shape) == (2, 2)

    preprocessor = modeling.RalfTaskPreprocessor(
        modeling.RalfTokenizerView(config), task="uncond"
    )
    prepared = preprocessor(
        modeling.RalfConditionalInputs(
            image=torch.zeros(2, 4, 8, 8),
            retrieved={},
        )
    )
    assert tuple(prepared["seq"].shape) == (2, 4)
    assert tuple(prepared["pad_mask"].shape) == (2, 4)

    try:
        processor.layout_tokenizer.encode_layout(
            labels=torch.zeros(1, 2, dtype=torch.long),
            bbox=torch.zeros(1, 2, 5),
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
