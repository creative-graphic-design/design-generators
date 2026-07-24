import subprocess
import sys
import textwrap


def test_runtime_import_hook_validates_layout_transformer_shapes():
    script = r"""
    import torch
    from jaxtyping import install_import_hook

    with install_import_hook("layout_transformer", "beartype.beartype"):
        import layout_transformer.modeling_layout_transformer as modeling
        import layout_transformer.pipeline_layout_transformer as pipeline
        import layout_transformer.processing_layout_transformer as processing
        from layout_transformer.relation_schema import LayoutObject, LayoutRelation

    processor = processing.LayoutTransformerProcessor.from_config(
        max_sequence_length=6,
        id2label={0: "person", 1: "table"},
        relation_id2label={0: "left of"},
    )
    encoded = processor(
        objects=[
            LayoutObject(id="person-1", label="person"),
            LayoutObject(id="table-1", label="table"),
        ],
        relations=[
            LayoutRelation(subject="person-1", predicate="left of", object="table-1")
        ],
        return_tensors="pt",
    )
    assert tuple(encoded["input_token"].shape) == (1, 6)
    assert tuple(encoded["src_mask"].shape) == (1, 1, 6)

    config = modeling.LayoutTransformerConfig(
        vocab_size=processor.tokenizer.vocab_size,
        obj_classes_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_sequence_length=6,
    )
    model = modeling.LayoutTransformerForLayoutGeneration(config)
    output = model(**encoded)
    assert tuple(output.coarse_box.shape) == (1, 6, 4)
    assert tuple(output.vocab_logits.shape) == (1, 6, processor.tokenizer.vocab_size)

    generated = pipeline.LayoutTransformerPipeline(
        model=model,
        processor=processor,
    )(
        objects=[
            LayoutObject(id="person-1", label="person"),
            LayoutObject(id="table-1", label="table"),
        ],
        relations=[
            LayoutRelation(subject="person-1", predicate="left of", object="table-1")
        ],
    )
    assert tuple(generated.bbox.shape) == (1, 2, 4)
    assert tuple(generated.labels.shape) == (1, 2)
    assert tuple(generated.mask.shape) == (1, 2)

    try:
        model(
            input_token=encoded["input_token"],
            input_obj_id=encoded["input_obj_id"],
            segment_label=encoded["segment_label"],
            token_type=encoded["token_type"],
            src_mask=encoded["src_mask"],
            bbox=torch.zeros(1, 6, 5),
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
