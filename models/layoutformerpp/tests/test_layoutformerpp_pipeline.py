from pathlib import Path

from laygen.pipelines import LayoutGenerationPipeline

from layoutformerpp import (
    LayoutFormerPPConfig,
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPPipeline,
    LayoutFormerPPProcessor,
    LayoutGenerationOutput,
)


def test_save_load_smoke(tmp_path: Path) -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="gen_t")
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        max_position_embeddings=16,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    pipe = LayoutFormerPPPipeline(model=model, processor=processor)
    assert isinstance(pipe, LayoutGenerationPipeline)

    pipe.save_pretrained(tmp_path)
    loaded_pipe = LayoutFormerPPPipeline.from_pretrained(tmp_path)
    assert isinstance(loaded_pipe, LayoutGenerationPipeline)

    pipe = loaded_pipe
    out = pipe(condition_type="label", labels=[["Text"]], max_length=2)
    assert isinstance(out, LayoutGenerationOutput)
    assert out.sequences is not None
    assert out.sequences.shape[0] == 1
