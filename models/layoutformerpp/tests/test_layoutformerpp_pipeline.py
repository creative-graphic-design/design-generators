from pathlib import Path

from layoutformerpp import (
    LayoutFormerPPConfig,
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPPipeline,
    LayoutFormerPPProcessor,
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
    model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    loaded_model = LayoutFormerPPForConditionalGeneration.from_pretrained(tmp_path)
    loaded_processor = LayoutFormerPPProcessor.from_pretrained(tmp_path)
    pipe = LayoutFormerPPPipeline(model=loaded_model, processor=loaded_processor)
    out = pipe(condition_type="label", labels=[["Text"]], max_length=2)
    assert out.sequences.shape[0] == 1
