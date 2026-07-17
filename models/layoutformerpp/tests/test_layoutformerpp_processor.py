import torch

from layoutformerpp import LayoutFormerPPProcessor


def test_processor_label_condition_and_postprocess() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="gen_t")
    batch = processor(
        condition_type="label_size",
        labels=[["Text"]],
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
    )
    assert batch["input_ids"].shape[0] == 1
    ids = processor.tokenizer.encode_text("label_1 0 0 10 10 |")["input_ids"]
    out = processor.post_process_layouts(ids)
    assert out.labels.tolist() == [[0]]
    assert out.mask.tolist() == [[True]]
    assert out.id2label[0] == "Text"
