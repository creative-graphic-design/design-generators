import numpy as np
import torch
from torch.nn import functional as F

import canvas_vae.metrics as canvas_vae_metrics
from canvas_vae import CanvasVAECrelloConfig, CanvasVAECrelloModel
from canvas_vae.data import (
    CrelloDocument,
    CrelloElement,
    CrelloProcessor,
    CrelloSplit,
    document_id,
)
from canvas_vae.metrics import (
    _scaled_mean_cosine_similarity,
    crello_field_statistics,
    crello_histogram_scores,
    crello_reconstruction_scores,
)
from canvas_vae.modeling_canvas_vae import CanvasVAECrelloModelOutput


VOCABULARIES: dict[str, list[str | int]] = {
    "length": list(range(1, 51)),
    "group": ["", "group"],
    "format": ["", "poster"],
    "canvas_width": [640],
    "canvas_height": [480],
    "category": ["", "food"],
    "type": [
        "",
        "textElement",
        "coloredBackground",
        "imageElement",
        "svgElement",
        "maskElement",
        "otherElement",
    ],
}


def make_element(kind: str, image_key: str, index: int) -> CrelloElement:
    return {
        "type": kind,
        "left": 0.1 * index,
        "top": 0.1,
        "width": 0.3,
        "height": 0.2,
        "opacity": 0.75,
        "color": [index * 20, 128, 255],
        "image_id": image_key,
    }


def make_document(source_id: str, types: tuple[str, ...]) -> CrelloDocument:
    elements = [
        make_element(kind, f"image-{source_id}-{index}", index)
        for index, kind in enumerate(types)
    ]
    return {
        "split": "train",
        "document_id": document_id(CrelloSplit.train, source_id),
        "context": {
            "id": source_id,
            "length": len(elements),
            "group": "group",
            "format": "poster",
            "canvas_width": 640,
            "canvas_height": 480,
            "category": "food",
        },
        "elements": elements,
    }


def make_batch():
    documents = [
        make_document(
            "a",
            (
                "textElement",
                "imageElement",
                "coloredBackground",
                "svgElement",
                "maskElement",
                "otherElement",
            ),
        ),
        make_document("b", ("imageElement", "textElement")),
    ]
    embeddings = {
        element["image_id"]: np.full(
            256, (document_index + element_index + 1) / 10, dtype=np.float32
        )
        for document_index, row in enumerate(documents)
        for element_index, element in enumerate(row["elements"])
    }
    return CrelloProcessor(VOCABULARIES, embeddings)(documents)


def make_config() -> CanvasVAECrelloConfig:
    return CanvasVAECrelloConfig(
        vocabularies=VOCABULARIES,
        latent_dim=16,
        num_heads=2,
        dropout=0.0,
    )


def exact_output(batch) -> CanvasVAECrelloModelOutput:
    config = make_config()
    length_logits = F.one_hot(batch["length"].squeeze(-1), 50).float() * 10
    context_logits = {
        field: F.one_hot(batch[field], config.context_field_sizes[field]).float() * 10
        for field in config.context_fields[1:]
    }
    sequence_logits = {}
    for field, classes in config.sequence_field_sizes.items():
        target = batch[field].squeeze(-1) if field != "color" else batch[field]
        sequence_logits[field] = F.one_hot(target, classes).float() * 10

    return CanvasVAECrelloModelOutput(
        length_logits=length_logits,
        context_logits=context_logits,
        sequence_logits=sequence_logits,
        numerical_predictions={"image_embedding": batch["image_embedding"]},
        mask=batch["element_mask"],
    )


def test_config_defaults_vocabularies_and_round_trip(tmp_path):
    config = CanvasVAECrelloConfig(vocabularies=VOCABULARIES)
    assert config.latent_dim == 512
    assert config.kl_weight == 32
    assert config.context_field_sizes["length"] == 50
    assert config.sequence_field_sizes["color"] == 16
    assert config.conditional_type_ids("color") == (1, 2)
    assert config.conditional_type_ids("image_embedding") == (4, 3, 5)

    config.save_pretrained(tmp_path)
    loaded = CanvasVAECrelloConfig.from_pretrained(tmp_path)
    assert loaded.vocabularies == config.vocabularies
    assert loaded.id2label == config.id2label
    assert loaded.primary_label_id == 0


def test_training_masks_color_channels_and_numerical_embeddings():
    batch = make_batch()
    model = CanvasVAECrelloModel(make_config()).train()
    output = model(batch, posterior_noise=torch.zeros(2, 16))

    assert output.loss is not None
    assert output.reconstruction_losses is not None
    assert output.sequence_logits["color"].shape == (2, 6, 3, 16)
    assert output.numerical_predictions["image_embedding"].shape == (2, 6, 256)

    color_values = F.cross_entropy(
        output.sequence_logits["color"].reshape(-1, 16),
        batch["color"].reshape(-1),
        reduction="none",
    ).reshape(2, 6, 3)
    expected_color = (
        (color_values * batch["color_mask"].unsqueeze(-1)).sum(dim=(1, 2)).mean()
    )
    torch.testing.assert_close(output.reconstruction_losses["color"], expected_color)

    embedding_error = (
        output.numerical_predictions["image_embedding"] - batch["image_embedding"]
    ).square().mean(dim=-1) * 256
    expected_embedding = (
        (embedding_error * batch["image_embedding_mask"]).sum(dim=1).mean()
    )
    torch.testing.assert_close(
        output.reconstruction_losses["image_embedding"], expected_embedding
    )
    output.loss.backward()


def test_reconstruction_metrics_match_vendor_field_channel_weights():
    batch = make_batch()
    scores = crello_reconstruction_scores(batch, exact_output(batch), make_config())
    assert scores["color"].shape == (2, 3)
    assert scores["total"].shape == (2, 1)
    for key in (
        "total",
        "type",
        "color",
        "image_embedding",
        "layout_acc",
        "layout_miou",
    ):
        torch.testing.assert_close(scores[key], torch.ones_like(scores[key]))


def test_scaled_cosine_uses_tensorflow_epsilon_for_near_zero_vectors():
    score = _scaled_mean_cosine_similarity(
        torch.tensor([[[1e-10, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[True]]),
        torch.tensor([[True]]),
    )

    torch.testing.assert_close(score, torch.tensor([0.50005]), rtol=1e-5, atol=1e-7)


def test_histogram_cosine_uses_tensorflow_epsilon_for_near_zero_vectors():
    scores = crello_histogram_scores(
        {"image_embedding": torch.tensor([[1e-10, 0.0]])},
        {"image_embedding": torch.tensor([[1.0, 0.0]])},
    )

    assert abs(scores["image_embedding"] - 0.49995) < 1e-7


def test_generation_statistics_match_vendor_color_channels_and_cosine_loss():
    batch = make_batch()
    config = make_config()
    output = exact_output(batch)
    reference = crello_field_statistics(batch, config)
    generated = crello_field_statistics(batch, config, output)
    assert reference["color"].shape == (16, 3)
    assert generated["color"].shape == (16, 3)

    scores = crello_histogram_scores(reference, generated)
    for key in ("type", "color", "group", "length"):
        assert scores[key] == 1.0
    assert abs(scores["image_embedding"]) < 1e-6
    assert 0.0 <= scores["total"] <= 1.0


def test_image_embedding_histogram_score_maps_negative_cosine_to_one():
    scores = crello_histogram_scores(
        {"image_embedding": torch.tensor([[1.0, 0.0]])},
        {"image_embedding": torch.tensor([[-1.0, 0.0]])},
    )

    assert scores["image_embedding"] == 1.0
    assert scores["total"] == 1.0


def test_crello_layout_metric_rasterizes_geometry_vocab_sizes(monkeypatch):
    batch = make_batch()
    output = exact_output(batch)
    config = make_config()
    config.max_length = 7
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def capture_grid_size(
        target: torch.Tensor, predicted: torch.Tensor, num_labels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del num_labels
        shapes.append((tuple(target.shape), tuple(predicted.shape)))
        return torch.tensor(0.0), torch.tensor(0.0)

    monkeypatch.setattr(canvas_vae_metrics, "_grid_scores", capture_grid_size)
    canvas_vae_metrics._crello_layout_scores(batch, output, config)

    assert shapes == [((64, 64), (64, 64)), ((64, 64), (64, 64))]


def test_generation_statistics_accept_empty_conditional_predictions():
    batch = make_batch()
    config = make_config()
    output = exact_output(batch)
    output.mask = torch.zeros_like(batch["element_mask"])

    generated = crello_field_statistics(batch, config, output)
    scores = crello_histogram_scores(crello_field_statistics(batch, config), generated)

    assert torch.equal(generated["color"], torch.zeros(16, 3))
    assert torch.equal(generated["image_embedding"], torch.zeros(1, 256))
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_save_pretrained_round_trip_preserves_crello_decode(tmp_path):
    model = CanvasVAECrelloModel(make_config()).eval()
    latents = torch.randn(3, 16, generator=torch.Generator().manual_seed(11))
    model.save_pretrained(tmp_path)
    loaded = CanvasVAECrelloModel.from_pretrained(tmp_path).eval()
    expected = model(latents=latents)
    actual = loaded(latents=latents)
    torch.testing.assert_close(actual.length_logits, expected.length_logits)
    torch.testing.assert_close(
        actual.sequence_logits["color"], expected.sequence_logits["color"]
    )
    torch.testing.assert_close(
        actual.numerical_predictions["image_embedding"],
        expected.numerical_predictions["image_embedding"],
    )
