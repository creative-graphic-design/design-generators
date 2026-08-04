from __future__ import annotations

from posterllava.model_card import build_posterllava_model_card


def test_model_card_contains_transformers_metadata() -> None:
    card = build_posterllava_model_card()

    data = card.data.to_dict()
    assert data["library_name"] == "transformers"
    assert data["license"] == "other"
