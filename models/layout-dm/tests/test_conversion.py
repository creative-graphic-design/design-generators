import pickle
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from laygen.common import DatasetName
from laygen.common.model_card import ParityMetric, layoutdm_model_card
from layout_dm.conversion import (
    load_cluster_centers,
    remap_denoiser_key,
    split_original_state_dict,
    write_layoutdm_model_card,
)


def test_remap_denoiser_key():
    assert (
        remap_denoiser_key("model.module.transformer.cat_emb.weight")
        == "transformer.cat_emb.weight"
    )


def test_split_original_state_dict_ignores_scheduler_buffers():
    state = {
        "model.module.transformer.cat_emb.weight": torch.zeros(1),
        "model.module.c_log_at": torch.zeros(1),
        "model.module.Lt_history": torch.zeros(1),
    }
    assert split_original_state_dict(state) == {
        "transformer.cat_emb.weight": state["model.module.transformer.cat_emb.weight"]
    }
    try:
        remap_denoiser_key("model.module.other.weight")
    except KeyError as exc:
        assert "model.module.other.weight" in str(exc)
    else:
        raise AssertionError("non-transformer key should fail")
    try:
        split_original_state_dict({"model.module.other.weight": torch.zeros(1)})
    except KeyError as exc:
        assert "model.module.other.weight" in str(exc)
    else:
        raise AssertionError("unexpected original key should fail")


def test_load_cluster_centers(tmp_path):
    cluster_dir = tmp_path / "clustering_weights"
    cluster_dir.mkdir()
    models = {
        f"{key}-32": SimpleNamespace(cluster_centers_=np.array([[0.3], [0.1], [0.2]]))
        for key in ("x", "y", "w", "h")
    }
    with (cluster_dir / "publaynet_max25_kmeans_train_clusters.pkl").open("wb") as f:
        pickle.dump(models, f)

    centers = load_cluster_centers(tmp_path, "publaynet")
    assert centers["x"] == [0.1, 0.2, 0.3]


def test_write_layoutdm_model_card(tmp_path):
    path = write_layoutdm_model_card(tmp_path, DatasetName.rico25)
    text = path.read_text(encoding="utf-8")
    front_matter = text.split("---", maxsplit=2)[1]
    metadata = yaml.safe_load(front_matter)

    assert path.name == "README.md"
    assert metadata["datasets"] == ["creative-graphic-design/Rico"]
    assert "tag:yaml.org" not in text
    assert "license: apache-2.0" in text
    assert "library_name: diffusers" in text
    assert "pipeline_tag: text-to-image" in text
    assert "creative-graphic-design/Rico" in text
    assert "LayoutDMPipeline.from_pretrained" in text
    assert "## Uses" in text
    assert "## Evaluation" in text
    assert "Tokenizer exact" in text
    assert "[More Information Needed]" not in text
    assert "This card follows" not in text
    assert "annotated model card" not in text
    assert "model card template" not in text


def test_model_card_yaml_front_matter_accepts_enum_metadata():
    card = layoutdm_model_card(
        dataset=DatasetName.publaynet,
        parity_metrics=[
            ParityMetric(
                dataset=DatasetName.publaynet,
                tokenizer_exact="1/1",
                deterministic_exact="1/1",
                logits_max_abs=0.0,
                logits_max_rel=0.0,
            )
        ],
    )
    text = str(card)
    metadata = yaml.safe_load(text.split("---", maxsplit=2)[1])

    assert metadata["tags"][-1] == "publaynet"
    assert metadata["datasets"] == ["creative-graphic-design/PubLayNet"]
    assert "tag:yaml.org" not in text
