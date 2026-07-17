import pickle

import pytest
import torch

from layout_dm.conversion import (
    load_cluster_centers,
    remap_denoiser_key,
    split_original_state_dict,
    write_layoutdm_model_card,
)


class FakeKMeans:
    def __init__(self, values: list[float]) -> None:
        self.cluster_centers_ = (
            torch.tensor(values, dtype=torch.float32).reshape(-1, 1).numpy()
        )


def test_remap_denoiser_key():
    assert (
        remap_denoiser_key("model.module.transformer.cat_emb.weight")
        == "transformer.cat_emb.weight"
    )
    with pytest.raises(KeyError):
        remap_denoiser_key("model.module.other.weight")


def test_split_original_state_dict_ignores_scheduler_buffers():
    state = {
        "model.module.transformer.cat_emb.weight": torch.zeros(1),
        "model.module.c_log_at": torch.zeros(1),
        "model.module.Lt_history": torch.zeros(1),
        "model.module.zero_vector": torch.zeros(1),
    }
    assert split_original_state_dict(state) == {
        "transformer.cat_emb.weight": state["model.module.transformer.cat_emb.weight"]
    }
    with pytest.raises(KeyError):
        split_original_state_dict({"model.module.other.weight": torch.zeros(1)})


def test_load_cluster_centers_sorts_original_kmeans_centers(tmp_path):
    cluster_dir = tmp_path / "clustering_weights"
    cluster_dir.mkdir()
    models = {f"{key}-32": FakeKMeans([0.8, 0.2, 0.5]) for key in ("x", "y", "w", "h")}
    with (cluster_dir / "rico25_max25_kmeans_train_clusters.pkl").open("wb") as f:
        pickle.dump(models, f)

    centers = load_cluster_centers(tmp_path, "rico25")

    assert centers["x"] == pytest.approx([0.2, 0.5, 0.8])


def test_write_layoutdm_model_card(tmp_path):
    path = write_layoutdm_model_card(tmp_path, "rico25")
    text = path.read_text(encoding="utf-8")

    assert path.name == "README.md"
    assert "license: apache-2.0" in text
    assert "library_name: diffusers" in text
    assert "pipeline_tag: text-to-image" in text
    assert "creative-graphic-design/rico25" in text
    assert "LayoutDMPipeline.from_pretrained" in text
    assert "## Uses" in text
    assert "## Evaluation" in text
    assert "Tokenizer exact" in text
    assert "[More Information Needed]" not in text
    assert "This card follows" not in text
    assert "annotated model card" not in text
    assert "model card template" not in text
