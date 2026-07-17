import torch

from laygen.common import DatasetName as LayoutDatasetName
from posgen.common import (
    DatasetName,
    PositionContent,
    assert_position_content_schema,
    id2label_for_dataset,
    label2id_for_dataset,
    labels_for_dataset,
    normalize_dataset_name,
    normalize_label,
    render_position_summary,
)


def test_position_content_schema_and_helpers():
    content = PositionContent(
        positions=torch.zeros(1, 2, 2),
        mask=torch.tensor([[True, False]]),
    )

    assert_position_content_schema(content)
    assert normalize_label("Anchor-Point") == "anchor_point"
    assert render_position_summary(content) == "1 active positions"


def test_posgen_dataset_registry_and_domain_split():
    assert {item.value for item in DatasetName} == {
        "cgl",
        "cgl_v2",
        "pku_posterlayout",
        "crello",
    }
    assert normalize_dataset_name(DatasetName.crello) is DatasetName.crello
    assert normalize_dataset_name("CGL-Dataset-v2") is DatasetName.cgl_v2
    assert normalize_dataset_name("pku-poster-layout") is DatasetName.pku_posterlayout
    assert labels_for_dataset("crello") == (
        "coloredBackground",
        "imageElement",
        "maskElement",
        "svgElement",
        "textElement",
    )
    assert label2id_for_dataset("cgl")["highlighted text"] == 4
    assert id2label_for_dataset("pku")[3] == "INVALID"
    assert set(item.value for item in DatasetName).isdisjoint(
        item.value for item in LayoutDatasetName
    )
    try:
        normalize_dataset_name("rico25")
    except ValueError as exc:
        assert "Unknown posgen dataset_name" in str(exc)
    else:
        raise AssertionError("layout dataset should not resolve in posgen")
