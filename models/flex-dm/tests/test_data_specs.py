"""Flex-DM data-spec tests."""

from flex_dm.data_specs import (
    attribute_groups_for_dataset,
    build_column_specs,
    id2label_from_vocabulary,
    load_builtin_spec,
)


def test_builtin_specs_and_groups() -> None:
    """Built-in specs expose the vendor sequence fields."""
    crello = load_builtin_spec("crello")
    rico = load_builtin_spec("rico")

    assert crello["name"] == "crello"
    assert rico["name"] == "rico"
    assert attribute_groups_for_dataset("crello")["pos"] == (
        "left",
        "top",
        "width",
        "height",
    )
    assert "txt" not in attribute_groups_for_dataset("rico")


def test_column_specs_and_vocabulary_mapping() -> None:
    """Column specs infer geometry bins and conditional Crello attributes."""
    specs = build_column_specs(
        dataset_name="crello",
        vocabulary={"type": ["textElement", "imageElement"]},
    )

    assert specs["left"]["input_dim"] == 64
    assert specs["image_embedding"]["type"] == "numerical"
    assert specs["color"]["loss_condition"]["key"] == "type"
    assert id2label_from_vocabulary("crello", {"type": ["a", "b"]}) == {
        0: "a",
        1: "b",
    }


def test_crello_loss_conditions_follow_vendor_lookup_order() -> None:
    """Crello attribute validity follows StringLookup vocabulary order."""
    specs = build_column_specs(
        dataset_name="crello",
        vocabulary={
            "type": {
                "svgElement": 118006,
                "textElement": 96879,
                "imageElement": 21306,
                "coloredBackground": 5791,
                "maskElement": 4117,
            },
        },
    )

    assert specs["color"]["loss_condition"]["mask"] == (
        False,
        False,
        True,
        False,
        True,
        False,
    )
    assert specs["font_family"]["loss_condition"]["mask"] == (
        False,
        False,
        True,
        False,
        False,
        False,
    )
