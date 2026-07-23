import csv

from PIL import Image

from cgb_dm.data import CGBDMDataPaths, CGBDMOriginalDataset


def test_original_dataset_reads_tiny_extract(tmp_path):
    root = tmp_path / "split"
    for rel in [
        "train/inpaint",
        "train/saliency",
        "train/saliency_sub",
        "csv",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for rel, mode in [
        ("train/inpaint/sample.png", "RGB"),
        ("train/saliency/sample.png", "L"),
        ("train/saliency_sub/sample.png", "L"),
    ]:
        Image.new(mode, (20, 20), color=255).save(root / rel)
    with (root / "csv/train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["poster_path", "box_elem", "cls_elem"]
        )
        writer.writeheader()
        writer.writerow(
            {"poster_path": "sample.png", "box_elem": "[0, 0, 10, 10]", "cls_elem": "1"}
        )
    with (root / "csv/train_sal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["poster_path", "box_elem"])
        writer.writeheader()
        writer.writerow({"poster_path": "sample.png", "box_elem": "[0, 0, 20, 20]"})

    paths = CGBDMDataPaths(root)
    assert paths.annotation_csv.name == "train.csv"
    dataset = CGBDMOriginalDataset(root, split="train")
    row = dataset[0]
    assert row["pixel_values"].shape[0] == 4
    assert row["layout"].shape == (16, 8)
    assert row["saliency_box"].shape == (1, 4)
