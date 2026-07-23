import csv
import json

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


def test_original_dataset_replays_manifest_with_vendor_encoding(tmp_path):
    root = tmp_path / "split"
    for rel in [
        "train/inpaint",
        "train/saliency",
        "train/saliency_sub",
        "csv",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for name in ["b.png", "a.png"]:
        Image.new("RGB", (20, 20), color=(255, 0, 0)).save(
            root / f"train/inpaint/{name}"
        )
        Image.new("L", (20, 20), color=0).save(root / f"train/saliency/{name}")
        Image.new("L", (20, 20), color=255).save(root / f"train/saliency_sub/{name}")
    with (root / "csv/train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["poster_path", "box_elem", "cls_elem"]
        )
        writer.writeheader()
        for name in ["a.png", "b.png"]:
            writer.writerow(
                {"poster_path": name, "box_elem": "[0, 0, 10, 10]", "cls_elem": "1"}
            )
    with (root / "csv/train_sal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["poster_path", "box_elem"])
        writer.writeheader()
        for name in ["a.png", "b.png"]:
            writer.writerow({"poster_path": name, "box_elem": "[0, 0, 20, 20]"})

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"names": ["b.png", "a.png"]}), encoding="utf-8")

    public_dataset = CGBDMOriginalDataset(root, split="train")
    vendor_dataset = CGBDMOriginalDataset(
        root, split="train", name_manifest=manifest, encoding="vendor"
    )

    assert public_dataset.names == ["a.png", "b.png"]
    assert vendor_dataset.names == ["b.png", "a.png"]
    row = vendor_dataset[0]
    assert row["pixel_values"].shape == (4, 384, 256)
    assert row["layout"].shape == (16, 8)
    assert row["layout"][-1, 0] == 1
