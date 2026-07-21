---
license: apache-2.0
library_name: transformers
tags:
  - layout-generation
  - masked-document-modeling
datasets:
  - cyberagent/crello
  - creative-graphic-design/Rico
---

# Flex-DM

Flex-DM ports the MFP masked multi-field document model from "Towards Flexible Multi-modal Document Models" to a Transformers-style package. The package exposes `FlexDmForMaskedDocumentModeling` for the PyTorch forward pass and `FlexDmPipeline` for layout-level completion, refinement, and Crello image/text feature infilling.

```python
import torch
from flex_dm import FlexDmPipeline

pipe = FlexDmPipeline.from_pretrained("creative-graphic-design/flex-dm-crello-explicit-ft")
out = pipe(
    condition_type="completion",
    bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
    labels=torch.tensor([[0]]),
    mask=torch.tensor([[True]]),
    feature_group="pos",
    seed=0,
)
```

## Supported Checkpoints

The original GCS weight archives contain one released checkpoint variant per dataset: `ours-exp-ft` for Crello and `ours-exp-ft` for RICO. The intended Hub ids are `creative-graphic-design/flex-dm-crello-explicit-ft` and `creative-graphic-design/flex-dm-rico-explicit-ft`.

## Datasets

Crello ordinary loading uses `cyberagent/crello`; exact parity uses the vendor GCS TFRecords, vocabulary, and checkpoints because those artifacts define the released preprocessing. RICO ordinary loading uses `creative-graphic-design/Rico` with `name="ui-screenshots-and-hierarchies-with-semantic-annotations"` only after vocabulary-order compatibility is confirmed.

## Evaluation

### Parity Results

| checkpoint | dataset | task cases | exact items | tolerance items | result |
| --- | --- | ---: | ---: | ---: | --- |
| `ours-exp-ft` | crello | 0 | 0 | 0 | not run |
| `ours-exp-ft` | rico | 0 | 0 | 0 | not run |

## Reproducibility

Run these commands to reproduce the original-implementation agreement checks once the GCS assets are available in the local cache.

```bash
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset all --assets weights
uv run --package flex-dm --extra data python models/flex-dm/scripts/download_original_assets.py --output-dir .cache/flex-dm/original --dataset crello --assets data
uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/inspect_tf_checkpoint.py --job-dir .cache/flex-dm/original/weights/crello/crello/ours-exp-ft --output-json .cache/flex-dm/reports/crello-ours-exp-ft-tf-vars.json
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm --extra vendor-tf python models/flex-dm/scripts/generate_reference_outputs.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/goldens/crello-ours-exp-ft --seed 0 --tasks elem,pos,attr,img,txt --num-iter 1,4
CUDA_VISIBLE_DEVICES=1 uv run --package flex-dm pytest models/flex-dm/tests/vendor_parity -m vendor_parity
uv run --package flex-dm --extra convert python models/flex-dm/scripts/convert_original_checkpoint.py --dataset crello --variant ours-exp-ft --asset-dir .cache/flex-dm/original --output-dir .cache/flex-dm/converted/flex-dm-crello-explicit-ft
uv run --package flex-dm python models/flex-dm/scripts/smoke_from_pretrained.py --checkpoint-dir .cache/flex-dm/converted/flex-dm-crello-explicit-ft
```

The `vendor-tf` extra uses a Python 3.11-compatible TensorFlow CPU runtime for checkpoint inspection and metadata generation. Exact TensorFlow 2.8/CUDA 11.3 parity remains a separate original-environment run before accepting final numeric parity.

## License

This package follows the repository license. The original Flex-DM implementation is Apache-2.0. Crello dataset licensing follows the Hugging Face metadata for `cyberagent/crello`; redistribution of converted weights remains blocked until the project-wide Hub publishing pass.

## Citation

```bibtex
@inproceedings{inoue2023flexdm,
  title = {Towards Flexible Multi-modal Document Models},
  booktitle = {CVPR},
  year = {2023}
}
```

Original implementation: CyberAgentAILab/flex-dm.
