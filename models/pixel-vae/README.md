---
language:
  - en
license: "apache-2.0"
library_name: "transformers"
pipeline_tag: "image-feature-extraction"
tags:
  - "pixel-vae"
  - "crello"
  - "layout-generation"
  - "image-feature-extraction"
datasets:
  - "cyberagent/crello"
model-index:
  - name: "creative-graphic-design/pixel-vae-crello"
    results:
      - task:
          type: "image-feature-extraction"
          name: "RGBA image embedding"
        dataset:
          type: "cyberagent/crello"
          name: "Crello v1"
          config: "crello-dataset-v1.zip"
          split: "Crello v1 training completed; full PixelVAE training and test evaluation not claimed"
        metrics:
          - type: "vendor-parity"
            value: "CPU parity passed; all 17 held-out metrics met frozen limits"
            name: "Agreement with the original implementation"
---

<!-- --8<-- [start:card] -->

# Model Card for creative-graphic-design/pixel-vae-crello

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2108.01249&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2108.01249)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=Crello&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/cyberagent/crello)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=tolerance-verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package implements PixelVAE as a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style image encoder and decoder. PixelVAE is the image encoder used by CanvasVAE; its MobileNetV2 maps each 256 × 256 RGBA image element to the 256-value posterior mean consumed by the layout model.

CPU parity compares the original TensorFlow implementation with the PyTorch package on one deterministic, untrained state. A separate CPU check compared 17 values with limits fixed in advance; all 17 passed. State mapping, image decoding, and encoder save/load checks passed.

The original and package each completed one 250-epoch Crello v1 run. Their validation values were `184.70` with deterministic posterior means and `193.7491` with sampled-posterior validation; they are references only and are not comparable. The planned three-seed training and test evaluations for both systems were deferred because the estimate exceeded the available compute limit. PixelVAE training reproduction is not verified. The CanvasVAE comparison uses a shared cache made with untrained original PixelVAE weights. It does not evaluate package PixelVAE training. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/TRAINING.md).

## Model Details

### Model Description

PixelVAE receives four-channel RGBA values at their original 256 × 256 size. A MobileNetV2 encoder produces posterior mean and log-variance vectors with 256 dimensions; `PixelVAEEncoder.encode` exports the posterior mean without sampling. The decoder produces 16 categorical logits for each RGBA channel after removing the lowest four bits of source pixels. Its objective combines alpha-masked RGB reconstruction loss, alpha reconstruction loss, a KL term weighted by 100, and L2 regularization on the variational heads.

- **Developed by:** Kota Yamaguchi.
- **Shared by:** creative-graphic-design.
- **Model type:** content-aware; task: evaluation; conditioning: evaluation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0.

### Model Sources

- **Repository:** [CanvasVAE source](https://github.com/CyberAgentAILab/canvas-vae), pinned at `bc1e2072ba3a253f1b099e8b0c604f6051e787da` for conversion and parity.
- **Paper:** [CanvasVAE, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html).
- **Released weights:** none; the conversion script can convert an original TensorFlow checkpoint or a deterministic untrained initialization.

## Supported Checkpoints

| Checkpoint | Hub ID                                     | Status        |
| ---------- | ------------------------------------------ | ------------- |
| Crello v1  | `creative-graphic-design/pixel-vae-crello` | not-published |

No trained checkpoint is included or published. The seed-0 untrained conversion is only a deterministic parity fixture and is not suitable for semantic retrieval or design decisions.

## Uses

### Direct Use

Use `PixelVAEEncoder` to reproduce original posterior-mean image embeddings or to supply those embeddings to a compatible CanvasVAE document pipeline. The document encoder accepts every element's PNG bytes, regardless of element type. PixelVAE training examples use the original three-type filter (`imageElement`, `maskElement`, and `svgElement`).

### Downstream Use

CanvasVAE may consume embeddings in the order recorded by its document manifest. Use the shared [embedding contract](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/EMBEDDING_CONTRACT.md) for exact PNG decoding, ID ownership, output dtype, and row ordering.

### Out-of-Scope Use

The seed-0 untrained encoder does not provide meaningful visual similarity, semantic understanding, or a quality-ranked embedding space. The package does not render vector documents or clear Crello designs for redistribution.

## Bias, Risks, and Limitations

This port targets the original 256 × 256 RGBA path and rejects other dimensions instead of resizing them. The `crello-dataset-v1.zip` archive and all derived embeddings remain private. Crello design previews may contain third-party artwork; do not redistribute the archive, previews, or embeddings.

### Recommendations

Use an encoder converted from a trained original checkpoint for research applications, preserve the original document element order, and evaluate downstream behavior on the intended data. CPU agreement on an untrained state verifies the conversion path, not trained embedding quality.

## How to Get Started with the Model

Install the package directly from this repository. `laygen` is included because PixelVAE uses the shared vendor-path helper during conversion.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "pixel-vae @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/pixel-vae"
```

For conversion and CPU reproduction, clone the repository and follow [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/REPRODUCING.md). The commands there save an encoder under `.cache/pixel-vae/converted/encoder`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package pixel-vae
uv run --package pixel-vae python
```

```python
from io import BytesIO

import torch
from PIL import Image

from pixel_vae import PixelVAEEncoder, PixelVAEImageProcessor

stream = BytesIO()
Image.new("RGBA", (256, 256)).save(stream, format="PNG")
# After Hub publication: from_pretrained("creative-graphic-design/pixel-vae-crello")
encoder = PixelVAEEncoder.from_pretrained(".cache/pixel-vae/converted/encoder").eval()
inputs = PixelVAEImageProcessor()(stream.getvalue())
with torch.inference_mode():
    embedding = encoder.encode(inputs.pixel_values)

print(embedding.shape, embedding.dtype)
```

```text
Captured output will be added after running the example against the converted CPU checkpoint.
```

## Training Details

### Training Data

| Dataset | Dataset ID                                                               | Notes                                                                                                        |
| ------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Crello  | [`cyberagent/crello`](https://huggingface.co/datasets/cyberagent/crello) | The public v1 source archive is available for original-code parity; keep the archive and derived data local. |

Original-code parity uses [crello-dataset-v1.zip](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip), 2,989,732,284 bytes with SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e`. Keep the archive and derived embeddings private.

### Training Procedure

This package includes a Lightning training module and a `traingen` configuration. The original and package Crello v1 training runs each completed 250 epochs. The package's recorded `193.7491` validation value was sampled-posterior; the original's `184.70` used deterministic posterior means. These values are not comparable. The epoch-249 best checkpoint used sampled-posterior validation, unlike the original's deterministic-mean selection. The exported encoder depends on that choice. The encoder was re-exported on CPU; its SHA-256 is `b7925fe15d057925d2b3c28cebe920aa7bb5b88d9988d29e450610b7ba5c9c22`. The existing cache came from the epoch-250 `last.ckpt` encoder (SHA-256 `dd8ed8e8a85bae96eeba8a2cbf5cfdad7b5e23a5456f856039db63ef279a53f2`), not the best encoder. No cache was regenerated from the best encoder. CanvasVAE uses a separate untrained-original cache. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/TRAINING.md) for the PixelVAE training status.

#### Preprocessing

The training data module reads the original image-record export under `.cache/pixel-vae/data/crello-image/`, which contains `count.json` and train/validation TFRecords with one `image` byte feature per example. Build these files using the commands in [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/TRAINING.md). The transform selects only `imageElement`, `maskElement`, and `svgElement` and deduplicates identical PNG bytes within each split. The image processor decodes each selected PNG to its original 256 × 256 RGBA pixels and scales byte values to float32 `[0, 1]`.

#### Training Hyperparameters

- **Training regime:** batch size 64 for 250 epochs with `KerasAdam`, constant learning rate `1e-4`, KL weight 100, and L2 weight `1e-6`; validation selects minimum reconstruction loss.

Train from the repository root after preparing the filtered image-record export:

```bash
uv sync --package pixel-vae --extra training
uv run --package pixel-vae --extra training traingen fit \
  --config models/pixel-vae/configs/training/pixel_vae_crello.yaml
```

The config sets `default_root_dir` to `.cache/pixel-vae/training-runs/crello` and leaves the checkpoint callback's `dirpath` unset. Lightning writes `best.ckpt` under `lightning_logs/version_N/checkpoints/` beneath that root; use that file for encoder export.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

CPU parity starts with a report-only diagnostic, then uses three fixed, distinct canonical train input pairs for calibration and the canonical test inputs for a fresh held-out check. The full archive audit checks PNG validity and dimensions by element type without writing image data into the repository.

#### Factors

The source and package receive identical PNG bytes decoded to RGBA uint8 values, scaled to float32 `[0, 1]`, and transposed to channels-first order for PyTorch. The test records the source commit, TensorFlow version, seed, encoder-state digest, and stable input IDs.

#### Metrics

Compare maximum absolute error for posterior means, posterior log variances, decoder logits, reconstruction, KL, total loss, gradients, Keras Adam moments, updated parameters, and the three-step trace. Calibration uses three independent CPU processes on distinct train image groups and freezes `max(L, ceil2(1.5 × max))` per metric. Limits round upward to two significant figures while preserving per-metric floor `L`. The per-metric floors are listed in [parity-limit-floors.json](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/parity-limit-floors.json). The held-out run uses a fresh normal-mode CPU process. Image decode and structural S0/S4 gates require exact equality.

### Parity Results

The CPU acceptance passed against one deterministic, untrained TensorFlow state copied to PyTorch. All 17 held-out metrics met their frozen per-metric limits; state mapping and exact decode checks passed. The example limit below illustrates the metric-specific thresholds; [PARITY_PROTOCOL.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/PARITY_PROTOCOL.md) lists every frozen limit alongside the original worker-run held-out maximum absolute error.

| Check                                 |                                                                      Cases | Criterion                                                                                         | Result                                        |
| ------------------------------------- | -------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| S0 state mapping                      |                         1 untrained TensorFlow state; all mapped variables | exact config, one-to-one keys, and bitwise transformed tensors                                    | pass; all checks exact                        |
| S1 forward and losses                 |                                                         6 held-out metrics | each metric within its frozen absolute bound; posterior mean `atol=1.4e-10`                       | pass; 6/6 metrics                             |
| S2 gradients and optimizer            |                                                         6 held-out metrics | each metric within its frozen absolute bound                                                      | pass; 6/6 metrics                             |
| S3 three-step trace                   |                                                         4 held-out metrics | each metric within its frozen absolute bound                                                      | pass; 4/4 metrics                             |
| S4 archive, decode, and serialization | 23,361 documents; 5 element types; 1 encoder round trip; 1 held-out metric | all archive PNGs valid at 256 × 256; exact RGBA decode and round trip; metric within frozen bound | pass; full audit and exact checks; 1/1 metric |

## Reproducibility

Follow [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/REPRODUCING.md) for CPU agreement, conversion, and local loading. It covers the pinned source, Crello inputs, three-run calibration, and held-out validation. [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/TRAINING.md) gives the training procedure and status.

## Technical Specifications

### Model Architecture and Objective

The encoder uses Keras MobileNetV2 topology with a four-channel first convolution, global average pooling, and two dense variational heads. The decoder projects 256 latent values to an 8 × 8 × 32 grid, applies five transposed-convolution upsampling layers and one output convolution, then reshapes logits to `[batch, 256, 256, 4, 16]`.

### Compute Infrastructure

#### Hardware

The 250-epoch package training used an A100 for epochs 1–227 and a Tesla V100-SXM2-32GB for the continuation.

#### Software

The model uses PyTorch and `transformers`. Conversion and source comparison require TensorFlow CPU 2.15.1. See the package `pyproject.toml` for workspace dependencies.

## Citation

```bibtex
@inproceedings{yamaguchi2021canvasvae,
  title={CanvasVAE: Learning to Generate Vector Graphic Documents},
  author={Yamaguchi, Kota},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

<!-- --8<-- [end:card] -->
