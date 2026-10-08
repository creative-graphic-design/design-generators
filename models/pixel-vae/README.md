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
          config: "pinned original GCS archive"
          split: "CPU implementation parity only; no trained-checkpoint evaluation"
        metrics:
          - type: "vendor-parity"
            value: "CPU S0-S4 evidence pending"
            name: "Agreement with the original implementation"
---

<!-- --8<-- [start:card] -->

# Model Card for creative-graphic-design/pixel-vae-crello

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2108.01249&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2108.01249)
![venue](https://img.shields.io/static/v1?label=venue&message=ICCV+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=Crello&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/cyberagent/crello)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=not-run&color=lightgrey&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports the image encoder and decoder from [CanvasVAE](https://openaccess.thecvf.com/content/ICCV2021/html/Yamaguchi_CanvasVAE_Learning_To_Generate_Vector_Graphic_Documents_ICCV_2021_paper.html), the ICCV 2021 vector-document VAE, into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style package. Its MobileNetV2 encoder maps 256 × 256 RGBA previews to the 256-dimensional posterior mean used by CanvasVAE document records. The package is a separate workspace member so it can be converted, serialized, and parity-checked independently before the document model consumes it.

This implementation task covers conversion and CPU agreement on one deterministic, untrained TensorFlow state. The CPU S0–S4 run is not yet recorded; this package makes no trained-checkpoint or embedding-quality claim. See [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/TRAINING.md) for scope and evidence and [EMBEDDING_CONTRACT.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/EMBEDDING_CONTRACT.md) for the shared Crello input and output contract.

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

This port targets the original 256 × 256 RGBA path and rejects other dimensions instead of resizing them. The pinned Crello v1 archive and all derived embeddings remain private. Crello design previews may contain third-party artwork; do not redistribute the archive, previews, or embeddings.

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

| Dataset | Dataset ID                                                               | Notes                                                                                                                                                 |
| ------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Crello  | [`cyberagent/crello`](https://huggingface.co/datasets/cyberagent/crello) | The original implementation task pins the separate Crello v1 GCS archive; its data and derived embeddings are private and are not in this repository. |

### Training Procedure

This package does not include a package-local training loop or a trained checkpoint. The original PixelVAE code supports training; the current change implements checkpoint conversion and CPU parity only. User scope excludes full training and S5 evaluation.

#### Preprocessing

Decode every document-stage `image_bytes` PNG to RGBA at its original 256 × 256 dimensions, then map byte values to float32 `[0, 1]`. The encoder consumes all element types for document construction. Original PixelVAE training records select only `imageElement`, `maskElement`, and `svgElement`.

#### Training Hyperparameters

- **Training regime:** not retrained in this task; the original configuration uses batch size 64, 250 epochs, Adam at 1e-4, KL weight 100, and L2 weight 1e-6.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

CPU parity uses a fixed set of six unique train IDs for calibration and six unique test IDs for held-out validation, selected from the pinned Crello v1 archive in canonical order. The full archive audit checks PNG validity and dimensions by element type without writing image data into the repository.

#### Factors

The source and package receive identical PNG bytes decoded to RGBA uint8 values, scaled to float32 `[0, 1]`, and transposed to channels-first order for PyTorch. The test records the source commit, TensorFlow version, seed, encoder-state digest, and stable input IDs.

#### Metrics

Compare maximum absolute error for posterior means, posterior log variances, decoder logits, reconstruction/KL/total loss, gradients, Keras Adam moments, updated parameters, and a three-step short trace. Calibration uses three independent CPU processes and freezes `ceil2(1.5 × max)` per metric, rounding upward to two significant figures. The held-out run uses a fresh normal-mode CPU process; image decode and structural S0/S4 gates require exact equality.

### Parity Results

not run: the calibrated CPU comparison has not yet been executed. Replace this table with measured maxima, frozen bounds, and pass counts before the pull request is ready for review.

| Check                                 |                                Cases | Criterion                                     | Result  |
| ------------------------------------- | -----------------------------------: | --------------------------------------------- | ------- |
| S0 state mapping                      |                              pending | exact transformed TensorFlow state            | pending |
| S1 forward and losses                 |                            6 metrics | calibrated per-metric bound                   | pending |
| S2 gradients and optimizer            |                            5 metrics | calibrated per-metric bound                   | pending |
| S3 three-step trace                   |                            3 metrics | calibrated per-metric bound                   | pending |
| S4 archive, decode, and serialization | full archive plus encoder round trip | exact source dimensions and output round trip | pending |

## Reproducibility

Reproduce agreement with the original TensorFlow implementation using [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/REPRODUCING.md), which covers the pinned source, private Crello archive, three-run CPU calibration, held-out validation, conversion, and local loading. No trained-checkpoint reproduction is claimed.

## Technical Specifications

### Model Architecture and Objective

The encoder uses Keras MobileNetV2 topology with a four-channel first convolution, global average pooling, and two dense variational heads. The decoder projects 256 latent values to an 8 × 8 × 32 grid, applies five transposed-convolution upsampling layers and one output convolution, then reshapes logits to `[batch, 256, 256, 4, 16]`.

### Compute Infrastructure

#### Hardware

The requested parity path is CPU-only. This task does not run GPU training or S5 evaluation.

#### Software

The model uses PyTorch and [`🤗transformers`](https://huggingface.co/docs/transformers/index). Conversion and source comparison require TensorFlow CPU 2.15.1. See the package `pyproject.toml` for workspace dependencies.

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
