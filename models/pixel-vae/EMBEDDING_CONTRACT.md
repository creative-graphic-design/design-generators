# PixelVAE encoder contract

Status: agreed encoder-side contract for Crello v1 coordination. The data package owns record and fixture production; this document pins the encoder behavior and shared input/output assumptions.

## Ownership

The PixelVAE package owns PNG decoding into model input tensors, posterior-mean inference, and the deterministic untrained TensorFlow state used for CPU parity. The Crello data package owns source-record selection, stable image IDs, masks, embedding fixture serialization, its ordered manifest, and SHA-256 rules. The package encoder accepts every document-stage PNG regardless of element type; the original image-record transform filters PixelVAE training examples to `imageElement`, `maskElement`, and `svgElement`.

## Input

Each Crello document-stage `image_bytes` value is a PNG preview of one visual element, of any element type. The source dataset guide describes these previews as 256 × 256 pixels. The original document transform decodes every element's PNG with `tf.io.decode_png(image_bytes, channels=4)`, producing an RGBA `uint8` image. The original encoder then applies `tf.image.convert_image_dtype(image, tf.float32)`, which maps byte values to float32 values in `[0, 1]` before MobileNetV2. The model's configured input shape is `(256, 256, 4)`.

The pinned Crello v1 archive audit found no invalid or non-256 × 256 document PNGs; `textElement` PNGs have one unique SHA-256 per split and the same digest across train, val, and test.

The package processor must preserve every decoded channel value, use RGBA channel order, and avoid resizing, cropping, alpha compositing, or other normalization. It must reject decoded images whose shape is not 256 × 256 × 4 with a clear error. Acceptance requires pixel-for-pixel equality with TensorFlow's decode for the parity fixtures. This validation applies to every document-stage element, including types whose embedding loss is masked later.

The package model receives batched float32 tensors in `[0, 1]` with shape `[batch, 4, 256, 256]`. This channels-first package interface is transposed from the original TensorFlow `[batch, 256, 256, 4]` tensor without changing values.

## Encoder output

The encoder returns the variational head's posterior mean without sampling, with dtype float32 and shape `[batch, 256]`. Do not normalize or quantize these values. The matching log-variance is available for parity diagnostics but is not part of the embedding fixture.

For a batch with stable image IDs, output row `i` belongs to input ID `i`; the encoder preserves the supplied order and does not create or rewrite IDs. A stable ID is `crello-v1/{split}/{sha256(image_bytes)}`, where the SHA-256 is over the exact PNG bytes and split is `train`, `val`, or `test`. The private fixture is a contiguous little-endian float32 array shaped `[N, 256]`, with one posterior-mean row per unique split-scoped image ID. Its ordered manifest records the format version, source archive digest, encoder-state digest, each ID and row index, each row SHA-256, and the array file SHA-256; serialization and hashing are owned by the data package. Canonical traversal is split order `train`, `val`, `test`, then UTF-8 bytewise `document_id`, preserving source element order within each document; a unique image ID's fixture row is placed at its first occurrence in that traversal. Repeated IDs reuse that row, and document output restores original element order.

## CPU parity state

Use one deterministically initialized, untrained TensorFlow PixelVAE state with the fixed production configuration, copy that state into the PyTorch package through the verified weight map, and use the same decoded RGBA examples in both encoders. The proposed seed is `0`; the run record must include the TensorFlow version, seed, source commit, state digest, and input IDs. The encoder-side tolerance is calibrated independently from CanvasVAE's tolerances.

This fixture checks decoding, weight mapping, posterior means, and data flow. It does not measure embedding quality from trained weights. It includes every document-stage element type; the source `generate_crello_image` transform's three-type filter selects PixelVAE training examples only. The later CanvasVAE loss mask controls which embeddings contribute to loss, not which elements are encoded or present in document records.

## Source behavior

- Original training image selection and PNG record extraction: `vendor/canvas-vae/src/preprocess/preprocess/transforms/generate_crello_image.py`.
- Original document-stage all-element encoding: `vendor/canvas-vae/src/preprocess/preprocess/transforms/generate_crello_document.py`.
- Original encoder input conversion and posterior mean: `vendor/canvas-vae/src/pixel-vae/pixelvae/model.py`.
- Original exported encoder selects the first output of `model.encoder`, which is `z_mean`: `vendor/canvas-vae/src/pixel-vae/pixelvae/train.py`.
