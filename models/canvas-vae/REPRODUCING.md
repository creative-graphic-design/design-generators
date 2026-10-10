# Reproducing CanvasVAE Agreement Checks

These commands rerun the agreement checks between this package and the [original CanvasVAE TensorFlow code](https://github.com/CyberAgentAILab/canvas-vae): RICO preparation, reference traces from the original trainer, the parity suite, checkpoint conversion, and a local `from_pretrained` smoke test. Run them from the repository root after `git submodule update --init vendor/canvas-vae`. Generated data, traces, and checkpoints stay under `.cache/canvas-vae/`.

Workflow order: download and prepare RICO, generate original-code references, run the parity suite, convert a checkpoint, then smoke-test `from_pretrained` local loading.

## Crello v1 data contract

This section is for maintainers reproducing the Crello v1 data path. It pins the input schema and deterministic IDs used to compare the original preprocessing with the package loader; it does not claim a Crello model or trained-encoder result.

Use the [original Crello v1 archive](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip) from vendor commit [`bc1e2072ba3a253f1b099e8b0c604f6051e787da`](https://github.com/CyberAgentAILab/canvas-vae/tree/bc1e2072ba3a253f1b099e8b0c604f6051e787da). The verified archive is 2,989,732,284 bytes with SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e`; `count.json` lists 18,768 train, 2,315 validation, and 2,278 test documents. The archive and every derived image or embedding remain private.

The original [Crello document specification](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/canvas-vae/canvasvae/data/crello-document-spec.yml#L1-L99) defines context fields `id`, `group`, `format`, `category`, `canvas_width`, `canvas_height`, and `length`; `id` is retained as identity metadata and is not a model input. Sequence fields are `type`, `left`, `top`, `width`, `height`, `color` (three RGB int64 channels), `opacity`, and `image_embedding` (float32[256]). The source parser gives scalar columns shape `[1]` by default, so scalar context fields use `[batch, 1]` and scalar sequence fields use `[batch, length, 1]`; color and image embeddings keep `[batch, length, 3]` and `[batch, length, 256]` ([original DataSpec](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/canvas-vae/canvasvae/data/spec.py#L165-L183)). Geometry uses 64 bins over [0, 1], opacity uses 8 bins over [0, 1], and each color channel uses 16 bins over [0, 255]. String and integer vocabularies retain the source `vocabulary.json` ordering and lookup settings.

`document_id` is `crello-v1/{split}/{source id}`, where `source id` is the original context `id`; split names are `train`, `val`, and `test`. `image_id` is `crello-v1/{split}/{sha256(image_bytes)}` over the exact PNG bytes. Equal IDs therefore have byte-identical encoder inputs. This matches the original image transform's byte-based `Distinct` identity within each split ([image transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_image.py#L22-L60)). The PixelVAE training population is exactly the image transform's `imageElement`, `maskElement`, and `svgElement` subset. The document transform preserves each source document ID and replaces every element's ordered `image_bytes` sequence with one embedding, with no element-type filter ([document transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_document.py#L24-L49)). Therefore the encoder input and posterior-mean fixture population is every distinct PNG referenced by any element type in the document stage; this is a superset of the PixelVAE training population. Document order is split order `train`, `val`, `test`, then UTF-8 bytewise `document_id`; element order stays as stored in the SequenceExample. Traverse documents in that order, deduplicate by `image_id` before encoding, and write one fixture row at the first occurrence of each ID. Encode unique images in that canonical order with a fixed batch size of 32; repeat the last real image to pad the final batch and discard its padded outputs. The original transform encodes each document's elements as a batch whose size varies by document, so TensorFlow CPU kernels can produce slightly different values for identical PNG bytes at different batch shapes. Those legacy duplicate differences are report-only; the fixture always stores the one canonical fixed-batch output per ID.

For a batch padded to its longest document, `element_mask[b, t]` is true exactly when `t < length[b]`. Sequence-field masks for `type`, geometry, and `opacity` equal `element_mask`; `color_mask` additionally requires type `textElement` or `coloredBackground`; `image_embedding_mask` additionally requires type `svgElement`, `imageElement`, or `maskElement`, as declared by the source spec. Padding has false masks. Non-applicable values remain present in the parsed source record; masks identify where each field contributes a loss.

The canonical private fixture is an `.npy` array of contiguous little-endian float32 values with shape `[N, 256]`, one posterior-mean row per unique split-scoped `image_id`. A neighboring `manifest.json` records the format version, source archive SHA-256, frozen TensorFlow encoder-state SHA-256, ordered `image_id` and row pairs, per-row embedding SHA-256 values, and the `.npy` file SHA-256; `manifest.sha256` hashes the manifest itself. The values come from the deterministic untrained TensorFlow encoder state used for the Crello fixture; the fixture stores no source images. The original document transform decodes PNGs as RGBA before encoding ([document transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_document.py#L40-L49)). The v1 guide describes each element preview as 256×256 pixels; decode it without resizing, cropping, or compositing and reject any decoded shape other than 256×256×4 ([dataset schema](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/docs/crello-dataset.md#L74-L90)). The fixture builder validates each cached PNG's SHA-256 against its `image_id`, encodes that unique PNG once in the fixed batch, and pairs output row `i` with manifest entry `i`; the document processor looks up the split-scoped ID and restores original element order.

CanvasVAE has no released weights, so agreement is checked on the training path. The stages S0-S4 are defined in the [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md); [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) records their results.

The `vendor` extra installs CUDA-enabled TensorFlow 2.15.1 and Apache Beam for the original-code workflow. The separate `convert` extra installs `tensorflow-cpu==2.15.1`, pinned Apache Beam, PyYAML, and the shared parity helpers for the Crello CPU data/reference commands below and original TensorFlow checkpoint conversion. It installs no NVIDIA CUDA wheels, so PyTorch keeps the NCCL library that matches its own CUDA wheel when the Crello CPU job imports both frameworks. A Lightning `last.ckpt` converts without a TensorFlow extra. Current training configs keep validation-best weights in `best.ckpt`; a separate unmonitored callback with `save_top_k: 1` retains one rolling checkpoint under Lightning’s epoch/step name and updates `last.ckpt` at configured checkpoint events. The S5 executor verifies the configured final epoch and global step before copying or uploading that file. Use `vendor` and `convert` in separate environments. Original GPU training and evaluation set `NVIDIA_TF32_OVERRIDE=0` and `TF_ENABLE_ONEDNN_OPTS=0`; the currently verified parity setup uses a CPU-only runtime with `CUDA_VISIBLE_DEVICES=""`.

The pinned [Crello document transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_document.py#L40-L49) calls `tf.debugging.assert_all_finite(embeddings)` without a message. In [TensorFlow 2.15.1](https://github.com/tensorflow/tensorflow/blob/v2.15.1/tensorflow/python/ops/numerics.py#L46-L72), `verify_tensor_all_finite_v2(x, message, name=None)` requires `message`; the unmodified vendor call therefore raises `TypeError` before reaching `check_numerics`. `run_crello_reference.py` installs `traingen_parity.tensorflow_compat.install_assert_all_finite_compat`, which supplies a default message while retaining TensorFlow's finite-value assertion; non-finite embeddings still raise `tf.errors.InvalidArgumentError`. The vendor commit remains unchanged.

The Crello vendor model calls its vector metric layer on every VAE call, and that layer concatenates per-field sparse accuracy and sequence metrics. TensorFlow 2.3 squeezes the target only when target and prediction ranks match; the Keras 2.15 public metric also squeezes a trailing singleton from its result, so context targets shaped [batch, 1] with logits shaped [batch, 1, classes] become rank 1 while sequence metrics remain rank 2 ([TensorFlow 2.3 implementation](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L3271-L3309), [Keras 2.15 implementation](https://github.com/keras-team/tf-keras/blob/v2.15.0/tf_keras/metrics/accuracy_metrics.py#L433-L465), [vendor metric concatenation](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/canvas-vae/canvasvae/models/metrics.py#L142-L195)). The [Crello comparison harness](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/scripts/compare_crello_training.py) installs the shared parity compatibility helper before building the reference VAE to restore the TensorFlow 2.3 output rank; package model code does not use this parity-only shim. Its S0 report records each vendor metric's target, prediction, and output shapes immediately before concatenation.

## Reproduce Crello data checks

Run these checks on a high-RAM CPU Colab because the original Beam Direct Runner holds its working data in process memory. The scripts use the local archive and private output directories; the embedding fixture includes derived data and must stay private. The dimension audit reports every non-256×256 image with its element types and stops before declaring the contract verified.

```bash
uv sync --package canvas-vae --extra convert
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/download_crello.py --archive .cache/canvas-vae/crello/crello-dataset-v1.zip --extract-to .cache/canvas-vae/crello/source
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/prepare_crello.py --source-dir .cache/canvas-vae/crello/source --output-dir .cache/canvas-vae/crello/package-run-1
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/prepare_crello.py --source-dir .cache/canvas-vae/crello/source --output-dir .cache/canvas-vae/crello/package-run-2
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_cache_runs.py --first-run .cache/canvas-vae/crello/package-run-1 --second-run .cache/canvas-vae/crello/package-run-2
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/run_crello_reference.py --vendor-root vendor/canvas-vae --source-dir .cache/canvas-vae/crello/source --encoder-path "$ENCODER_STATE_PATH" --output-dir .cache/canvas-vae/crello/reference
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_beam_runs.py --first-run .cache/canvas-vae/crello/reference/run-1 --second-run .cache/canvas-vae/crello/reference/run-2
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/write_crello_fixture.py --reference-run .cache/canvas-vae/crello/reference/run-1 --package-dir .cache/canvas-vae/crello/package-run-1 --fixture-dir .cache/canvas-vae/crello/fixture --encoder-path "$ENCODER_STATE_PATH" --encoder-state-sha256 "$ENCODER_STATE_SHA256"
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_s4.py --vendor-root vendor/canvas-vae --reference-run .cache/canvas-vae/crello/reference/run-1 --package-dir .cache/canvas-vae/crello/package-run-1 --fixture-dir .cache/canvas-vae/crello/fixture
```

Set `ENCODER_STATE_PATH` and `ENCODER_STATE_SHA256` to the frozen untrained TensorFlow encoder state used for the canonical Crello fixture and its SHA-256. The reference command runs both original transforms twice with Beam 2.76's in-process `FnApiRunner`; its comparison checks exact serialized document fields and filtered image IDs under stable IDs. If a single fresh reference run is needed after the independent-run stability check has passed, run the command with `--run-indices 2`; it still generates both original image and document stages together. The package replay comparison hashes every prepared file, including every cached PNG. The fixture command logs the maximum absolute difference between each repeated image's legacy document-batch embedding and its first occurrence, including both document IDs, element indexes, and original batch sizes; it does not fail on that difference. `compare_crello_s4.py` creates and sets up `CanvasVAECrelloDataModule`, iterates the production train, validation, and test DataLoaders, and compares each ordered batch's IDs, fields, masks, and fixture embedding rows with vendor records. It replays each split deterministically through a second production DataModule and compares batch IDs and values exactly. It uses original records for source fields and the canonical fixture for `image_embedding`, because legacy per-document encoder batches can differ for repeated IDs. Keep the source archive, package runs, references, and fixture under private storage; do not commit or publish them.

## Prepare data

```bash
uv sync --package canvas-vae --extra training --extra vendor
uv run --package canvas-vae models/canvas-vae/scripts/download_rico.py
uv run --package canvas-vae models/canvas-vae/scripts/prepare_rico.py
```

`download_rico.py` fetches the RICO semantic-annotation archive (MD5 `5dd3372e2d99b342958136e394d4de79`) to `.cache/canvas-vae/data/semantic_annotations.zip`. `prepare_rico.py` writes the package splits to `.cache/canvas-vae/data/rico/`.

## Generate original-code references

```bash
export NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=""
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py beam
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py trace
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py trace --output-dir .cache/canvas-vae/reference/trace-repeat
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py stream
```

`beam` runs the original Apache Beam preprocessing into `.cache/canvas-vae/original/data/rico/`. `trace` exports static configuration, initial weights, the first 50 training batches with their posterior noise, a fixed-batch forward and one-step trace, and a 50-step trajectory to `.cache/canvas-vae/reference/trace/`; the repeated trace measures the original's run-to-run envelope. `stream` dumps every original record and the first 90 training, 6 validation, and 6 test batches of the original data streams to `.cache/canvas-vae/reference/stream/`.

## Run the parity suite

```bash
PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -rs
```

`PARITY_REQUIRE=1` turns missing references into failures. Measured differences are written to `.cache/canvas-vae/reference/reports/`.

## Convert a checkpoint and smoke-test loading

An original TensorFlow checkpoint prefix such as `initial.ckpt` or `final.ckpt` needs the `convert` extra. A Lightning `last.ckpt` file uses PyTorch loading and needs no TensorFlow extra. For a full run, use only a `last.ckpt` that passes the executor's final-epoch and global-step check.

```bash
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/convert_original_checkpoint.py --checkpoint .cache/canvas-vae/reference/trace/initial/initial.ckpt --vocabulary .cache/canvas-vae/original/data/rico/vocabulary.json --output-dir .cache/canvas-vae/converted/initial
uv run --package canvas-vae models/canvas-vae/scripts/smoke_from_pretrained.py --checkpoint .cache/canvas-vae/converted/initial
```

For a Lightning checkpoint from `traingen fit`, run the same converter with `uv run --package canvas-vae` and no TensorFlow extra.

## Crello model agreement checks (CPU S0–S3; S4 loader replay)

The CPU S0–S3 command compares the package Crello model with the pinned TensorFlow vendor on the prepared splits and shared posterior-mean fixture. It covers static configuration, forward outputs and losses, gradients, optimizer state, and batch-normalization statistics. In the latest run pinned to package commit `95ddde1a9f1b9ef221e06350f6b054574c4fdec5`, S0 passed; three independent calibration children froze limits before three independent held-out children. All six children exited 0, but the aggregate held-out gate failed because the masked `s2_updated_parameters` comparison reached `1.94486e-5` against `1.9e-5` for `encoder.blocks.0.norm2.bias` on batch 0; S0–S3 parity is not confirmed. S4 production DataModule/DataLoader replay passed exact equality for train, validation, and test, including deterministic replay. These checks do not run full training or evaluate a trained checkpoint, so no full-run or trained-model quality result is claimed.

S0 needs only the `count.json` and `vocabulary.json` files from the original run-1 small reports; it does not need document splits or the embedding fixture. It compares vendor and package vocabularies, field shapes and class counts, type-conditioned masks, parameter mapping and counts, and the package state after copying the vendor's seeded initialization. Run it from the repository root after initializing `vendor/canvas-vae`:

```bash
CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_training.py s0 --data-dir .cache/canvas-vae/crello/original-run-1/document --report-dir .cache/canvas-vae/reference/crello-model
```

For the S1–S3 CPU comparison, make the prepared canonical splits available under `package-run-1` and publish `manifest.json`, `manifest.sha256`, and `posterior_means.npy` in one fixture directory. Pass that directory's Hugging Face `tree` URL, the posterior-array SHA-256, and the manifest SHA-256 to one command. The runner downloads those three files, verifies both supplied digests and the manifest sidecar, loads image IDs from the manifest, and checks their canonical order against the package splits. It writes `calibration-plan.json` with the fixed batch selection and limit formula before starting three independent calibration processes from the same seeded initial state, freezes each metric's maximum-derived limit, and then runs the canonical test batches. Reports are written before assertions. An optional restricted token file can be supplied with `--fixture-token-file`, or its path can be passed through `CANVAS_VAE_FIXTURE_TOKEN_FILE`; the runner reads and deletes that file after the Hub download.

```bash
CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_training.py run --data-dir .cache/canvas-vae/crello/package-run-1 --fixture-hub-location "$CRELLO_FIXTURE_HUB_LOCATION" --fixture-array-sha256 "$CRELLO_FIXTURE_ARRAY_SHA256" --fixture-manifest-sha256 "$CRELLO_FIXTURE_MANIFEST_SHA256" --report-dir .cache/canvas-vae/reference/crello-model
```

The `run` command executes S0–S3 on CPU and evaluates canonical held-out batches only after calibration limits are frozen; it does not run S4 or S5. S4 uses `models/canvas-vae/scripts/compare_crello_s4.py` to iterate the production DataModule and compare the three deterministic loader streams. To rerun individual model phases with an already downloaded fixture, use the marked `vendor_parity` test below; it requires local prepared splits and fixture files.

```bash
uv sync --package canvas-vae --extra training --extra convert
PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra convert --with pytest pytest models/canvas-vae/tests/vendor_parity/test_canvas_vae_crello_training_parity.py -m "vendor_parity and training" -k crello_s0_s3 -rs
```
