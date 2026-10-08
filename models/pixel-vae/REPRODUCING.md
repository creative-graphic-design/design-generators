# Reproducing PixelVAE CPU agreement

Workflow order: download and audit the private source, generate references from the deterministic untrained state, run CPU parity checks, convert a state, then smoke-test local loading.

These commands compare the package with the pinned [CanvasVAE TensorFlow source](https://github.com/CyberAgentAILab/canvas-vae) and convert its weights. Run them from the repository root in a CPU environment with enough temporary disk for the private 2.99 GB Crello v1 archive and extracted TFRecords. Keep source data, checkpoints, and parity outputs under `.cache/pixel-vae/` during the run. Do not commit them or publish them publicly; persist required run evidence only in the project's private artifact repository.

The package is a separate workspace member because its encoder and checkpoint conversion need an independently loadable interface and CPU parity result before CanvasVAE consumes its embeddings. The [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md) defines the shared stage vocabulary; [PARITY_PROTOCOL.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/PARITY_PROTOCOL.md) fixes PixelVAE's CPU metrics and bounds before the run.

## Install and initialize the source

```bash
git submodule update --init vendor/canvas-vae
uv sync --package pixel-vae --extra vendor --extra parity
```

The `vendor` extra supplies TensorFlow CPU 2.15.1. The `parity` extra supplies the shared Keras-compatible optimizer and trace utilities. The submodule is read-only and pinned at `bc1e2072ba3a253f1b099e8b0c604f6051e787da`.

## Download and audit the private Crello v1 source

The archive is available from the original public GCS URL. Its 2,989,732,284 bytes and SHA-256 are pinned to issue #31. Keep it and all extracted records private.

```bash
mkdir -p .cache/pixel-vae
curl --fail --location \
  https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip \
  --output .cache/pixel-vae/crello-dataset-v1.zip
test "$(stat -c %s .cache/pixel-vae/crello-dataset-v1.zip)" -eq 2989732284
sha256sum .cache/pixel-vae/crello-dataset-v1.zip
uv run --package pixel-vae --extra vendor \
  models/pixel-vae/scripts/audit_crello_png_dimensions.py \
  --archive .cache/pixel-vae/crello-dataset-v1.zip
```

The audit verifies the archive digest, checks all source document counts, decodes every element PNG to RGBA, and reports malformed or non-256 × 256 examples by element type. Any such entry fails the CPU acceptance suite and must be reported with its type and source location.

## Convert the deterministic untrained state

The converter defaults to TensorFlow seed 0 and writes both the full model and an independently loadable encoder. Supply `--checkpoint` only for a local original TensorFlow checkpoint prefix.

```bash
CUDA_VISIBLE_DEVICES="" TF_ENABLE_ONEDNN_OPTS=0 \
  uv run --package pixel-vae --extra vendor \
  models/pixel-vae/scripts/convert_original_checkpoint.py \
  --output-dir .cache/pixel-vae/converted
```

For a provided original checkpoint, add `--checkpoint <local-checkpoint-prefix>`. The script writes `conversion.json`, `config.json`, and safetensors weights under `.cache/pixel-vae/converted/`; encoder-only files are under `.cache/pixel-vae/converted/encoder/`.

## Calibrate three independent CPU runs

Each command below starts a fresh pytest process. The source state, seed, input IDs, and metrics are fixed by [PARITY_PROTOCOL.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/PARITY_PROTOCOL.md).

```bash
for repeat in 1 2 3; do
  CUDA_VISIBLE_DEVICES="" TF_ENABLE_ONEDNN_OPTS=0 \
    PARITY_REQUIRE=1 PIXELVAE_PARITY_MODE=calibration PIXELVAE_PARITY_REPEAT="$repeat" \
    uv run --package pixel-vae --extra vendor --extra parity --with pytest \
    pytest models/pixel-vae/tests/vendor_parity -m vendor_parity -q
done
uv run --package pixel-vae --extra parity \
  models/pixel-vae/scripts/calibrate_cpu_limits.py
```

Calibration writes three raw JSON records and the frozen `ceil2(1.5 × max)` bounds under `.cache/pixel-vae/parity/calibration/`. The limit utility rejects missing metrics, mismatched source-state digests or input IDs, differing configurations or TensorFlow versions, and repeats from the same process.

## Run the fresh held-out CPU check

```bash
CUDA_VISIBLE_DEVICES="" TF_ENABLE_ONEDNN_OPTS=0 \
  PARITY_REQUIRE=1 PIXELVAE_PARITY_MODE=heldout \
  uv run --package pixel-vae --extra vendor --extra parity --with pytest \
  pytest models/pixel-vae/tests/vendor_parity -m vendor_parity -q
```

The test uses held-out test IDs, requires the frozen calibration file, and writes `.cache/pixel-vae/parity/heldout.json`. It checks strict state conversion, TensorFlow/PyTorch forward and update traces, every Crello v1 image type, exact PNG preprocessing, the encoder contract shape and dtype, and local serialization.

## Run package unit tests and load the encoder

```bash
uv run --package pixel-vae --with pytest --with pytest-cov pytest \
  models/pixel-vae/tests -m "not vendor_parity and not integration" \
  --cov=pixel_vae --cov-report=term-missing --cov-fail-under=90
uv run --package pixel-vae python -c 'from pixel_vae import PixelVAEEncoder; encoder = PixelVAEEncoder.from_pretrained(".cache/pixel-vae/converted/encoder"); print(encoder.config.latent_dim)'
```

The encoder returns the float32 posterior mean with shape `[batch, 256]`. Use `eval()` and `torch.inference_mode()` for deterministic embedding export. The model's posterior log-variance is available from its forward output for parity diagnostics.
