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

The archive is [crello-dataset-v1.zip](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip), 2,989,732,284 bytes with SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e`. Keep the archive and all extracted records private.

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

## Run CPU diagnostics, calibration, and held-out parity

The runner first exercises the input selectors on a synthetic TFRecord archive, then runs the PixelVAE and `traingen` member tests with coverage, downloads and audits the pinned Crello v1 archive, and plans every diagnostic, calibration, and held-out input selection before constructing a model. The plan step writes the input sidecars and exits on any short list or missing per-type occurrence. The runner then performs one report-only diagnostic, three fresh calibration processes using fixed distinct train input groups, freezes limits, and starts a fresh held-out process. The selection rule and per-metric lower bounds are fixed in [PARITY_PROTOCOL.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/pixel-vae/PARITY_PROTOCOL.md).

```bash
PIXELVAE_PARITY_DIR=.cache/pixel-vae/parity/run-<full-commit-sha> \
  bash models/pixel-vae/scripts/run_cpu_parity_acceptance.sh
```

The runner stores the archive audit, plan manifest, diagnostic report, three pre-comparison input selections, calibration records, frozen limits, and held-out report beneath the selected parity directory. The limit utility checks each input-selection sidecar against its metric record and rejects repeated S1 or training IDs, unexpected indexes, mismatched source-state digests, differing configurations or TensorFlow versions, and records from the same process. Held-out validation uses the fixed canonical test inputs and requires the frozen limits.

## Run package unit tests and load the encoder

```bash
uv run --package pixel-vae --with pytest --with pytest-cov pytest \
  models/pixel-vae/tests -m "not vendor_parity and not integration" \
  --cov=pixel_vae --cov-report=term-missing --cov-fail-under=90
uv run --package pixel-vae python -c 'from pixel_vae import PixelVAEEncoder; encoder = PixelVAEEncoder.from_pretrained(".cache/pixel-vae/converted/encoder"); print(encoder.config.latent_dim)'
```

The encoder returns the float32 posterior mean with shape `[batch, 256]`. Use `eval()` and `torch.inference_mode()` for deterministic embedding export. The model's posterior log-variance is available from its forward output for parity diagnostics.
