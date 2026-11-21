# Reproducing SAWNA

This procedure compares the package with the pinned Space implementation and generates the three README examples. It uses the checkpoint revisions, prompts, boxes, and acceptance rules in the model card.

Workflow order: download the pinned Space and model files, generate reference noise, run GPU acceptance with pytest, then smoke-test `from_pretrained` loading.

## Sources and Checkpoints

| Source               | Revision                                   |
| -------------------- | ------------------------------------------ |
| EQUES Space          | `e0a8f843a0b4e62c0650b03d5f335271ecb36d0b` |
| Stable Diffusion 1.5 | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| Stable Diffusion 2.1 | `bb2154823665391b4fb29b0b9cf82a198964ee05` |
| Stable Diffusion XL  | `462165984030d82259a11f4367a4eed129e94a7b` |

The reference source is downloaded to a temporary cache at the pinned Space revision. It is not copied into the package or repository. The test records SHA-256 hashes for `app.py` and `tkg_dm.py`.

## Environment

Use the repository lock and [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) `0.39.0`. Run asset-backed acceptance on one Colab GPU type for all three models. Use fp16, seed `1234`, prompt `A majestic lion in a natural landscape`, box `(0.3, 0.3, 0.7, 0.7)`, target shift `0.07`, automatic blur, 25 steps, guidance scale `7.5`, and no negative prompt.

The default output path is `.cache/sawna/parity`. Set `SAWNA_PARITY_OUTPUT_DIR` to a durable run directory before execution. The result directory contains one PNG and one tensor archive per checkpoint, plus `parity-results.json`. The JSON records source and model revisions, dependency versions, GPU, scheduler hashes, generation arguments, and result hashes.

The full three-checkpoint run is budgeted for one Colab L4 session with a two-hour wall-clock limit. Record the selected GPU, elapsed runtime, and before-and-after compute-unit balance with the run results. Do not purchase compute units for this run.

## Package Tests

From the repository root, run the unit and integration tests with the workspace package selected:

```bash
uv run --package sawna pytest models/sawna/tests --cov=sawna --cov-report=term-missing
```

The suite covers box-mask construction, validation, zero-box behavior, Stable Diffusion and SDXL initial-noise preparation, scheduler scaling, and the custom pipeline component selection. It requires at least 90 percent package coverage.

## Asset-Backed Acceptance

Run the vendor comparison with `PARITY_REQUIRE=1` so absent GPU access or source assets fail the check rather than skip it:

```bash
PARITY_REQUIRE=1 \
SAWNA_PARITY_OUTPUT_DIR=/content/sawna-parity \
uv run --package sawna pytest models/sawna/tests/vendor_parity/test_sawna_parity.py -q
```

The test downloads the pinned Space source and each public checkpoint. It compares package `epsilon_masked` tensors with the source output from the same fp16 noise, boxes, target shift, and automatic blur. It then compares image tensors from the source `diffusers` pipeline and the SAWNA pipeline with the same masked latents and sampler-generator state. All three checkpoint cases require bitwise equality. The SAWNA pipeline also runs twice with the same public generator argument to verify repeatability.

The test saves its generated examples as `sawna-<checkpoint>-lion-seed1234.png`. Upload those files to `creative-graphic-design/documentation-images` under `sawna/`, preserving the names used in the README. Verify the uploaded files by downloading each one and comparing its SHA-256 with the local output. Keep the JSON and tensor archives with the run record outside git.

## Recording Results

Record revisions, source hashes, package commit, GPU, runtime, elapsed time, and compute-unit balance. Include dependency versions, scheduler hashes, generation settings, comparison results, and image hashes. Report pass and skip counts separately. A passing run has three exact masked-noise comparisons, three exact generated-image comparisons, and three repeatability comparisons.

Do not describe parity as verified when the required run is absent or has skipped cases. If a comparison is not bitwise equal, preserve the artifacts and identify the first differing tensor before proposing a tolerance. A tolerance requires a measured numerical cause and written justification before changing the acceptance criterion.
