# Reproducing Collage Diffusion Checks

This guide runs the synthetic CPU contract tests and the private comparison between the authors' notebook behavior and this `🧨diffusers` pipeline.

Workflow order: download the upstream example inputs in Colab, run CPU contract tests, run the temporary notebook reference and package comparison, then smoke-test local pipeline loading.

The authors' notebook is the single behavior reference. Its temporary comparison runner is prepared and run in Colab and is not stored in this repository. The source repository and the notebook-linked image and embedding assets have no declared distribution license; do not add the notebook code or those assets here. The notebook names `stabilityai/stable-diffusion-2-1-base`, which now returns HTTP 404. Use `sd2-community/stable-diffusion-2-1-base` at revision `4e63672c03103b6c636b8fb4119ba982469b2955`; the matching weight hashes and component-file SHA-256 manifest are recorded in the [package README](README.md#base-checkpoint-provenance-and-integrity). Both comparison paths must load the same pinned snapshot. Save comparison output only in private Hub storage and keep its path out of this repository.

## CPU Workspace Setup

Run these commands in the named Colab CPU session. Use the repository's standard workspace dependency configuration.

```bash
uv sync --package collagediffusion
```

## Run Synthetic Tests

The tests use tiny generated images and fake pipeline components. They do not download the Stable Diffusion checkpoint or notebook assets.

```bash
uv run --package collagediffusion pytest models/collage-diffusion/tests -m "not vendor_parity and not integration" --cov=collagediffusion --cov-report=term-missing
```

## Run Repository Checks

```bash
SKIP=uv-lock uv run --package design-generators --group dev pre-commit run --all-files
uv run --package collagediffusion ty check models/collage-diffusion
```

## Compare Against the Notebook

Download the notebook example layers and its selected textual-inversion embedding only into the Colab session. Use one temporary script to run the notebook's processing and this package sequentially with the same resized inputs, scheduler, seed `0`, and 10 steps. The notebook's `EulerAncestralDiscreteScheduler` is used for both paths. The default comparison is a single notebook example and one output pair.

Record the RGB output shapes, maximum absolute pixel difference, PSNR, input hashes, software versions, and any per-step latent differences available from the temporary reference run. If a comparison differs, record the first stage that diverges among compositing, prompt tokens, text embeddings, initial latents, attention scores, scheduler output, per-layer mask blending, and decoding. The comparison is behavioral evidence for this input and seed, not the paper's full multi-scene evaluation.

Estimate runtime and CU before starting the comparison. Begin on ordinary Colab CPU at 0.08 CU/hour and use high-RAM CPU at 0.26 CU/hour only after measuring a memory shortage. Keep setup, rehearsal, comparison, upload, and teardown within the 1 CU worker limit. Stop and report measured evidence if this limit is at risk. Stop the named session after upload and verify that it is no longer running.

## Smoke-Test Base Checkpoint Loading

This command downloads the pinned mirror into the Colab session and confirms that the standard `🧨diffusers` loader instantiates the package pipeline. It does not publish model weights.

The synthetic-component test also exercises `🧨diffusers` standard `save_pretrained` and `from_pretrained` methods without downloading a base checkpoint.

```bash
uv run --package collagediffusion python -c 'from collagediffusion import CollageDiffusionPipeline; pipe = CollageDiffusionPipeline.from_pretrained("sd2-community/stable-diffusion-2-1-base", revision="4e63672c03103b6c636b8fb4119ba982469b2955"); print(type(pipe).__name__)'
```

The comparison images, latents, metrics, and private Hub location are not committed to the repository or written into this guide.
