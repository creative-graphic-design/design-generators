# Reproducing Collage Diffusion Checks

This guide runs the synthetic CPU contract tests and the private comparison between the authors' notebook behavior and this `🧨diffusers` pipeline.

Workflow order: download the upstream example inputs from the model repository's [`source-assets/`](https://huggingface.co/creative-graphic-design/collage-diffusion/tree/main/source-assets) in Colab, run CPU contract tests, run the temporary notebook reference and package comparison, then smoke-test local pipeline loading.

The authors' notebook is the single behavior reference. Its temporary comparison runner is prepared and run in Colab and is not stored in this repository. The source repository and the notebook-linked image and embedding assets have no declared distribution license. Keep the notebook code and source assets out of this Git repository; the linked assets and their provenance records are archived in the model repository's [`source-assets/`](https://huggingface.co/creative-graphic-design/collage-diffusion/tree/main/source-assets). The notebook names `stabilityai/stable-diffusion-2-1-base`, which now returns HTTP 404. Use `sd2-community/stable-diffusion-2-1-base` at revision `4e63672c03103b6c636b8fb4119ba982469b2955`; the matching weight hashes and component-file SHA-256 manifest are recorded under “Base Checkpoint Provenance and Integrity” in the package README. Both comparison paths must load the same pinned snapshot. Save comparison output only in private Hub storage and keep its path out of this repository.

## Parity-Sensitive Implementation Choices

These details explain behavior that must remain aligned when comparing the package with the notebook reference:

- Layer compositing passes each RGBA image as its own Pillow paste mask. For partially transparent pixels, Pillow applies alpha both as source opacity and as mask opacity, so the effective alpha is squared. The visible-mask and noise-map paths preserve that behavior.
- The reference inserts a textual-inversion token before its layer description. When textual inversion is enabled for any layer, it widens every matched token span one position to the left and leaves the end unchanged. A layer without its own embedding can therefore include the preceding word, such as `a`; this edge case is retained for comparison.
- Attention masks are converted to float64 and resized to each query grid with bilinear interpolation, `align_corners=False`, and `antialias=False`. This matches the reference environment's torchvision 0.15.1 tensor-resize behavior: all four Toy Box masks matched exactly at 64×64, while antialiasing introduced a maximum mask difference of 0.5504.
- The source-strength map is quantized to uint8 before nearest-neighbor resizing to the latent grid. Attention guidance requires scheduler `sigmas`; preservation uses the timestep-dependent threshold and neighborhood dilation.

## CPU Workspace Setup

Run these commands in the named Colab CPU session. Use the repository's standard workspace dependency configuration.

```bash
uv sync --package collage-diffusion
```

## Run Synthetic Tests

The tests use tiny generated images and fake pipeline components. They do not download the Stable Diffusion checkpoint or notebook assets.

```bash
uv run --package collage-diffusion pytest models/collage-diffusion/tests -m "not vendor_parity and not integration" --cov=collage_diffusion --cov-report=term-missing
```

## Run Repository Checks

```bash
SKIP=uv-lock uv run --package design-generators --group dev pre-commit run --all-files
uv run --package collage-diffusion ty check models/collage-diffusion
```

## Compare Against the Notebook

Download the Bento Box, Cake Table, and Toy Box layers and their selected textual-inversion embeddings from the linked [`source-assets/`](https://huggingface.co/creative-graphic-design/collage-diffusion/tree/main/source-assets) directory only into the Colab session. Run the notebook cells directly as the reference and this package separately with the same resized input, prompt, scheduler, seed, and 10 requested steps for seeds `0` and `1`. Use the notebook's `EulerAncestralDiscreteScheduler` for both paths. The measured matrix contains six pairs: Bento Box and Toy Box use strength `0.8` and `noise_blur=20`; Cake Table uses strength `0.75` and `noise_blur=10`; active steps are 8, 8, and 7, respectively.

Record the RGB output shapes, RMSE, maximum absolute pixel difference, PSNR, input hashes, software versions, and per-step latent differences from the temporary reference run. If a comparison differs, record the first stage that diverges among compositing, prompt tokens, text embeddings, initial latents, attention scores, scheduler output, per-layer mask blending, and decoding. The six measured conditions are worker-run evidence and do not replace the coordinator's independent acceptance rerun.

Estimate runtime and CU before starting the comparison. Begin on ordinary Colab CPU at 0.08 CU/hour and use high-RAM CPU at 0.26 CU/hour only after measuring a memory shortage. Keep setup, rehearsal, comparison, upload, and teardown within the 1 CU worker limit. Stop and report measured evidence if this limit is at risk. Stop the named session after upload and verify that it is no longer running.

## Smoke-Test Base Checkpoint Loading

This command downloads the pinned mirror into the Colab session and confirms that the standard `🧨diffusers` loader instantiates the package pipeline. It does not publish model weights.

The synthetic-component test also exercises `🧨diffusers` standard `save_pretrained` and `from_pretrained` methods without downloading a base checkpoint.

```bash
uv run --package collage-diffusion python -c 'from collage_diffusion import CollageDiffusionPipeline; pipe = CollageDiffusionPipeline.from_pretrained("sd2-community/stable-diffusion-2-1-base", revision="4e63672c03103b6c636b8fb4119ba982469b2955"); print(type(pipe).__name__)'
```

The comparison images, latents, metrics, and private Hub location are not committed to the repository or written into this guide.
