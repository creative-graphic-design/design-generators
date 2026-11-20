# TKG-DM reproduction

This package checks the TKG-DM initial-noise transformation against the author implementation and compares one SDXL output using pinned model revisions. Both model families also load through the local custom-pipeline entry point. Stable Diffusion 1.5 has no author pipeline, so its checks cover the shared noise transformation and repeatable package inference. No author source or generated image fixture is stored in this repository.

Workflow order: install the package, check out the pinned author reference, download the pinned base checkpoints, compare noise and image outputs, run package tests, and preview the Hub publisher. The parity cases also smoke-test local loading with `DiffusionPipeline.from_pretrained` for both model families.

## References and implementation differences

The author implementation is [`ryugo417/TKG-DM` at `6dae8837021e4dd2eeea8e53b9ca7c5c8447bd10`](https://github.com/ryugo417/TKG-DM/tree/6dae8837021e4dd2eeea8e53b9ca7c5c8447bd10). It contains an SDXL generation pipeline and shared noise-processing utilities. The repository does not declare a license, so its code is checked out only in a temporary Colab directory as an execution reference.

The [Creative Graphic Design fork](https://github.com/creative-graphic-design/TKG-DM/tree/b1494d14b6f659d6b25a925b119a1ef4c6ddc037) is licensed under Apache-2.0. It adds a Stable Diffusion 1.5 pipeline, tests, and notebooks, along with [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) pipeline classes. Both fork pipelines discard caller-supplied `latents` before delegating to the `diffusers` implementation of `prepare_latents`; this package preserves them. The author example passes `negative_prompts`; `diffusers` 0.39.0 expects `negative_prompt`, so the parity run adds only that keyword adapter.

The paper specifies `target_shift=0.07` for both model families. The author SDXL implementation uses `0.11`, and it is the SDXL end-to-end reference here. The SDXL pipeline default is therefore `0.11`; the SD 1.5 pipeline default is `0.07` because the authors provide no SD 1.5 pipeline. In the noise transformation, `target_shift` raises the initial positive-value ratio of latent channels 1 and 2 by that amount.

## Fixed comparison conditions

The conditions and acceptance thresholds are fixed before the comparison run. The author reference and package use the same Colab GPU type, runtime, pinned `diffusers` version, base model revision, prompts, precision, scheduler, seed, and inference settings. The author code has no dependency lockfile, so both implementations use the versions selected by this repository's `uv.lock`.

| Setting         | Stable Diffusion 1.5                                                                    | SDXL                                       |
| --------------- | --------------------------------------------------------------------------------------- | ------------------------------------------ |
| Base model      | `stable-diffusion-v1-5/stable-diffusion-v1-5`                                           | `stabilityai/stable-diffusion-xl-base-1.0` |
| Base revision   | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`                                              | `462165984030d82259a11f4367a4eed129e94a7b` |
| Prompt          | `yellow lemon and slice, realistic, photo-realistic, 4K, high resolution, high quality` | Same                                       |
| Negative prompt | `background, character, cartoon, anime, text, fail, low resolution`                     | Same                                       |
| Seed            | `1234`                                                                                  | `1234`                                     |
| Precision       | fp16                                                                                    | fp16                                       |
| Scheduler       | DDIM                                                                                    | DDIM                                       |
| Inference steps | 50                                                                                      | 50                                         |
| Guidance scale  | 7.5                                                                                     | 7.5                                        |
| Output          | 512×512                                                                                 | 1024×1024                                  |
| `target_shift`  | 0.07                                                                                    | 0.11                                       |

Noise outputs from the same fp16 input latent must satisfy `rtol=1e-5` and `atol=1e-6`. The complete SDXL normalized RGB images must have PSNR of at least 48 dB and MAE no greater than `1/255`. SD 1.5 package inference is repeated with the same seed and settings; the two RGB outputs must be identical. The SDXL image criteria and noise tolerances are not adjusted after results are observed.

The comparison passed on a Colab Tesla T4 using Python 3.13.16, PyTorch 2.13.0+cu130, `diffusers` 0.39.0, and [`🤗transformers`](https://huggingface.co/docs/transformers/index) 5.13.0. The reference was pinned to `6dae8837021e4dd2eeea8e53b9ca7c5c8447bd10`, and both base checkpoints used the revisions listed above. Noise transformation had a maximum absolute error of 0 for both model families. The SDXL package and reference images were pixel-identical at 1024×1024 (PSNR ∞ dB, MAE 0). Repeated SD 1.5 package inference produced identical 512×512 RGB images with SHA-256 `4c76d1b6af804a6660f0a5ca2b3d335118b0af497cb361e997c418c5f6becdc3`. The README example PNG SHA-256 values are `54ce56eb88007f66f186e0d7962586e4ace6f4ea9163aee76adc8e0ccd0c7ea7` for SD 1.5 and `1ca97db64162d18cb78c10473c905d38f0f2b9821e8f2ca221973876ca39b4fe` for SDXL.

## Colab GPU run

Use one selected CUDA device in a named Colab session. The reference checkout and generated outputs remain outside the repository:

```bash
export CUDA_VISIBLE_DEVICES=<gpu-index>
export TKG_DM_AUTHOR_REPO="$TMPDIR/tkg-dm-author"
export TKG_DM_PARITY_OUTPUT_DIR="$TMPDIR/tkg-dm-parity-output"
git clone https://github.com/ryugo417/TKG-DM.git "$TKG_DM_AUTHOR_REPO"
git -C "$TKG_DM_AUTHOR_REPO" checkout 6dae8837021e4dd2eeea8e53b9ca7c5c8447bd10
mkdir -p "$TKG_DM_PARITY_OUTPUT_DIR"
UV_NO_CONFIG=1 UV_EXCLUDE_NEWER='7 days' uv sync --package tkg-dm --locked
PARITY_REQUIRE=1 \
TKG_DM_AUTHOR_REPO="$TKG_DM_AUTHOR_REPO" \
TKG_DM_PARITY_OUTPUT_DIR="$TKG_DM_PARITY_OUTPUT_DIR" \
UV_NO_CONFIG=1 UV_EXCLUDE_NEWER='7 days' \
uv run --package tkg-dm --with pytest --locked pytest \
  models/tkg-dm/tests/vendor_parity/test_tkg_dm_parity.py -s
```

The tests record the GPU, Python and library versions, source revisions, comparison metrics, output dimensions, and image hashes in `results.json` under `TKG_DM_PARITY_OUTPUT_DIR`. The two package example images are written there as `sd15_readme_example.png` and `sdxl_readme_example.png`; the SDXL reference and package comparison images are recorded separately. Copy only the two package examples to the public `creative-graphic-design/documentation-images` dataset under `tkg-dm/` after verifying their hashes.

The author repository is a temporary reference only. Do not copy its files into the package or commit its checkout, model weights, generated outputs, or image fixtures.

## Package tests and Hub dry-run

Run the model's network-free unit tests with the member selected:

```bash
UV_NO_CONFIG=1 UV_EXCLUDE_NEWER='7 days' \
uv run --package tkg-dm --with pytest --with pytest-cov --locked pytest \
  models/tkg-dm/tests -m "not vendor_parity and not integration" \
  --cov=tkg_dm --cov-report=term-missing --cov-fail-under=90
```

When published, the code-only Hub repository will contain the TKG-DM pipeline and neither base checkpoint. Preview its exact allowlisted files without a Hugging Face token:

```bash
UV_NO_CONFIG=1 UV_EXCLUDE_NEWER='7 days' \
uv run --package tkg-dm --locked python \
  models/tkg-dm/scripts/publish_to_hub.py \
  --revision "$(git rev-parse HEAD)" --dry-run
```

The main-branch publication workflow runs the publisher after the package is merged. A real Hub upload requires `creative-graphic-design/tkg-dm` to exist. The shared fine-grained `HF_TOKEN` must have content-write access to the selected published model repositories, including this one.
