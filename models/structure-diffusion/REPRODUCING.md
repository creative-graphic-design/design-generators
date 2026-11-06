# Reproducing StructureDiffusion CPU Agreement

This guide runs package tests and compares generated images against a reference image produced on a standard Colab CPU runtime. The test directly compares noun phrases, CLIP token spans, and independently assembled K/V conditioning with the package values. It generates the reference image by running the pinned reference attention processors in a separate base [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) pipeline, then compares the final reference and package images; it does not capture or compare per-layer attention outputs. Both paths use `diffusers` for model loading and sampling, so the comparison does not claim parity with the reference checkpoint loader or native sampler implementation.

Run from the repository root. Do not install the model package dependencies on a machine whose free disk space is below 3 GB. Generated images and metric JSON files belong under `.cache/structure_diffusion/` and are not committed.

Workflow order: install the CPU environment, download parser and model assets, run unit and repository checks, compare the package with the reference, then stop the Colab session.

## 1. Check out the source revision

Create a named CPU session and clone the repository revision being evaluated. Use `main` after the package is merged; for a pull request, set `REVISION` to its branch name. Initialize the pinned vendor submodule used by the comparison.

```bash
colab new -s structure-diffusion-cpu-parity
REVISION=main
git clone --depth 1 --branch "$REVISION" https://github.com/creative-graphic-design/design-generators.git /content/design-generators-sdg
cd /content/design-generators-sdg
git submodule update --init vendor/structured-diffusion-guidance
python -m pip install uv
```

The named standard-CPU session is reused for dependency installation, checks, rehearsal, comparison, private result upload and readback, and teardown. Do not request an accelerator. The total session budget is 0.5 CU (6.25 hours at 0.08 CU/hour); stop if the measured projection would exceed this cap.

## 2. Install the workspace and parser models

Install all workspace members so repository-wide type checks have their declared dependencies. Exclude PyPI's Torch and TorchVision wheels, then install CPU-only Torch from the official PyTorch CPU index. The lock resolves Torch 2.13.0 for this package.

```bash
uv sync --all-packages --group dev --group docs \
  --no-install-package torch --no-install-package torchvision
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"
uv pip install --python .venv/bin/python "einops>=0.8"
export UV_NO_SYNC=1
.venv/bin/python -c "import stanza; stanza.download('en', processors='tokenize,pos,constituency')"
```

The default comparison uses the constituency parser. Scene-graph parser installation is optional for this run. The base checkpoint downloads from `CompVis/stable-diffusion-v1-4` during comparison and remains in the Colab Hugging Face cache.

## 3. Run repository and package checks

The repository pre-commit command skips the lockfile writer, aggregate member tests, and Vulture; its duplicate-code hook remains available to run locally. CI runs Vulture in the separate Dead Code job and duplicate-code in the separate Duplicate Code job on merge queue and manual dispatches from non-main branches. Duplicate Code compares changed Python files with `origin/main`. Main pushes, scheduled runs, and dispatches to `main` have no targets. Dead Code continues full-repository checks on those events. The member test command enforces the package's 90% coverage floor.

```bash
SKIP=uv-lock,pytest-models,vulture .venv/bin/pre-commit run --all-files
.venv/bin/ruff check models/structure-diffusion
.venv/bin/ruff format --check models/structure-diffusion
.venv/bin/ty check models/structure-diffusion
mkdir -p /content/sdg-root-coverage
COVERAGE_FILE=/content/sdg-root-coverage/.coverage \
  .venv/bin/coverage run --rcfile=/dev/null --include='scripts/*.py' \
  -m pytest tests -m "not integration" --import-mode=importlib
COVERAGE_FILE=/content/sdg-root-coverage/.coverage \
  .venv/bin/coverage report --rcfile=/dev/null
.venv/bin/zensical build --strict -f mkdocs.yml
UV_NO_SYNC=1 MEMBER_TEST_RUNNER_DEPS=0 scripts/run_member_tests.sh models/structure-diffusion
UV_NO_SYNC=1 uv run --package structure-diffusion pytest models/structure-diffusion/tests -m "not vendor_parity and not integration"
```

## 4. Rehearse at 64×64

Use the same marked test for the rehearsal and the final 512×512 comparison. The rehearsal checks model loading, parsing, token alignment, independent vendor conditioning, vendor attention, and CPU memory use; it is not the reported image-parity result. Both runs receive `A red car and a white sheep`; the official demo's `preprocess_prompts` lowercases it before parsing. A direct call to `get_token_alignment_map` with mismatched casing can fail, but that is not the demo's end-to-end behavior.

```bash
UV_NO_SYNC=1 PYTHONPATH=vendor/structured-diffusion-guidance PARITY_REQUIRE=1 SDG_COMPARE_RESOLUTION=64 uv run --package structure-diffusion pytest models/structure-diffusion/tests/vendor_parity/test_cpu_image_parity.py -m vendor_parity -vv -s
```

Record `elapsed_seconds`, peak memory, and process exit status. Estimate the 512×512 comparison from the rehearsal timing before continuing; proceed only if the full run remains within the 0.5 CU cap. The final run uses the same 10 PNDM steps through the PLMS path in `diffusers` and the same initial latent shape `[1, 4, 64, 64]` for each path.

The independent comparison on commit `786fcf6cfbcfb68b9c9565d8ee275c32cd5a28ac` also measured the 64×64 case. It completed in 126.777 seconds: noun phrases and spans matched exactly, the measured maximum K/V difference was 0.0, and the final images were bit-exact; see the formal results below.

## 5. Compare at 512×512

Proceed only if the measured time projection remains within the 0.5 CU worker budget. The comparison fixture uses the input prompt (`A red car and a white sheep`), seed 42, constituency parsing, guidance scale 7.5, 10 PNDM steps with `skip_prk_steps=True` through the PLMS path in `diffusers`, CPU float32, and one precomputed initial latent for both paths. The package receives the capitalized prompt; the vendor reference applies its own `preprocess_prompts` lowercasing before parsing and token alignment.

```bash
UV_NO_SYNC=1 PYTHONPATH=vendor/structured-diffusion-guidance PARITY_REQUIRE=1 SDG_COMPARE_RESOLUTION=512 uv run --package structure-diffusion pytest models/structure-diffusion/tests/vendor_parity/test_cpu_image_parity.py -m vendor_parity -vv -s
```

The test writes `.cache/structure_diffusion/parity/cpu-512.json` with phrase lists, independently mapped CLIP spans, key/value conditioning differences, initial-latent SHA-256, model-load and per-pipeline generation times, total elapsed time, peak process RSS, and maximum, mean, and root-mean-square absolute pixel differences. It also writes `package-512.png` and `reference-512.png`. Before stopping the Colab session, store result files in private storage, download them again, and verify each SHA-256 digest. Keep credentials out of notebook output and logs; inspect logs for token-shaped strings before uploading. The test requires each maximum absolute key/value conditioning difference to be at most `1e-6` and the mean absolute pixel difference to be below `1e-3`; the formal result had zero measured K/V differences and bit-exact images with equal SHA-256 values.

The reviewer independently ran the comparison from commit `786fcf6cfbcfb68b9c9565d8ee275c32cd5a28ac` on a standard CPU with input prompt `A red car and a white sheep`, seed 42, guidance scale 7.5, the same initial CPU latent for both paths, and 10 PNDM steps through the PLMS path in `diffusers`. The vendor demo lowercases its prompt before parsing and alignment. The result details are recorded in the [formal parity comment](https://github.com/creative-graphic-design/design-generators/issues/485#issuecomment-6608854871).

| Measurement                                        | 64×64                                                              | 512×512                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Noun phrases                                       | `a red car`; `a white sheep`                                       | `a red car`; `a white sheep`                                       |
| Vendor/package BOS-inclusive CLIP spans            | `[1, 4)`; `[5, 8)` / match                                         | `[1, 4)`; `[5, 8)` / match                                         |
| Key/value count                                    | 1 / 3                                                              | 1 / 3                                                              |
| Initial latent SHA-256                             | `9ff459eefd9169b11138967326b63683050096a4f7b9df76ecb1bb893ad4c84f` | `e97f3de04c7c6ba86354db4402b91d4402ede9c72aabed29a96dd7d635a798ba` |
| Maximum absolute key/value conditioning difference | 0.0 / 0.0                                                          | 0.0 / 0.0                                                          |
| Maximum / mean / RMS absolute pixel difference     | 0.0 / 0.0 / 0.0                                                    | 0.0 / 0.0 / 0.0                                                    |
| Reference/package image SHA-256                    | Match                                                              | Match                                                              |
| Test runtime                                       | 126.777 seconds                                                    | 948.902 seconds                                                    |

At each resolution, noun phrases and token spans matched exactly, the measured maximum differences in keys and values were 0.0, and the final reference and package images were bit-exact with matching SHA-256 values. The reference attention processors generate the reference images; the test does not compare per-layer attention outputs. Model loading and sampling use `diffusers` in both paths, so this comparison does not claim parity with the vendor checkpoint loader or sampler.

## 6. Check outputs and stop the Colab session

After uploading and reading the private Hub artifacts back, stop the named session and verify that it is no longer assigned:

```bash
colab stop -s structure-diffusion-cpu-parity
colab sessions
```

Record that private artifact readback and SHA-256 verification passed. Do not publish the private parity result bundle or the reference/package comparison images.

The package does not convert the Stable Diffusion checkpoint, so no checkpoint-conversion command is required. The CPU unit suite covers a local `save_pretrained` to `from_pretrained` round trip with tiny randomly initialized components; it needs neither network access nor Stable Diffusion weights.
