# Reproducing StructureDiffusion CPU Agreement

This guide runs package tests and compares generated images against the pinned official attention implementation on a Colab CPU runtime. The comparison uses the official `CrossAttention.multi_qkv` implementation with the same Stable Diffusion v1.4 weights in the released pipeline format, parser outputs, scheduler, seed, and starting latent. It does not run or patch the vendor demo's CUDA-only model loader.

Run from the repository root. Do not install the model package dependencies on a machine whose free disk space is below 3 GB. Generated images and metric JSON files belong under `.cache/structurediffusion/` and are not committed.

Workflow order: install the CPU environment, download parser and model assets, run unit and repository checks, compare the package with the reference, then stop the Colab session.

## 1. Check out the source revision

Create a named CPU session and clone the repository revision being evaluated. Use `main` after the package is merged; for a pull request, set `REVISION` to its branch name. Initialize the pinned vendor submodule used by the comparison.

```bash
colab new -s structurediffusion-cpu-parity
REVISION=main
git clone --depth 1 --branch "$REVISION" https://github.com/creative-graphic-design/design-generators.git /content/design-generators-sdg
cd /content/design-generators-sdg
git submodule update --init vendor/structured-diffusion-guidance
python -m pip install uv
```

The named session is reused for dependency installation, checks, rehearsal, comparison, result download, and teardown. Use a standard or high-RAM CPU runtime as needed; do not request an accelerator.

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

The repository pre-commit command skips the lockfile writer, aggregate member tests, and Vulture, which CI runs in separate jobs. The member test command enforces the package's 90% coverage floor.

```bash
SKIP=uv-lock,pytest-models,vulture .venv/bin/pre-commit run --all-files
.venv/bin/ruff check models/structurediffusion
.venv/bin/ruff format --check models/structurediffusion
.venv/bin/ty check models/structurediffusion
mkdir -p /content/sdg-root-coverage
COVERAGE_FILE=/content/sdg-root-coverage/.coverage \
  .venv/bin/coverage run --rcfile=/dev/null --include='scripts/*.py' \
  -m pytest tests -m "not integration" --import-mode=importlib
COVERAGE_FILE=/content/sdg-root-coverage/.coverage \
  .venv/bin/coverage report --rcfile=/dev/null
.venv/bin/zensical build --strict -f mkdocs.yml
UV_NO_SYNC=1 MEMBER_TEST_RUNNER_DEPS=0 scripts/run_member_tests.sh models/structurediffusion
UV_NO_SYNC=1 uv run --package structurediffusion pytest models/structurediffusion/tests -m "not vendor_parity and not integration"
```

## 4. Rehearse at 64×64

Use the same marked test for the rehearsal and the final 512×512 comparison. The rehearsal checks model loading, parsing, token alignment, the vendor processor, and CPU memory use; it is not the reported image-parity result.

```bash
UV_NO_SYNC=1 PYTHONPATH=vendor/structured-diffusion-guidance PARITY_REQUIRE=1 SDG_COMPARE_RESOLUTION=64 uv run --package structurediffusion pytest models/structurediffusion/tests/vendor_parity/test_cpu_image_parity.py -m vendor_parity -vv -s
```

Record `elapsed_seconds`, peak memory, and process exit status. Estimate the 512×512 comparison from the rehearsal timing before continuing; the final run uses the same 10 steps and initial latent shape `[1, 4, 64, 64]` for each path.

The executed 64×64 rehearsal passed in 96.9 seconds with peak RSS 4,957 MiB and zero pixel difference. This is a CPU execution and memory check; the reported parity result is the 512×512 comparison below.

## 5. Compare at 512×512

Proceed only if the measured time projection remains within the approved worker budget of 1 CU. The comparison fixture uses one prompt (`a red car and a white sheep`), seed 42, constituency parsing, guidance scale 7.5, 10 DDIM steps, CPU float32, and one precomputed initial latent for both paths. It uses lowercase because the official reference's case-sensitive parser-to-token alignment raises `IndexError` for an uppercase `A`; the package accepts capitalized prompts, as tested by its CPU suite.

```bash
UV_NO_SYNC=1 PYTHONPATH=vendor/structured-diffusion-guidance PARITY_REQUIRE=1 SDG_COMPARE_RESOLUTION=512 uv run --package structurediffusion pytest models/structurediffusion/tests/vendor_parity/test_cpu_image_parity.py -m vendor_parity -vv -s
```

The test writes `.cache/structurediffusion/parity/cpu-512.json` with the phrase list, CLIP spans, initial-latent SHA-256, model-load and per-pipeline generation times, total elapsed time, peak process RSS, and maximum, mean, and root-mean-square absolute pixel differences. Generated images are compared in memory and are not saved. The 512×512 CPU comparison passed; reference generation took 430.2 seconds and package generation took 451.6 seconds, with all three absolute pixel-difference metrics equal to `0.0`.

## 6. Check outputs and stop the Colab session

Download the JSON result if it is needed for the PR, then stop the named session and verify that no session remains:

```bash
colab download -s structurediffusion-cpu-parity /content/design-generators-sdg/.cache/structurediffusion/parity/cpu-512.json .cache/structurediffusion/parity/cpu-512.json
colab stop -s structurediffusion-cpu-parity
colab sessions
```

The package does not convert the Stable Diffusion checkpoint, so no checkpoint-conversion command is required. The CPU unit suite covers a local `save_pretrained` to `from_pretrained` round trip with tiny randomly initialized components; it needs neither network access nor Stable Diffusion weights.
