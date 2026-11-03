# Reproducing StructureDiffusion CPU Agreement

This guide runs package tests and compares generated images against a pinned attention reference on a standard Colab CPU runtime. It checks noun-phrase extraction, CLIP token spans, independently assembled keys and values, cross-attention, and image output. The reference uses a separate base `diffusers.StableDiffusionPipeline` instance with the reference `CrossAttention.multi_qkv`; the package uses `StructureDiffusionPipeline`. Both model loading and sampling use Diffusers, so the comparison does not claim parity with the reference checkpoint loader or native sampler implementation.

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

Use the same marked test for the rehearsal and the final 512×512 comparison. The rehearsal checks model loading, parsing, token alignment, independent vendor conditioning, vendor attention, and CPU memory use; it is not the reported image-parity result. Both runs receive `A red car and a white sheep`; the official demo's `preprocess_prompts` lowercases it before parsing. A direct call to `get_token_alignment_map` with mismatched casing can fail, but that is not the demo's end-to-end behavior.

```bash
UV_NO_SYNC=1 PYTHONPATH=vendor/structured-diffusion-guidance PARITY_REQUIRE=1 SDG_COMPARE_RESOLUTION=64 uv run --package structurediffusion pytest models/structurediffusion/tests/vendor_parity/test_cpu_image_parity.py -m vendor_parity -vv -s
```

Record `elapsed_seconds`, peak memory, and process exit status. Estimate the 512×512 comparison from the rehearsal timing before continuing; proceed only if the full run remains within the 0.5 CU cap. The final run uses the same 10 PNDM steps through Diffusers' PLMS path and the same initial latent shape `[1, 4, 64, 64]` for each path.

The executed 64×64 rehearsal passed in 92.2 seconds with peak RSS 5,497 MiB and zero pixel difference. This is a CPU execution and memory check; the reported parity result is the 512×512 comparison below.

## 5. Compare at 512×512

Proceed only if the measured time projection remains within the 0.5 CU worker budget. The comparison fixture uses the input prompt (`A red car and a white sheep`), seed 42, constituency parsing, guidance scale 7.5, 10 Diffusers PNDM steps with `skip_prk_steps=True`, CPU float32, and one precomputed initial latent for both paths. The package receives the capitalized prompt; the vendor reference applies its own `preprocess_prompts` lowercasing before parsing and token alignment.

```bash
UV_NO_SYNC=1 PYTHONPATH=vendor/structured-diffusion-guidance PARITY_REQUIRE=1 SDG_COMPARE_RESOLUTION=512 uv run --package structurediffusion pytest models/structurediffusion/tests/vendor_parity/test_cpu_image_parity.py -m vendor_parity -vv -s
```

The test writes `.cache/structurediffusion/parity/cpu-512.json` with phrase lists, independently mapped CLIP spans, key/value conditioning differences, initial-latent SHA-256, model-load and per-pipeline generation times, total elapsed time, peak process RSS, and maximum, mean, and root-mean-square absolute pixel differences. It also writes `package-512.png` and `reference-512.png`. Before stopping the Colab session, store result files in private storage, download them again, and verify each SHA-256 digest. Keep credentials out of notebook output and logs; inspect logs for token-shaped strings before uploading. The reported image criterion is mean absolute pixel difference below `1e-3`.

The pre-commit working-tree run at 512×512 reported phrase and CLIP-span matches, zero maximum key and value differences, and zero maximum, mean, and RMS image differences. Its runner wall time was 996.7 seconds (981.2 seconds reported by the test), with 6,573.1 MiB peak RSS, or approximately 0.022 CU on a standard CPU at 0.08 CU/hour. This run was based on commit `7a8a3a6836ae60cf7328a4cc439d94ccc5222449` with uncommitted changes, so these values are preliminary and are not formal parity evidence. The metrics JSON did not contain a Git SHA. Formal parity is pending an independent rerun by the reviewer on the pushed PR head; do not report these values as the formal result.

## 6. Check outputs and stop the Colab session

After uploading and reading the private Hub artifacts back, stop the named session and verify that it is no longer assigned:

```bash
colab stop -s structurediffusion-cpu-parity
colab sessions
```

Record that private artifact readback and SHA-256 verification passed. Do not publish the result repository or generated images.

The package does not convert the Stable Diffusion checkpoint, so no checkpoint-conversion command is required. The CPU unit suite covers a local `save_pretrained` to `from_pretrained` round trip with tiny randomly initialized components; it needs neither network access nor Stable Diffusion weights.
