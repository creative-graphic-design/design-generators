---
name: model-conversion
description: Use when implementing one design-generators model issue, converting a vendor layout/poster generation model into a Transformers or Diffusers-style workspace member with vendor parity, README/model-card documentation, and PR checklist reporting.
---

# Model Conversion

Use this skill when a model issue is ready for implementation. It assumes the
repository-level invariants in `AGENTS.md` and the live checklist in issue #60.
Do not copy large checklist text here; read those sources at the start of each
conversion.

## Before Editing

1. Create or switch to a fresh worktree based on the current `origin/main`.
2. Read, in this order:
   - `AGENTS.md`
   - issue #60
   - issue #2 body and the comments for unified interface v1/v2, decisions,
     data-source policy, status/tracking, model-card policy, and shared library
     naming
   - issue #64 for `lib/laygen`, `lib/posgen`, and import direction
   - the target model issue plan comment
   - every later amendment or review comment on the target issue
3. Treat target-issue amendments as higher priority than the original plan.
4. For plans that add public methods or override `from_pretrained`,
   `save_pretrained`, `generate`, or other entry points on Hugging Face base
   classes, require an explicit justification line before `plan-agreed`. The
   coordinator must check that line before applying `plan-agreed`, so
   non-idiomatic APIs are caught while the plan is still cheap to change.
5. Add the `in-progress` label to the target model issue.
6. Confirm the model slug, Python package name, Hub repo ids, datasets, license
   status, and whether the implementation belongs in Transformers, Diffusers, a
   recipe, training code, or Pydantic AI.
7. If the work reveals stale, incorrect, or missing guidance, do not silently
   work around it; make the small in-scope fix or propose a focused `meta`
   follow-up.

## Source Language

Main package code under `models/*/src` and `lib/*/src` must read as this
repository's own implementation. Do not describe runtime modules, public
arguments, comments, or docstrings as vendor wrappers, vendor-compatible
surfaces, or ports of vendor code. Use repository-owned wording such as
`released`, `checkpoint`, `reference`, `source`, or `original-code dependency`
when the distinction is needed.

Vendor-language references are limited to conversion-responsibility modules,
`tests/vendor_parity`, `REPRODUCING.md`, and `TRAINING.md`. If a package needs
to compare against an original implementation, keep that detail in conversion,
reference-generation, or parity-test paths rather than the public runtime API.

## Package Shape

Create one uv workspace member under `models/<slug>/`:

```text
models/<slug>/
  pyproject.toml
  README.md
  scripts/
  src/<package>/
  tests/
  tests/vendor_parity/
```

Use the literature method name for `<slug>` and `<package>`. Put original
implementation dependencies in a `vendor` optional extra and keep `vendor/`
submodules read-only.

Name direct package modules under `models/<slug>/src/<package>/` with
Hugging Face-style core filenames when they implement package surface or core
runtime behavior: `configuration_<package>.py`, `modeling_<package>.py`,
`pipeline_<package>.py`, `scheduling_<package>.py`,
`processing_<package>.py`, `tokenization_<package>.py`,
`image_processing_<package>.py`, or `generation_<package>.py`. Keep
repository convention files such as `conversion.py` and narrowly scoped domain
helpers only when they are covered by `scripts/check_module_naming.py`; extend
that checker deliberately before adding a new direct helper filename.

Use shared libraries by import:

```python
from laygen.common.outputs import LayoutGenerationOutput
from laygen.common.bbox import ltwh_to_xywh, ltrb_to_xywh
from laygen.common.testing import assert_layout_output_schema
```

Use `posgen.common` only for poster/content-aware helpers that already exist.
Do not copy common bbox, label, output, or testing helpers into the model
package.

## Public Interface

Expose the agreed generation surface even when the model rejects some modes:

```python
def __call__(
    self,
    *,
    batch_size: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    condition_type: str = "unconditional",
    labels: torch.Tensor | np.ndarray | list | None = None,
    bbox: torch.Tensor | np.ndarray | list | None = None,
    mask: torch.Tensor | np.ndarray | list | None = None,
    num_elements: int | list[int] | torch.Tensor | None = None,
    box_format: Literal["xywh", "ltwh", "ltrb"] = "xywh",
    normalized: bool = True,
    canvas_size: tuple[int, int] | None = None,
    num_inference_steps: int | None = None,
    output_type: Literal["dataclass", "dict"] = "dataclass",
    return_intermediates: bool = False,
    **model_kwargs,
) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
    ...
```

Normalize vendor aliases to canonical `condition_type` names before dispatch.
Unsupported conditions must raise explicit errors. `generator` wins over `seed`.
Return normalized center `xywh` boxes and mask-based padding.

For v2 models, add the relevant public inputs from issue #2 (`prompt`,
`content`, `image`, `saliency`, `scene_graph`, `relations`, `hierarchy`,
`retrieval`, `retrieval_examples`, or `label_texts`) without replacing the v1
schema.

Discrete-vocabulary layout tokenizers should subclass
`transformers.PreTrainedTokenizer` unless the target issue documents a concrete
conflict. Serialize special tokens and auxiliary artifacts such as cluster
centers with tokenizer files.

### Public API Idioms

Transformers and Diffusers base classes should keep their upstream contracts.
Before implementing or reviewing a plan, verify these rules:

- A `transformers.PreTrainedModel` subclass must implement `forward`. If the
  computation structurally cannot be represented as one forward pass, such as a
  multi-stage composition through decoded text, do not make the composite a
  `PreTrainedModel`; compose the stages in the pipeline layer instead. This
  keeps the model class loadable, callable, and inspectable through standard
  Transformers behavior, while the pipeline owns cross-model orchestration as
  Diffusers pipelines do.
- Model classes expose only standard model entry points: `forward` and, when
  the model is token-generative, token-level `generate`. Layout-level APIs that
  return `LayoutGenerationOutput` belong on `pipeline.__call__`; do not add
  model methods such as `generate_layout`. This keeps layout orchestration,
  condition normalization, output schema construction, and `generator`/`seed`
  precedence in one public surface.
- Transformers-side layout pipelines must subclass
  `laygen.pipelines.LayoutGenerationPipeline`. Plain pipeline classes and
  `transformers.Pipeline` subclasses are non-conforming because
  `transformers.Pipeline` is designed for registered single-model tasks with
  the preprocess, `_forward`, and postprocess contract, not custom layout tasks
  or multi-model composition. The shared laygen base gives Transformers-side
  packages the pipeline role that `DiffusionPipeline` gives Diffusers packages:
  subfolder `from_pretrained`/`save_pretrained`, device and dtype handling,
  `generator` over `seed`, and a `__call__` contract returning
  `laygen.modeling_outputs.LayoutGenerationOutput`.
- Do not fully override `from_pretrained` or `save_pretrained` in a way that
  bypasses the standard loading and serialization machinery. If an override is
  unavoidable, document the reason in the PR body. Standard loading is what
  makes local smoke tests, Hub checkpoints, config round-trips, and downstream
  tooling work predictably.
- Public constructors must not synthesize default configs. Require an explicit
  config at construction time or derive it from a loaded artifact such as
  `model.config`; do not hide `config or SomeConfig(...)` fallbacks in
  tokenizers, processors, pipelines, agents, or conversion helpers.
- Use upstream class suffixes only when the class satisfies the upstream
  contract. For example, `ForConditionalGeneration` implies seq2seq-style
  `forward` plus `generate`; do not reuse that suffix for classes that cannot
  satisfy those methods. Class names should tell users which Transformers idiom
  is safe to rely on.
- Vendor-specific decoding that cannot be expressed as a stateless
  `LogitsProcessor` may live as a model-side helper called by the pipeline, but
  it must not become the public generation API. This allows stateful or
  layout-aware decoding internals without teaching users a second, model-level
  generation surface.

## Data

Keep dataset loading behind processors so sources can change without touching
model code. Prefer `creative-graphic-design/*` datasets and use the pinned
configs from issue #2 and issue #60.

Use builders, streaming, synthetic rows, or tiny local fixtures in ordinary
tests. Do not write tests that download PubLayNet, poster datasets, vendor
weights, or large cache bundles.

When a required dataset is absent from the org, use the original vendor
distribution or the approved external source named in issue #2, and leave a TODO
that points to the missing import.

## Vendor Parity

Implement parity in this order:

1. Add scripts that download vendor weights or point to an existing cache.
2. Add a reference-generation script that runs the vendor code with fixed seeds
   and one selected GPU.
3. Save generated goldens outside git; commit only regeneration metadata.
4. Add `tests/vendor_parity/` tests that skip cleanly when weights, vendor deps,
   or goldens are absent.
5. Compare exact token/id outputs where deterministic; compare logits and
   floating outputs by bitwise equality by default.
6. Add `save_pretrained` -> `from_pretrained` smoke tests that run without
   network or vendor weights.

For LLM API methods, parity checks cover prompt byte identity, exemplar
selection, parser behavior, and repair/retry policy.

Use tolerance-based floating parity only after identifying and justifying the
root cause in the PR body. When outputs diverge, first check alignment of TF32
settings, attention implementation paths such as SDPA versus vendor handwritten
attention, floating-point operation order, and dtype derivation order. These
are recurring parity hazards, as seen in the LACE `SinusoidalPosEmb` path and
the four-factor LayouSyn parity investigation. If replacing vendor code with a
shared implementation, verify equivalence on vendor real-scale inputs before
claiming parity; tiny synthetic inputs can hide scale-dependent numeric drift.
This keeps tolerance thresholds from masking accidental behavior changes.

Vendor parity is complete only after the released weights have been obtained,
the vendor code has generated reference outputs, the real comparison suite has
passed, and a local `save_pretrained` -> `from_pretrained` smoke test has
passed. A skip-only hook plus a follow-up issue is not a completed parity path:
the coordinator must be able to independently rerun and accept the parity
suite, and hooks that only skip leave nothing to verify.

## README And Model Cards

Start `models/<slug>/README.md` from
`references/model-readme-template.md`, which follows the Hugging Face Hub model
card metadata spec and the `huggingface_hub` official
`modelcard_template.md` headings. Fill every placeholder with the target
issue's concrete model, checkpoint, dataset, parity, license, and citation
details. Keep the final README in model-card style. Include:

- overview and original implementation link
- install and `from_pretrained` usage
- supported checkpoints and intended Hub ids
- datasets and pinned configs
- reproducibility summary with vendor-parity numbers
- license status and citation
- `Reproducibility` link to `models/<slug>/REPRODUCING.md`

The user-facing install snippet must use pip direct references to this
repository's package subdirectories. Include workspace libraries that are not
published on PyPI, such as `laygen` or `posgen`, in the same command as the
model package. Preserve clone + uv commands only for development or
`REPRODUCING.md` workflows.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "posgen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/posgen" \
  "<package-name> @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/<slug>"
```

Omit `posgen` when the model does not depend on it, and keep extras on the
shared package requirement when the model depends on one, for example
`laygen[agents]`.

Every model README must include a `### Parity Results` section under `## Evaluation`. Put
the vendor-parity summary in a numeric table that states what was compared, the
number of cases, the match criterion, and the result. Prose inside
`## Reproducibility` is not enough. Put the mechanical walkthrough in
`REPRODUCING.md`; reviewers run that file's download, reference-generation,
parity, conversion, and smoke-test commands.
`references/model-readme-template.md` as the reference format.

The `Reproducibility` section must open with one sentence that states how to
reproduce the original-implementation agreement checks. The remaining commands
must be copy-pasteable and ordered: download vendor assets, generate vendor
references with `CUDA_VISIBLE_DEVICES`, run `pytest -m vendor_parity`, convert
checkpoints, and run `from_pretrained` smoke tests.

Generate Hub model cards through `laygen.common.model_card` and the official
Hugging Face template. Do not push model weights or Hub repos from ordinary
implementation PRs unless the coordinator explicitly asks for publish.

## Validation

Run commands through uv with the package selected:

```bash
uv run --package <member-name> ruff check .
uv run --package <member-name> ty check .
uv run --package <member-name> pytest models/<slug>/tests -m "not vendor_parity and not integration"
SKIP=uv-lock uv run pre-commit run --all-files
```

For root-only documentation changes, the final pre-commit command is still
required. If a package has extras for vendor or parity work, document the exact
extra in the README and PR body.

## Pull Request

Open the PR against `main`. In the PR description, include:

- target issue and implemented scope
- issue #60 checklist verification, with deviations quoted explicitly
- parity results and commands used
- tests run locally
- Hub publish status; normally "not pushed"
- follow-ups and unresolved source/license questions

PR bodies must follow `.github/PULL_REQUEST_TEMPLATE.md`: `Summary`, `Changes`,
`Verification`, `Checklist`, and `Deviations / Follow-ups`. The
`gh pr create --body` command bypasses GitHub's template auto-fill, so
CLI-created PRs must use `--body-file .github/PULL_REQUEST_TEMPLATE.md` as the
starting point and fill in that structure. Mark checklist items with `[x]` only
when they were actually verified.
When creating a PR, add the same lane/topic labels as the implementation issue;
do not add issue-only status labels such as `plan-agreed`, `in-progress`, or
`parity-verified` to PRs.

Do not self-merge. Completion is a PR with green CI or a clearly documented CI
blocker.
