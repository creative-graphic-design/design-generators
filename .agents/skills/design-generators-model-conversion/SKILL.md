---
name: design-generators-model-conversion
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

## Repository Implementation Contract

- Keep original implementations under `vendor/` read-only. Isolate their dependencies behind a model package's `vendor` optional extra.
- Main package code (`models/*/src`, `lib/*/src`) and configs must not reference the vendor/original implementation in identifiers, docstrings, comments, or config names; vendor references belong only in conversion modules, `tests/vendor_parity`, and `REPRODUCING.md` / `TRAINING.md` docs.
- `laygen.common.vendor` is the narrow shared-library exception for resolving parity submodule checkouts; keep it documented in `scripts/check_src_vendor_language.py` if it remains in package source.
- Tensor and array annotations in package source (`models/*/src`, `lib/*/src`) must use fully qualified jaxtyping shaped types such as `Float[torch.Tensor, "..."]`; raw `torch.Tensor` and `np.ndarray` annotations are prohibited outside `scripts/jaxtyping_baseline.txt`.
- Write jaxtyping shaped types inline at the annotation site. Do not introduce module-level aliases such as `FooTensor = Float[...]` or `FooTensor: TypeAlias = Float[...]`; existing aliases are tracked only in `scripts/jaxtyping_alias_baseline.txt`.
- Model package names and Hub repo ids use the method name known in the literature, not necessarily the vendor repository slug.
- Example: vendor `const-layout` becomes package `layoutganpp` and Hub ids such as `creative-graphic-design/layoutganpp-rico`.
- Core model-package modules under `models/*/src/<pkg>/` must follow Hugging Face-style filenames with the package suffix, such as `configuration_<pkg>.py`, `modeling_<pkg>.py`, `pipeline_<pkg>.py`, `scheduling_<pkg>.py`, `processing_<pkg>.py`, `tokenization_<pkg>.py`, `image_processing_<pkg>.py`, and `generation_<pkg>.py`.
- Repository convention files and domain helpers must be explicitly allowed by `scripts/check_module_naming.py`; do not add new ad hoc core names.

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

### Repository Public Interface Contract

- Return the common layout schema: `bbox`, `labels`, `mask`, and `id2label`, with optional `sequences`, `scores`, `trajectory`, and `intermediates`.
- Transformers models return `laygen.modeling_outputs.LayoutGenerationOutput`; Diffusers pipelines return `laygen.pipelines.pipeline_output.LayoutGenerationOutput`. Both are explicit dataclasses and schema tests assert field names, order, and defaults stay aligned.
- Transformers-side layout pipelines subclass `laygen.pipelines.LayoutGenerationPipeline`; plain classes and `transformers.Pipeline` subclasses are non-conforming.
- Public `bbox` is normalized center `xywh` in `[0, 1]`, regardless of vendor internals such as `ltwh`, `ltrb`, bins, analog bits, or text tokens.
- Public `mask=True` means a valid element. Padding is represented by `mask`, never by reserving public label id `0`.
- `labels` are dataset-local integer ids unless a model explicitly documents request-local open-vocabulary ids.
- Persist `id2label` in config/model cards and return it with outputs.
- Public constructors must not synthesize default configs; require explicit config or derive it from a loaded artifact such as `model.config`.
- `generator` is the exact reproducibility API and takes precedence over `seed`.
- Canonical `condition_type` names are v1 `unconditional`, `label`, `label_size`, `completion`, `refinement`; v2 adds `text`, `content_image`, `relation`, `hierarchical`, `retrieval`. Normalize vendor aliases and raise explicit errors for unsupported modes.
- Constrained string options in public APIs use `Literal` aliases or `StrEnum` classes rather than bare `str` annotations.
- Discrete-vocabulary layout tokenizers subclass `transformers.PreTrainedTokenizer`; serialize auxiliary data with tokenizer files. Use a custom class only when the base class truly conflicts and document the reason.
- Store hierarchy, retrieval, open-vocabulary, attention, and other auxiliary data in `intermediates`; do not add an `extras` field.

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

### Repository Data Contract

- Prefer datasets hosted by the `creative-graphic-design` Hugging Face org. Check [issue #2 (umbrella plan)](https://github.com/creative-graphic-design/design-generators/issues/2) before adding a new data source.
- Use `creative-graphic-design/Rico` with `name="ui-screenshots-and-hierarchies-with-semantic-annotations"` for RICO25; the default config is metadata-only. RICO13 needs a vendor-derived mapping.
- PubLayNet is `creative-graphic-design/PubLayNet`; avoid any test path that could download the full dataset.
- Crello uses `cyberagent/crello` as the canonical source until an org mirror exists; `creative-graphic-design/Desigen` is not a Crello substitute.
- Respect pinned dataset quirks from [issue #2 (umbrella plan)](https://github.com/creative-graphic-design/design-generators/issues/2) and [issue #60 (implementation checklist)](https://github.com/creative-graphic-design/design-generators/issues/60): Magazine is polygon-based and train-only, PKU has an `INVALID` class and pixel `ltrb` boxes, and CGL-v2 needs `ralf-style` for validation/saliency use cases.

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

- Golden parity fixtures are generated by running vendor code, not handwritten.
- Do not commit golden tensors, images, weights, or large downloaded artifacts. Commit only metadata needed to regenerate them: seeds, conditions, environment notes, config hashes, and script arguments.
- Run parity generation on one explicitly selected GPU with fixed seeds.
- Coordinator parity reruns must set `PARITY_REQUIRE=1` so missing local assets fail loudly instead of turning an all-skip run into an apparent success.
- For LLM API or in-context methods, parity means prompt bytes, exemplar selection, parser behavior, and repair policy match the paper/vendor path.

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

## Documentation Adapter

For documentation, use the `design-generators-documentation` skill (read `.agents/skills/design-generators-documentation/SKILL.md`).

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
