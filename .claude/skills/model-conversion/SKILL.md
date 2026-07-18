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
4. Add the `in-progress` label to the target model issue.
5. Confirm the model slug, Python package name, Hub repo ids, datasets, license
   status, and whether the implementation belongs in Transformers, Diffusers, a
   recipe, training code, or Pydantic AI.

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
   floating outputs with documented tolerances.
6. Add `save_pretrained` -> `from_pretrained` smoke tests that run without
   network or vendor weights.

For LLM API methods, parity checks cover prompt byte identity, exemplar
selection, parser behavior, and repair/retry policy.

Vendor parity is complete only after the released weights have been obtained,
the vendor code has generated reference outputs, the real comparison suite has
passed, and a local `save_pretrained` -> `from_pretrained` smoke test has
passed. A skip-only hook plus a follow-up issue is not a completed parity path:
the coordinator must be able to independently rerun and accept the parity
suite, and hooks that only skip leave nothing to verify.

## README And Model Cards

Write `models/<slug>/README.md` in model-card style. Include:

- overview and original implementation link
- install and `from_pretrained` usage
- supported checkpoints and intended Hub ids
- datasets and pinned configs
- reproducibility summary with vendor-parity numbers
- license status and citation
- `Reproducibility`

Every model README must include a top-level `## Parity Results` section. Put
the vendor-parity summary in a numeric table that states what was compared, the
number of cases, the match criterion, and the result. Prose inside
`## Reproducibility` is not enough. Use `models/layout-dm/README.md` as the
reference format.

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

Do not self-merge. Completion is a PR with green CI or a clearly documented CI
blocker.
