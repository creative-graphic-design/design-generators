# <model-slug>

<!-- Choose the package/model slug used under models/<slug>/ and in Hub ids. -->

<one-sentence overview with method name, venue, and arXiv or paper link>

<!-- Include the literature method name, conference or journal if known, and a stable paper URL. -->

## Supported Checkpoints

<!-- Use planned Hub ids from the model issue. Mark Status as public, not-pushed, or planned. -->

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| <dataset-or-variant> | `creative-graphic-design/<hub-repo-id>` | <public-or-not-pushed> |

## Installation

<!-- Use the workspace member name from models/<slug>/pyproject.toml. -->

```bash
uv sync --package <member-name>
```

## Usage

<!-- Use the public from_pretrained surface exposed by this model package. -->

```python
from <package_name> import <PipelineOrModelClass>

pipe = <PipelineOrModelClass>.from_pretrained(
    "creative-graphic-design/<hub-repo-id>",
)
out = pipe(batch_size=1, seed=0)

print(out.bbox)    # normalized center xywh boxes
print(out.labels)  # dataset-local integer labels
print(out.mask)    # valid element mask
```

## Datasets

<!-- Use approved creative-graphic-design dataset ids and note dataset-specific quirks from issue #2 and issue #60. -->

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| <dataset-name> | `creative-graphic-design/<dataset-id>` | <pinned-config-or-dataset-quirk> |

## Parity Results

<!-- Fill with numeric results from the real vendor-parity suite; do not replace this with prose. -->

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| <tokenizer-or-forward-or-sampling-check> | <case-count> | <exact-or-tolerance> | <passed-count-or-error> |

## Reproducibility

This section reproduces the original-implementation agreement checks for <model-slug>.

<!-- Keep these commands copy-pasteable because coordinators run them mechanically during review. -->

Download vendor assets:

```bash
uv run --package <member-name> python models/<slug>/scripts/download_original.py \
  --output-dir .cache/<slug>/original
```

Generate vendor reference outputs:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package <member-name> --extra vendor python models/<slug>/scripts/generate_reference_outputs.py \
  --output-dir models/<slug>/tests/vendor_parity/fixtures/<dataset-name> \
  --seed <seed>
```

Convert checkpoints:

```bash
uv run --package <member-name> --extra convert python models/<slug>/scripts/convert_original_checkpoint.py \
  --output-dir .cache/<slug>/converted/<hub-repo-id>
```

Run vendor parity tests:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package <member-name> pytest models/<slug>/tests/vendor_parity -m vendor_parity -rs
```

Run `from_pretrained` smoke tests:

```bash
uv run --package <member-name> python - <<'PY'
from pathlib import Path
from <package_name> import <PipelineOrModelClass>

path = Path(".cache/<slug>/converted/<hub-repo-id>")
pipe = <PipelineOrModelClass>.from_pretrained(path)
out = pipe(batch_size=1, seed=0)
assert out.bbox.shape[-1] == 4
assert out.labels.shape == out.mask.shape
print(out.bbox.shape, out.labels.shape)
PY
```

## Original Implementation

<!-- Link the exact upstream repository, release, or archive used for conversion and parity. -->

- Original implementation: <original-implementation-url>
- Released weights: <released-weights-url-or-status>

## License And Citation

<!-- State the upstream license, package license implications, and the citation required by the paper. -->

License: <license-name-or-review-needed>

```bibtex
<bibtex-citation>
```
