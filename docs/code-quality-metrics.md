# Python code-quality metrics

The [Code Complexity workflow](../.github/workflows/complexity.yml) runs on
pull requests, merge-queue changes, merges to `main`, and manual dispatch.

- [cccc](https://github.com/moznion/cccc) measures cognitive and cyclomatic
  complexity for first-party Python code in `lib/`, `models/`, `scripts/`,
  and `tools/`. The `cccc.toml` configuration excludes tests, generated
  files, and vendored submodules. No uv environment or model weights are needed.
- The workflow checks that the JSON report has valid summary values, then
  lists the 20 highest-cognitive-complexity functions in the GitHub Actions
  job summary.
- [octocov](https://github.com/k1LoW/octocov) presents the cccc summary
  as custom metrics, and also measures the Python code-to-test ratio.
  It posts a best-effort pull request comment and a job summary.
  `main` runs save history to the `octocov` branch, under
  `complexity-reports/`, and PRs compare against that history.
  Pull requests from forks cannot post comments with read-only tokens,
  but their job summaries remain available.
- Existing member-level pytest coverage thresholds and Codecov uploads
  remain the source of truth for test coverage. The complexity workflow
  deliberately does not combine partial member coverage into a misleading
  whole-repository coverage percentage.

## Running the same analysis locally

Install `cccc` using the
[upstream installation instructions](https://github.com/moznion/cccc#install--build),
then run from the repository root:

```bash
cccc lib models scripts tools > cccc.json
jq '.summary' cccc.json
jq -r '[.files[] | .path as $path | .functions[] | recurse(.children[]?) |
  {path: $path, name: .name, line: .line, cognitive: .cognitive}] |
  sort_by(-.cognitive) | .[:20][]' cccc.json
```

## Strict cognitive complexity gate

The workflow uses `max-cognitive = 15` in `cccc.toml`, equivalent to
running `cccc --max-cognitive 15 lib models scripts tools`.
**Every first-party Python function is subject to this limit**, including
pre-existing code. The workflow fails if any function has cognitive
complexity greater than 15; there are no grandfathered exceptions.

The first measurement on PR #537 counted **178 existing functions above 15**
(out of 3,967 functions across 614 files). Consequently the new check will
**intentionally fail** until those functions have been simplified.
These functions are not silently waived. The workflow keeps the cccc and
octocov reporting steps running despite a threshold violation, then fails
in the final gate step so the failure is visible.

The octocov custom metrics also include the number of functions above
15, 30 and 50 and the complexity distribution, so progress can be tracked
as existing functions are refactored. Because model implementations must
preserve reference behavior, changes to numerical paths require parity
verification before being accepted.

Locally, the command exits nonzero when the limit is exceeded:

```bash
cccc --max-cognitive 15 lib models scripts tools > cccc.json
```
