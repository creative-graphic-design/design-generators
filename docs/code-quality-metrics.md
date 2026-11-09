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

## Baseline quality gate

The complexity workflow fails when any of these conditions is violated:

| Metric | Initial limit |
| --- | ---: |
| Python parse errors | 0 |
| Cognitive complexity P95 | 15 |
| Cognitive complexity maximum | 125 |
| Cyclomatic complexity P95 | 11 |
| Cyclomatic complexity maximum | 107 |
| Functions with cognitive complexity >15 | 178 |
| Functions with cognitive complexity >30 | 39 |
| Functions with cognitive complexity >50 | 16 |

These limits come from the first successful cccc run on PR #537
(614 files, 3,967 functions). They are **initial regression budgets**,
not target values or an endorsement of highly complex functions. The
workflow also verifies that its recursive function list matches the
summary's function count. The >15, >30, and >50 counts are published
to octocov for trend monitoring.

The aggregate limits permit an increase in one function's complexity
when other functions improve enough to offset it. A future follow-up
should implement a per-function changed-code gate, with a recommended
cognitive threshold of 15 for new functions and a no-regression policy
for existing functions. Model implementation refactors must preserve
vendor parity and numerical reproducibility.
