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

Initial adoption is **measurement-only**: no arbitrary cognitive or
cyclomatic threshold was introduced for pre-existing code. After establishing
the baseline, consider an incremental budget that prevents new regressions
without requiring wholesale refactoring of reproduction-sensitive models.
