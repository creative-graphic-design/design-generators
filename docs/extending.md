---
icon: lucide/git-branch
tags:
  - Conventions
  - Contributors
---

# Extending

This guide defines how downstream projects, model ports, and AI coding agents should extend `design-generators` and the framework packages it builds on. The goal is to keep changes small enough to review, test, upgrade, and attribute.

## Why This Guide

This repository wraps research systems in public `transformers`-, `diffusers`-, and `pydantic-ai`-style interfaces. Those frameworks are large projects with their own release cadence and compatibility contracts. Copying their source trees, or copying a whole `design-generators` package into another project, makes the real change hard to identify and usually breaks upgrades, licensing review, and provenance tracking.

## Anti-Pattern

Do not vendor a full framework or repository source tree and edit files in place.

```text
my-project/
  transformers/
    pipelines/
      image_generation.py  # edited local copy
  design-generators/
    lib/laygen/
      src/laygen/pipelines.py  # edited local copy
```

That layout hides the actual behavior delta inside unrelated upstream files. It also makes security updates and dependency upgrades manual because the copied code is no longer connected to the released package that users install.

## Recommended Path

Install pinned released packages and extend the narrow public surface you need. For layout generation, prefer subclassing the shared pipeline/config/model classes and keep new behavior in your own module.

```python
from __future__ import annotations

from laygen.pipelines import LayoutGenerationPipeline
from laygen.pipelines.pipeline_output import LayoutGenerationOutput


class MyLayoutPipeline(LayoutGenerationPipeline):
    """Small extension that keeps the upstream package installed normally."""

    def __call__(
        self,
        *,
        condition_type: str = "unconditional",
        num_layouts: int = 1,
        generator=None,
    ) -> LayoutGenerationOutput:
        outputs = super().__call__(
            condition_type=condition_type,
            num_layouts=num_layouts,
            generator=generator,
        )
        return LayoutGenerationOutput(
            bbox=outputs.bbox,
            labels=outputs.labels,
            mask=outputs.mask,
            id2label=outputs.id2label,
            intermediates={"source": "my-layout-pipeline"},
        )
```

Keep package pins visible in `pyproject.toml`, `uv.lock`, `requirements.txt`, or the consuming project's equivalent lockfile. Keep subclass tests focused on the behavior you changed rather than re-testing the entire upstream framework.

## Last-Resort Internal Modifications

Internal copies are allowed only after an upstream extension point is not sufficient.

1. Open or link an upstream issue or pull request that explains the missing extension point.
2. Copy only the smallest single module that must change, not a package tree.
3. Record the upstream package name, version, commit, file path, license, and local diff summary next to the copied module.
4. Add a test that fails if the copied module drifts from the documented upstream baseline without an explicit provenance update.

For model ports in this repository, `vendor/` remains pinned and read-only. Vendor code is used to regenerate parity fixtures and compare behavior; production wrappers should import released dependencies and local package code instead of mutating vendor sources.

## Repository Practice

`design-generators` depends on released `transformers`, `diffusers`, `huggingface_hub`, and related packages. Model packages expose public wrappers by subclassing or composing those APIs. When original research repositories are needed, they live under `vendor/` as pinned read-only submodules or external checkouts for parity workflows.

Generated model cards, README files, and docs should describe the wrapper behavior and provenance plainly. They should not imply that local copied framework trees are the supported integration path.

## AI Coding Agents

AI coding agents MUST keep framework and repository deltas reviewable. Install pinned dependencies, add focused subclasses or adapter modules, and explain any new public interface in the PR. Do not paste a framework source tree, mutate vendored code, or silently replace dependency imports with local copies.
