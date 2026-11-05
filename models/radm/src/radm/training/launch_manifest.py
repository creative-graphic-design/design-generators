"""Shared schema for RADM CGL launch manifests."""

from __future__ import annotations

from typing import Literal, TypedDict


class CGLS5ConfigIdentity(TypedDict):
    """Repository config path and content digest used by a CGL training run."""

    path: str
    sha256: str


class CGLS5DataRoot(TypedDict):
    """Visible alias and canonical target of the CGL data root."""

    alias: str
    target: str


class CGLS5CheckpointRule(TypedDict):
    """Checkpoint selection rule and the expected terminal checkpoint path."""

    type: Literal["top_k_val_loss", "final_iteration"]
    checkpoint: str
    filename: str


class CGLS5LaunchManifest(TypedDict):
    """Identity, runtime, data, and artifact record for one CGL training launch."""

    source_commit: str
    vendor_gitlink_commit: str
    vendor_checkout_commit: str
    arm: Literal["original", "package"]
    training_seed: int
    runtime: dict[str, str | int]
    config: CGLS5ConfigIdentity
    max_steps: int
    output_root: str
    data_root: CGLS5DataRoot
    text_feature_inventory: dict[str, dict[str, int]]
    vendor_compatibility: dict[str, str] | None
    checkpoint_rule: CGLS5CheckpointRule
    evaluator_command: list[str]
    evaluator_cwd: str
    artifacts: dict[str, str]
    launched_at: str
