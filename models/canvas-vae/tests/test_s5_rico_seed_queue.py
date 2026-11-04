import importlib.util
import json
import sys
from pathlib import Path

import pytest


QUEUE_PATH = Path(__file__).parents[1] / "scripts" / "s5_rico_seed_queue.py"
QUEUE_SPEC = importlib.util.spec_from_file_location("s5_rico_seed_queue", QUEUE_PATH)
assert QUEUE_SPEC is not None and QUEUE_SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(QUEUE_SPEC)
QUEUE_SPEC.loader.exec_module(QUEUE)


def test_queue_launch_forwards_hub_repository() -> None:
    repository = "example-owner/test-artifacts"
    manifest_path = QUEUE.DEFAULT_MANIFEST
    parity_path = QUEUE.REPOSITORY_ROOT / "models/canvas-vae/TRAINING.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_commit"] = "a" * 40
    manifest["queue_status"] = "ready"
    manifest["evaluation_path_parity"] = {
        "path": "models/canvas-vae/TRAINING.md",
        "sha256": QUEUE.sha256_file(parity_path),
    }
    manifest["runtime_plan"]["source_commit"] = "a" * 40

    _, command = QUEUE.launch_preconditions(manifest, manifest_path, repository)

    assert command[-2:] == ["--hub-repo", repository]


@pytest.mark.parametrize("repository", [None, "   "])
def test_queue_refuses_launch_without_hub_repository(
    monkeypatch: pytest.MonkeyPatch,
    repository: str | None,
) -> None:
    args = [str(QUEUE_PATH), "--launch"]
    if repository is not None:
        args.extend(["--hub-repo", repository])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(QUEUE, "load_manifest", lambda _path: {})
    monkeypatch.setattr(QUEUE, "current_source_state", lambda: ("a" * 40, True))
    monkeypatch.setattr(
        QUEUE,
        "run_queue",
        lambda *_args: pytest.fail("queue must not run without a Hub repository"),
    )

    assert QUEUE.main() == QUEUE.BLOCKED_EXIT
