"""Write immutable, redacted records from a parity pytest process."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypeAlias


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


class _Report(Protocol):
    nodeid: str
    when: str
    outcome: str
    duration: float
    failed: bool
    longrepr: object


class _PluginManager(Protocol):
    def register(self, _plugin: "_RunRecorder", name: str) -> None: ...


class _Config(Protocol):
    pluginmanager: _PluginManager


class _Session(Protocol):
    """Pytest session passed to the finish hook; no attributes are consumed."""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _redact(value: JSONValue) -> JSONValue:
    if isinstance(value, str):
        for key in (
            "LAYOUTFORMERPP_REPORT_DATA_ROOT",
            "LAYOUTFORMERPP_REPORT_REPO_ROOT",
        ):
            actual = os.environ.get(key)
            if actual:
                replacement = (
                    "<your data root>" if "DATA_ROOT" in key else "<repo root>"
                )
                value = value.replace(actual, replacement)
        return value
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def _write_once(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


class _RunRecorder:
    def __init__(self) -> None:
        self.output_dir = Path(os.environ["LAYOUTFORMERPP_RECORD_DIR"])
        self.stage = os.environ["LAYOUTFORMERPP_RUN_STAGE"]
        self.repeat = int(os.environ["LAYOUTFORMERPP_RUN_REPEAT"])
        self.started_at = os.environ["LAYOUTFORMERPP_RUN_STARTED_AT"]
        self.reports: list[dict[str, JSONValue]] = []

    def pytest_runtest_logreport(self, report: _Report) -> None:
        if report.when not in {"setup", "call", "teardown"}:
            return
        entry: dict[str, JSONValue] = {
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "duration_seconds": report.duration,
        }
        if report.failed:
            entry["failure"] = str(report.longrepr)
        self.reports.append(entry)

    def pytest_sessionfinish(self, session: _Session, exitstatus: int) -> None:
        del session
        call_reports = [item for item in self.reports if item["phase"] == "call"]
        counts = {
            outcome: 0
            for outcome in ("passed", "failed", "skipped", "xfailed", "xpassed")
        }
        for item in call_reports:
            outcome = item["outcome"]
            if not isinstance(outcome, str):
                raise TypeError("pytest report outcome must be a string")

            counts[outcome] = counts.get(outcome, 0) + 1
        record = {
            "record_version": 1,
            "stage": self.stage,
            "repeat": self.repeat,
            "started_at": self.started_at,
            "finished_at": _now(),
            "source_commit": os.environ["LAYOUTFORMERPP_SOURCE_COMMIT"],
            "runtime_freeze_sha256": os.environ["LAYOUTFORMERPP_RUNTIME_FREEZE_SHA256"],
            "runtime_freeze": os.environ["LAYOUTFORMERPP_RUNTIME_FREEZE_RELATIVE"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "data_root": "<your data root>",
            "gate": {
                "rtol": 1e-4,
                "atol": 1e-5,
                "s3_repeat_is_reproducibility_evidence": self.stage == "s3"
                and self.repeat == 2,
            },
            "pytest_exit_status": exitstatus,
            "counts": counts,
            "reports": _redact(self.reports),
        }
        encoded = (
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        record_path = self.output_dir / "record.json"
        _write_once(record_path, encoded)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        _write_once(self.output_dir / "record.freeze", f"sha256:{digest} record.json\n")


def pytest_configure(config: _Config) -> None:
    config.pluginmanager.register(_RunRecorder(), "layoutformerpp-run-recorder")
