import json
import os
import subprocess
from pathlib import Path


SETUP_SCRIPT = Path(__file__).parents[1] / "scripts" / "colab_uv_setup.sh"


def _fake_tools(tmp_path: Path, commit: str) -> tuple[Path, dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    git_script = fake_bin / "git"
    git_script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        "with Path(os.environ['FAKE_CALLS_PATH']).open('a') as calls:\n"
        "    calls.write(json.dumps(['git', *arguments]) + '\\n')\n"
        "if arguments[:2] == ['check-ref-format', '--branch']:\n"
        "    raise SystemExit(0)\n"
        "if arguments and arguments[0] == 'clone':\n"
        "    Path(arguments[-1]).mkdir(parents=True)\n"
        "if len(arguments) >= 4 and arguments[2:4] == ['rev-parse', 'HEAD']:\n"
        "    print(os.environ['FAKE_COMMIT'])\n"
        "if len(arguments) >= 4 and arguments[2:4] == ['submodule', 'update']:\n"
        "    print(\"Submodule 'vendor/canvas-vae' registered for path 'vendor/canvas-vae'\")\n"
        "",
        encoding="utf-8",
    )
    uv_script = fake_bin / "uv"
    uv_script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['FAKE_CALLS_PATH']).open('a') as calls:\n"
        "    calls.write(json.dumps(['uv', *sys.argv[1:]]) + '\\n')\n",
        encoding="utf-8",
    )
    git_script.chmod(0o700)
    uv_script.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_CALLS_PATH": str(calls_path),
            "FAKE_COMMIT": commit,
        }
    )

    return calls_path, environment


def _setup_command(repo_path: Path, commit: str) -> list[str]:
    return [
        "bash",
        str(SETUP_SCRIPT),
        "--source-url",
        "https://github.com/creative-graphic-design/design-generators.git",
        "--source-branch",
        "feat/canvas-vae-crello-model",
        "--source-commit",
        commit,
        "--repo-dir",
        str(repo_path),
        "--package",
        "canvas-vae",
        "--extra",
        "training",
        "--extra",
        "vendor",
        "--submodules",
    ]


def test_shared_setup_checks_out_submodules_and_syncs_requested_extras(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    calls_path, environment = _fake_tools(tmp_path, commit)
    repo_path = tmp_path / "repo"

    result = subprocess.run(
        _setup_command(repo_path, commit),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]

    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == ["git"] * 7 + ["uv"]
    assert calls[1][1] == "clone"
    assert calls[6][1:] == [
        "-C",
        str(repo_path),
        "submodule",
        "update",
        "--init",
        "--recursive",
    ]
    assert calls[-1] == [
        "uv",
        "sync",
        "--frozen",
        "--project",
        str(repo_path),
        "--package",
        "canvas-vae",
        "--extra",
        "training",
        "--extra",
        "vendor",
    ]


def test_shared_setup_fails_closed_when_checkout_commit_differs(
    tmp_path: Path,
) -> None:
    calls_path, environment = _fake_tools(tmp_path, "b" * 40)

    result = subprocess.run(
        _setup_command(tmp_path / "repo", "a" * 40),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]

    assert result.returncode == 78
    assert "source checkout differs" in result.stderr
    assert calls[-1] == ["git", "-C", str(tmp_path / "repo"), "rev-parse", "HEAD"]


def test_shared_setup_emits_export_for_synced_environment(tmp_path: Path) -> None:
    commit = "c" * 40
    calls_path, environment = _fake_tools(tmp_path, commit)
    repo_path = tmp_path / "repo"
    environment_path = tmp_path / "vendor environment"
    command = [
        "bash",
        "-c",
        'eval "$("$@")"; printf "%s\\n" "$UV_PROJECT_ENVIRONMENT"',
        "environment-probe",
        *_setup_command(repo_path, commit),
        "--uv-environment",
        str(environment_path),
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{environment_path}\n"
    assert "Submodule 'vendor/canvas-vae' registered" in result.stderr
    assert "colab_setup_passed" in result.stderr
    assert calls[-1] == [
        "uv",
        "sync",
        "--frozen",
        "--project",
        str(repo_path),
        "--package",
        "canvas-vae",
        "--extra",
        "training",
        "--extra",
        "vendor",
    ]
