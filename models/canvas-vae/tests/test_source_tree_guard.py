import subprocess
import sys
from pathlib import Path


GUARD = Path(__file__).parents[1] / "scripts" / "source_tree_guard.py"


def test_source_tree_guard_rejects_changes_between_seeds(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source_root)], check=True)
    source = source_root / "model.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source_root), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        check=True,
    )
    manifest = tmp_path / "seed-source.json"

    def run(action: str, seed: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(GUARD),
                action,
                "--source-root",
                str(source_root),
                "--manifest",
                str(manifest),
                "--seed",
                str(seed),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert run("record", 0).returncode == 0
    assert run("verify", 1).returncode == 0
    source.write_text("value = 2\n", encoding="utf-8")
    changed = run("verify", 2)
    assert changed.returncode == 1
    assert "source-tree hash changed before seed 2" in changed.stderr
