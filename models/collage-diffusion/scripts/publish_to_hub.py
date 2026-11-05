#!/usr/bin/env python3
"""Publish the Collage custom pipeline and model card to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import Requirement

if TYPE_CHECKING:
    from huggingface_hub import HfApi

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = (
    PACKAGE_ROOT / "src" / "collage_diffusion" / "pipeline_collage_diffusion.py"
)
README_SOURCE = PACKAGE_ROOT / "README.md"
HUB_REPOSITORY_ID = "creative-graphic-design/collage-diffusion"
UPLOAD_ALLOW_PATTERNS = ("README.md", "pipeline.py")
SOURCE_ASSET_PREFIX = "source-assets/"
PLANNED_HUB_BADGE = (
    "![hub](https://img.shields.io/static/v1?label=hub&message=planned-code-only"
    "&color=orange&style=flat-square&logo=huggingface&logoColor=white)"
)
PUBLISHED_HUB_BADGE = (
    "![hub](https://img.shields.io/static/v1?label=hub&message=published-code-only"
    "&color=brightgreen&style=flat-square&logo=huggingface&logoColor=white)"
)
ARCHIVE_URL_MARKER = "/archive/main.zip"
ARCHIVE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
LAYGEN_REQUIREMENT_PATTERN = re.compile(
    r'(?m)^\s*"(?P<requirement>laygen\s*@\s*https:'
    r'//[^"\s]+)"\s*\\?\s*$'
)


def prepare_upload_files(
    pipeline_source: bytes,
    readme_text: str,
    revision: str,
) -> dict[str, bytes]:
    """Build the exact Hub files from the canonical pipeline and package README."""
    if not ARCHIVE_REVISION_PATTERN.fullmatch(revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")

    if not readme_text.startswith("---\n"):
        raise ValueError("package README must begin with YAML frontmatter")

    frontmatter_end = readme_text.find("\n---\n", 4)
    if frontmatter_end < 0:
        raise ValueError("package README has no closing YAML frontmatter delimiter")

    frontmatter = readme_text[4:frontmatter_end]
    metadata_lines = frontmatter.splitlines()
    license_line_indexes = [
        index
        for index, line in enumerate(metadata_lines)
        if line.startswith("license:")
    ]
    if len(license_line_indexes) != 1:
        raise ValueError("package README must have exactly one frontmatter license")

    del metadata_lines[license_line_indexes[0]]
    card_text = "---\n" + "\n".join(metadata_lines) + "\n---\n"
    card_text += readme_text[frontmatter_end + 5 :]

    if card_text.count(ARCHIVE_URL_MARKER) != 2:
        raise ValueError("package README must document both archive installs once")

    card_text = card_text.replace(ARCHIVE_URL_MARKER, f"/archive/{revision}.zip")
    card_text = card_text.replace(PLANNED_HUB_BADGE, PUBLISHED_HUB_BADGE, 1)
    return {"README.md": card_text.encode("utf-8"), "pipeline.py": pipeline_source}


def extract_laygen_requirement(card_text: str) -> str:
    """Return the laygen direct reference documented in the published card."""
    matches = list(LAYGEN_REQUIREMENT_PATTERN.finditer(card_text))
    if len(matches) != 1:
        raise ValueError("published README must contain one laygen archive install")

    requirement_text = matches[0].group("requirement")
    requirement = Requirement(requirement_text)
    if requirement.name != "laygen" or requirement.url is None:
        raise ValueError("published README laygen install must be a direct reference")

    expected_url_prefix = "https:" + (
        "//github.com/creative-graphic-design/design-generators/archive/"
    )
    if not requirement.url.startswith(expected_url_prefix):
        raise ValueError("published README must use the repository source archive")

    if "#subdirectory=lib/laygen" not in requirement.url:
        raise ValueError("published README must select the lib/laygen subdirectory")

    archive_revision = requirement.url.removeprefix(expected_url_prefix).split(".", 1)[
        0
    ]
    if not ARCHIVE_REVISION_PATTERN.fullmatch(archive_revision):
        raise ValueError("published README laygen archive must pin a full commit SHA")

    return requirement_text


def verify_laygen_install(card_text: str) -> None:
    """Install the card's laygen archive into a temp target and import its helper."""
    requirement = extract_laygen_requirement(card_text)

    with tempfile.TemporaryDirectory(prefix="collage-laygen-smoke-") as temp_directory:
        target_directory = Path(temp_directory) / "target"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target_directory),
                requirement,
            ],
            check=True,
        )

        environment = os.environ.copy()
        inherited_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(target_directory), inherited_pythonpath) if item
        )
        environment["LAYGEN_SMOKE_TARGET"] = str(target_directory)
        import_check = (
            "import os\n"
            "from pathlib import Path\n"
            "import laygen\n"
            "from laygen.common.randomness import randn, resolve_torch_generator\n"
            "target = Path(os.environ['LAYGEN_SMOKE_TARGET']).resolve()\n"
            "origin = Path(laygen.__file__).resolve()\n"
            "assert origin.is_relative_to(target), (origin, target)\n"
            "assert callable(randn) and callable(resolve_torch_generator)\n"
        )
        subprocess.run(
            [sys.executable, "-c", import_check],
            check=True,
            env=environment,
        )


def _read_source_files(revision: str) -> dict[str, bytes]:
    return prepare_upload_files(
        PIPELINE_SOURCE.read_bytes(),
        README_SOURCE.read_text(encoding="utf-8"),
        revision,
    )


def _source_asset_files(api: HfApi, *, revision: str) -> set[str]:
    return {
        path
        for path in api.list_repo_files(
            HUB_REPOSITORY_ID,
            repo_type="model",
            revision=revision,
        )
        if path.startswith(SOURCE_ASSET_PREFIX)
    }


def verify_hub_write_access(api: HfApi) -> None:
    """Fail before upload when the token cannot write to this model repository."""
    try:
        api.auth_check(
            repo_id=HUB_REPOSITORY_ID,
            repo_type="model",
            write=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not verify HF_TOKEN content-write access to {HUB_REPOSITORY_ID}. "
            "Add this repository to the Hugging Face fine-grained token with write permission."
        ) from error


def _verify_uploaded_files(
    api: HfApi,
    files: dict[str, bytes],
    *,
    revision: str,
) -> str:
    from huggingface_hub import hf_hub_download

    repository_files = set(
        api.list_repo_files(
            HUB_REPOSITORY_ID,
            repo_type="model",
            revision=revision,
        )
    )
    missing_files = set(files) - repository_files
    if missing_files:
        raise RuntimeError(f"Hub publication is missing files: {sorted(missing_files)}")

    card_text = ""
    for filename, expected_data in files.items():
        downloaded_path = hf_hub_download(
            repo_id=HUB_REPOSITORY_ID,
            filename=filename,
            repo_type="model",
            revision=revision,
            token=api.token,
        )
        downloaded_data = Path(downloaded_path).read_bytes()
        if downloaded_data != expected_data:
            raise RuntimeError(f"Hub read-back differs from the staged {filename}")

        if filename == "README.md":
            card_text = downloaded_data.decode("utf-8")

    return card_text


def write_upload_staging(staging_directory: Path, files: dict[str, bytes]) -> None:
    """Write only the two explicitly allowed files into an empty staging directory."""
    if set(files) != set(UPLOAD_ALLOW_PATTERNS):
        raise ValueError(
            "Hub staging input must contain only README.md and pipeline.py"
        )

    if any(staging_directory.iterdir()):
        raise ValueError("Hub staging directory must be empty")

    for filename, contents in files.items():
        (staging_directory / filename).write_bytes(contents)

    staged_files = {path.name for path in staging_directory.iterdir()}
    if staged_files != set(UPLOAD_ALLOW_PATTERNS):
        raise RuntimeError(f"unexpected files in Hub staging directory: {staged_files}")


def publish(*, token: str, revision: str) -> None:
    """Upload the two declared files and verify them without touching other paths."""
    from huggingface_hub import HfApi

    files = _read_source_files(revision)
    api = HfApi(token=token)
    verify_hub_write_access(api)
    parent_commit = api.model_info(HUB_REPOSITORY_ID, revision="main").sha
    source_assets_before = _source_asset_files(api, revision=parent_commit)
    if not source_assets_before:
        raise RuntimeError("the existing Hub repository has no source-assets files")

    with tempfile.TemporaryDirectory(prefix="collage-hub-publish-") as temp_directory:
        staging_directory = Path(temp_directory)
        write_upload_staging(staging_directory, files)

        commit_info = api.upload_folder(
            repo_id=HUB_REPOSITORY_ID,
            repo_type="model",
            folder_path=str(staging_directory),
            allow_patterns=list(UPLOAD_ALLOW_PATTERNS),
            commit_message=f"Publish Collage Diffusion pipeline at {revision[:12]}",
            parent_commit=parent_commit,
        )

    published_revision = commit_info.oid
    card_text = _verify_uploaded_files(
        api,
        files,
        revision=published_revision,
    )
    source_assets_after = _source_asset_files(api, revision=published_revision)
    if source_assets_after != source_assets_before:
        raise RuntimeError(
            "Hub publication changed the existing source-assets file set"
        )

    verify_laygen_install(card_text)
    print(
        f"Published and verified {', '.join(UPLOAD_ALLOW_PATTERNS)} at "
        f"{published_revision}; preserved {len(source_assets_after)} source-assets files."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=os.environ.get("GITHUB_SHA"),
        help="full commit SHA used to pin Hub code and package archives",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and list the exact upload files without contacting the Hub",
    )
    args = parser.parse_args(argv)
    if not args.revision:
        parser.error("--revision or GITHUB_SHA is required")

    if args.dry_run:
        files = _read_source_files(args.revision)
        for filename, contents in sorted(files.items()):
            print(f"{filename}: {len(contents)} bytes")
        print(f"Hub target: {HUB_REPOSITORY_ID}")
        print("Hub write: skipped (--dry-run)")
        return 0

    if not ARCHIVE_REVISION_PATTERN.fullmatch(args.revision):
        parser.error("--revision must be a full lowercase 40-character commit SHA")

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        parser.error("HF_TOKEN is required for Hub publication")

    publish(token=token, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
