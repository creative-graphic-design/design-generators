"""Publish workspace-member guides outside the generated API tree."""

import os
from pathlib import Path
import re
import shutil
import tomllib

from gen_api_pages import GROUPS

ROOT = Path(__file__).resolve().parents[1]
GUIDES = {
    "README.md": ("index.md", "package"),
    "REPRODUCING.md": ("reproducing.md", "refresh-cw"),
    "TRAINING.md": ("training.md", "graduation-cap"),
}
LINK = re.compile(r"\]\((?P<link>(?:lib|models|docs)/[^)\s]+\.md(?:#[^)\s]+)?)\)")


def _replace(root: Path, output: Path, match: re.Match[str]) -> str:
    path, _, fragment = match["link"].partition("#")
    parts = Path(path).parts
    if (
        len(parts) == 3
        and parts[0] in GROUPS
        and parts[2] in GUIDES
        and (root / path).is_file()
    ):
        target = Path("docs", GROUPS[parts[0]], parts[1], GUIDES[parts[2]][0])
    elif path.startswith("docs/"):
        target = Path(path)
    else:
        return match.group(0)

    relative = Path(os.path.relpath(target, output.parent))
    route = (
        "./"
        if relative == Path("index.md")
        else f"{relative.with_suffix('').as_posix()}/"
    )
    return f"]({route}{f'#{fragment}' if fragment else ''})"


def publish(root: Path = ROOT) -> None:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    summaries = {group: [] for group in GROUPS.values()}
    for group in GROUPS.values():
        shutil.rmtree(root / "docs" / group, ignore_errors=True)

    for pattern in data["tool"]["uv"]["workspace"]["members"]:
        for member in sorted(root.glob(pattern)):
            if not (member / "pyproject.toml").is_file():
                continue

            group = GROUPS[member.relative_to(root).parts[0]]
            written = []
            for source_name, (target_name, icon) in GUIDES.items():
                source = member / source_name
                if not source.is_file():
                    continue

                output = Path("docs", group, member.name, target_name)
                text = source.read_text(encoding="utf-8")
                if text.startswith("---\n") and "\n---\n" in text:
                    text = text.split("\n---\n", 1)[1]

                output_path = root / output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                body = LINK.sub(
                    lambda match: _replace(root, output, match), text
                ).rstrip()
                output_path.write_text(
                    f"---\nicon: lucide/{icon}\ntags:\n  - {group.title()}\n---\n\n{body}\n",
                    encoding="utf-8",
                )
                written.append((source_name, target_name))

            if written:
                summaries[group].append((member.name, written))

    for group, members in summaries.items():
        lines = [f"* [{member}]({member}/index.md)" for member, _ in members]
        lines += [
            f"  * [{name.removesuffix('.md').title()}]({member}/{target})"
            for member, written in members
            for name, target in written
            if name != "README.md"
        ]
        if lines:
            summary = root / "docs" / group / "SUMMARY.md"
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    publish()
