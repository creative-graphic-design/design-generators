from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType


def load_check_changed_urls() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / ("check_changed_urls.py")
    )
    spec = importlib.util.spec_from_file_location("check_changed_urls", module_path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_changed_urls = load_check_changed_urls()
HTTPS = "https://"
HTTP = "http://"
GITHUB_HOST = "github.com"
HUGGING_FACE_HOST = "huggingface.co"
EXAMPLE_HOST = "example.com"
TEMPORARY_HOST = "temporary.example.test"
REMOVED_HOST = "removed.example.test"
DESIGN_GENERATORS_URL = (
    f"{HTTPS}{GITHUB_HOST}/creative-graphic-design/design-generators"
)
SMARTTEXT_BAD_URL = f"{HTTPS}{GITHUB_HOST}/chenqi008/SmartText"
SMARTTEXT_GOOD_URL = f"{HTTPS}{GITHUB_HOST}/intchous/SmartText"


def test_extract_urls_from_added_lines_only_reads_added_external_urls() -> None:
    diff_text = "\n".join(
        [
            "diff --git a/README.md b/README.md",
            "--- a/README.md",
            "+++ b/README.md",
            "@@ -0,0 +1,2 @@",
            f"-removed {HTTPS}{REMOVED_HOST}/not-added",
            f"+added [repo]({DESIGN_GENERATORS_URL}).",
            f"+two {SMARTTEXT_GOOD_URL} and {HTTP}{EXAMPLE_HOST}/demo",
            "",
        ]
    )

    assert check_changed_urls.extract_urls_from_added_lines(diff_text) == [
        check_changed_urls.ChangedUrl(
            url=DESIGN_GENERATORS_URL,
            source="README.md:1",
        ),
        check_changed_urls.ChangedUrl(
            url=SMARTTEXT_GOOD_URL,
            source="README.md:2",
        ),
        check_changed_urls.ChangedUrl(
            url=f"{HTTP}{EXAMPLE_HOST}/demo",
            source="README.md:2",
        ),
    ]


def test_extract_urls_from_added_lines_ignores_lycheeignore_patterns() -> None:
    diff_text = "\n".join(
        [
            "diff --git a/.lycheeignore b/.lycheeignore",
            "--- a/.lycheeignore",
            "+++ b/.lycheeignore",
            "@@ -0,0 +1,1 @@",
            f"+^{HTTPS}{HUGGING_FACE_HOST}/creative-graphic-design/.*",
            "",
        ]
    )

    assert check_changed_urls.extract_urls_from_added_lines(diff_text) == []


def test_check_changed_urls_passes_for_live_url_with_mocked_checker() -> None:
    urls = [
        check_changed_urls.ChangedUrl(
            DESIGN_GENERATORS_URL,
            "README.md:1",
        )
    ]

    results = check_changed_urls.check_changed_urls(
        urls,
        ignore_patterns=[],
        checker=lambda url: check_changed_urls.UrlCheckResult(
            url=url,
            outcome="ok",
            status=200,
        ),
    )

    assert results == [
        check_changed_urls.UrlCheckResult(
            url=DESIGN_GENERATORS_URL,
            outcome="ok",
            status=200,
        )
    ]
    assert check_changed_urls.report_results(results) == 0


def test_check_changed_urls_fails_for_added_404_with_mocked_checker() -> None:
    urls = [
        check_changed_urls.ChangedUrl(
            SMARTTEXT_BAD_URL,
            "models/smarttext/README.md:58",
        )
    ]

    results = check_changed_urls.check_changed_urls(
        urls,
        ignore_patterns=[],
        checker=lambda url: check_changed_urls.UrlCheckResult(
            url=url,
            outcome="fail",
            status=404,
        ),
    )

    assert results == [
        check_changed_urls.UrlCheckResult(
            url=SMARTTEXT_BAD_URL,
            outcome="fail",
            status=404,
        )
    ]
    assert check_changed_urls.report_results(results) == 1


def test_check_changed_urls_skips_ignored_urls_without_calling_checker() -> None:
    urls = [
        check_changed_urls.ChangedUrl(
            f"{HTTPS}{HUGGING_FACE_HOST}/creative-graphic-design/not-yet-published",
            "README.md:1",
        )
    ]
    patterns = check_changed_urls.load_ignore_patterns(
        Path(__file__).resolve().parents[1] / ".lycheeignore"
    )

    def checker(url: str) -> object:
        raise AssertionError(f"checker should not be called for {url}")

    results = check_changed_urls.check_changed_urls(
        urls,
        ignore_patterns=patterns,
        checker=checker,
    )

    assert results == [
        check_changed_urls.UrlCheckResult(
            url=f"{HTTPS}{HUGGING_FACE_HOST}/creative-graphic-design/not-yet-published",
            outcome="ignored",
        )
    ]
    assert check_changed_urls.report_results(results) == 0


def test_check_changed_urls_treats_5xx_as_warning_success() -> None:
    urls = [
        check_changed_urls.ChangedUrl(
            f"{HTTPS}{TEMPORARY_HOST}/service",
            "README.md:1",
        )
    ]

    results = check_changed_urls.check_changed_urls(
        urls,
        ignore_patterns=[],
        checker=lambda url: check_changed_urls.UrlCheckResult(
            url=url,
            outcome="warning",
            status=503,
        ),
    )

    assert check_changed_urls.report_results(results) == 0
