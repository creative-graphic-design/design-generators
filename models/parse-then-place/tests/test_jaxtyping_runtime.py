import subprocess
import sys
import textwrap


def _assert_probe_rejected(code: str, expected: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected in result.stderr


def _assert_probe_passes(code: str) -> None:
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


def test_processor_hook_liveness_rejects_generated_id_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["parse_then_place.processing_parse_then_place"], "beartype.beartype"
        ):
            from parse_then_place import ParseThenPlaceProcessor

        ParseThenPlaceProcessor.from_config("rico").postprocess_ir(torch.zeros(1, 2))
        """,
        "ParseThenPlaceProcessor.postprocess_ir",
    )


def test_processor_hook_liveness_rejects_generated_id_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(
            ["parse_then_place.processing_parse_then_place"], "beartype.beartype"
        ):
            from parse_then_place import ParseThenPlaceProcessor

        ParseThenPlaceProcessor.from_config("rico").postprocess_ir(torch.zeros(1, 2, 1, dtype=torch.long))
        """,
        "ParseThenPlaceProcessor.postprocess_ir",
    )


def test_processor_without_hook_accepts_string_sequence_path() -> None:
    _assert_probe_passes(
        """
        from parse_then_place import ParseThenPlaceProcessor

        assert ParseThenPlaceProcessor.from_config("rico").postprocess_ir(["TEXT 0 0 1 1"]) == ["text 0 0 1 1"]
        """
    )
