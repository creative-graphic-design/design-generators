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


def test_selection_hook_liveness_rejects_content_bbox_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import numpy as np
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutprompter.selection"], "beartype.beartype"):
            from layoutprompter.selection import ContentAwareExemplarSelection

        selector = ContentAwareExemplarSelection(train_data=[], candidate_size=0, num_prompt=0)
        selector._to_binary_mask(np.zeros((1, 4), dtype=np.int64))
        """,
        "ContentAwareExemplarSelection._to_binary_mask",
    )


def test_selection_hook_liveness_rejects_content_bbox_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import numpy as np
        from jaxtyping import install_import_hook

        with install_import_hook(["layoutprompter.selection"], "beartype.beartype"):
            from layoutprompter.selection import ContentAwareExemplarSelection

        selector = ContentAwareExemplarSelection(train_data=[], candidate_size=0, num_prompt=0)
        selector._to_binary_mask(np.zeros((1, 1, 4), dtype=np.float32))
        """,
        "ContentAwareExemplarSelection._to_binary_mask",
    )


def test_selection_without_hook_accepts_dtype_annotation_mismatch() -> None:
    _assert_probe_passes(
        """
        import numpy as np

        from layoutprompter.selection import ContentAwareExemplarSelection

        selector = ContentAwareExemplarSelection(train_data=[], candidate_size=0, num_prompt=0)
        assert selector._to_binary_mask(np.zeros((1, 4), dtype=np.int64)).shape == (150, 102)
        """
    )
