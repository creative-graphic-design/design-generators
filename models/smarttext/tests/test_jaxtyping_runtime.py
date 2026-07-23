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


def test_scorer_hook_liveness_rejects_score_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["smarttext.modeling_smarttext"], "beartype.beartype"):
            from smarttext.modeling_smarttext import SmartTextScorerOutput

        SmartTextScorerOutput(scores=torch.zeros(2, dtype=torch.long))
        """,
        "SmartTextScorerOutput.__init__",
    )


def test_scorer_hook_liveness_rejects_score_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["smarttext.modeling_smarttext"], "beartype.beartype"):
            from smarttext.modeling_smarttext import SmartTextScorerOutput

        SmartTextScorerOutput(scores=torch.zeros(1, 2))
        """,
        "SmartTextScorerOutput.__init__",
    )


def test_scorer_without_hook_accepts_output_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from smarttext.modeling_smarttext import SmartTextScorerOutput

        assert SmartTextScorerOutput(scores=torch.zeros(2, dtype=torch.long)).scores.shape == (2,)
        assert SmartTextScorerOutput(scores=torch.zeros(1, 2)).scores.shape == (1, 2)
        """
    )
