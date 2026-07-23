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


def test_agent_hook_liveness_rejects_label_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_gpt.agent"], "beartype.beartype"):
            from layout_gpt import LayoutGPTAgent

        LayoutGPTAgent()(
            prompt="make a layout",
            train_examples=[],
            labels=torch.zeros(1, 2),
        )
        """,
        "LayoutGPTAgent.__call__",
    )


def test_agent_hook_liveness_rejects_bbox_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_gpt.agent"], "beartype.beartype"):
            from layout_gpt import LayoutGPTAgent

        LayoutGPTAgent()(
            prompt="make a layout",
            train_examples=[],
            bbox=torch.zeros(1, 2),
        )
        """,
        "LayoutGPTAgent.__call__",
    )


def test_schema_without_hook_converts_output() -> None:
    _assert_probe_passes(
        """
        from layout_gpt.schema import LayoutGPTOutput, LayoutItem2D

        output = LayoutGPTOutput(
            prompt="p",
            canvas_size=256,
            items=[LayoutItem2D(label="text", left=0, top=0, width=1, height=1)],
            raw_text="",
            id2label={0: "text"},
        ).to_layout_generation_output()
        assert output.bbox.shape[-1] == 4
        """
    )
