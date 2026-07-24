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


def test_generation_hook_liveness_rejects_input_ids_dtype_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_action.generation_layout_action"], "beartype.beartype"):
            from layout_action.generation_layout_action import LayoutActionSamplingConfig, sample_action_tokens
            from layout_action.configuration_layout_action import LayoutActionSamplingMode

        class DummyModel:
            def get_block_size(self):
                return 8

            def __call__(self, input_ids):
                batch, seq = input_ids.shape
                return type("Out", (), {"logits": torch.zeros(batch, seq, 4)})()

        sample_action_tokens(
            DummyModel(),
            torch.zeros(1, 1),
            max_new_tokens=1,
            sampling=LayoutActionSamplingConfig(mode=LayoutActionSamplingMode.greedy),
        )
        """,
        "sample_action_tokens",
    )


def test_generation_hook_liveness_rejects_logits_rank_mismatch() -> None:
    _assert_probe_rejected(
        """
        import torch
        from jaxtyping import install_import_hook

        with install_import_hook(["layout_action.generation_layout_action"], "beartype.beartype"):
            from layout_action.generation_layout_action import top_k_logits

        top_k_logits(torch.zeros(1, 2, 4), 1)
        """,
        "top_k_logits",
    )


def test_generation_without_hook_accepts_annotation_mismatches() -> None:
    _assert_probe_passes(
        """
        import torch

        from layout_action.configuration_layout_action import LayoutActionSamplingMode
        from layout_action.generation_layout_action import LayoutActionSamplingConfig, sample_action_tokens, top_k_logits

        class DummyModel:
            def get_block_size(self):
                return 8

            def __call__(self, input_ids):
                batch, seq = input_ids.shape
                return type("Out", (), {"logits": torch.zeros(batch, seq, 4)})()

        assert sample_action_tokens(
            DummyModel(),
            torch.zeros(1, 1),
            max_new_tokens=1,
            sampling=LayoutActionSamplingConfig(mode=LayoutActionSamplingMode.greedy),
        ).shape == (1, 2)
        assert top_k_logits(torch.zeros(1, 2, 4), 1).shape == (1, 2, 4)
        """
    )
