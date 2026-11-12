from __future__ import annotations

import subprocess
from pathlib import Path


def test_production_segment_accepts_shortened_end_epoch() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_crello_training_segment.sh"
    guard = "((start_epoch < end_epoch && end_epoch <= 250))"
    assert guard in runner.read_text(encoding="utf-8")

    for start, end, accepted in (
        (193, 227, True),
        (193, 193, False),
        (193, 251, False),
    ):
        result = subprocess.run(
            ["bash", "-c", f"start_epoch={start}; end_epoch={end}; {guard}"],
            check=False,
        )
        assert (result.returncode == 0) is accepted


def test_segment_trainer_args_use_absolute_step_limit_without_epoch_cap() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_crello_training_segment.sh"
    script = runner.read_text(encoding="utf-8")
    fit_args = script.split("fit_args=(", maxsplit=1)[1].split("\n)", maxsplit=1)[0]
    shell = f"""
training_config=config.yaml
accelerator=gpu
run_dir=run
target_global_step=350000
data_dir=data
batch_size=64
start_epoch=245
start_global_step=343000
fit_args=(
{fit_args}
)
printf '%s\\n' "${{fit_args[@]}}"
"""
    result = subprocess.run(
        ["bash", "-c", shell],
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = result.stdout.splitlines()

    assert "--trainer.max_epochs=-1" in arguments
    assert "--trainer.max_epochs=250" not in arguments
    assert "--trainer.max_steps=350000" in arguments
