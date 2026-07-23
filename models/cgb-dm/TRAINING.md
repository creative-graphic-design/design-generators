# Training CGB-DM

CGB-DM training is driven by the shared class-path
[LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
entry point. The model package provides the `LightningModule`,
`LightningDataModule`, and YAML configs; optimizer and scheduler defaults are
declared in those YAML files through LightningCLI class paths.

```bash
uv run --package cgb-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml
```

Use `models/cgb-dm/configs/training/smoke.yaml` for local and CI configuration
smoke checks. Full 500-epoch PKU/CGL training runs outside regular CI and is
reported as run metadata until the paired full-run statistical comparison is
complete.

## Parity Stages

| Stage | Scope |
| --- | --- |
| S0 | Batch loading and field ordering from the CGB-DM asset layout. |
| S1 | Training-step tensor trace through timestep sampling, masking, and noise addition. |
| S2 | Epsilon prediction and loss comparison against generated reference metadata. |

## S0 Data Order Deviation

The original training loader iterates `os.listdir(train/inpaint)`, so its row
order depends on filesystem state. The package default remains sorted for
deterministic training data access. Parity captures the original loader's
observed filename order into a regenerated `.cache` manifest and replays that
manifest explicitly for vendor-compatible S0 checks. The manifest is
regeneration metadata and is not committed.

The gated PKU check now runs a fixed-batch original-code training step and the
package training step with the same RNG state. S1 compares timestep ids, noise,
condition masks, noised layouts, predicted epsilon, loss, and content-graphic
balance weights. S2 compares clipped gradients, post-Adam parameters, and Adam
state after one update. Floating comparisons use narrow tolerances because the
same CUDA model path produces small roundoff differences across separate
forward/backward executions.

Full [PKU PosterLayout](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023)
and CGL asset runs remain outside regular CI.
