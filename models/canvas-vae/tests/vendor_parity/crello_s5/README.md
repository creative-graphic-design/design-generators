# Crello full-run tooling

These scripts run both CanvasVAE systems with one fixed PixelVAE embedding cache. Use them after the required pre-run checks and comparison scope are recorded on [issue #31](https://github.com/creative-graphic-design/design-generators/issues/31). Set `CANVAS_VAE_S5_SOURCE_COMMIT` to the exact commit to be checked before training.

Required environment variables:

- `CANVAS_VAE_GPU`
- `CANVAS_VAE_S5_RUN_ROOT`
- `CANVAS_VAE_PACKAGE_ENV` and `CANVAS_VAE_ORIGINAL_ENV`
- `CANVASVAE_TF_GPU_SITE`
- `CANVAS_VAE_PACKAGE_FREEZE_REFERENCE` and `CANVAS_VAE_ORIGINAL_FREEZE_REFERENCE`

Optional input overrides are `CANVAS_VAE_S5_INPUTS`, `CANVAS_VAE_ORIGINAL_DATA`, `CANVAS_VAE_PACKAGE_DATA`, and `CANVAS_VAE_EMBEDDING_CACHE`. The default input root is `.cache/canvas-vae/s5-path-a/inputs`.

Run each isolated preflight, then launch both lanes from the repository root:

```bash
bash models/canvas-vae/tests/vendor_parity/crello_s5/scripts/preflight_original.sh
bash models/canvas-vae/tests/vendor_parity/crello_s5/scripts/preflight_package.sh
bash models/canvas-vae/tests/vendor_parity/crello_s5/scripts/launch_s5.sh
```

The launcher refuses to reuse a run manifest and checks memory, compute applications, both preflights, and the pinned source commit. It starts both lanes and records their PIDs. Each lane checks the source tree before every training seed and waits for its memory margin. The package lane runs seeds 0–2 by default; pass explicit seed numbers to run selected seeds in a fresh run root.

Set `CANVAS_VAE_ORIGINAL_SEED0_RERUN_OUTPUT` to a new directory under the run root before using `scripts/original_seed0_rerun.sh`. Inputs, environments, checkpoints, logs, and run records stay under `.cache`; do not commit them.
