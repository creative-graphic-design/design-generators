# Reproducing CGB-DM Checks

The parity workflow stages CGB-DM through S0-S2 first and does not download the
13 GB dataset during tests.

```bash
uv run --package cgb-dm python models/cgb-dm/scripts/download_original_assets.py
CUDA_VISIBLE_DEVICES=0 uv run --package cgb-dm python models/cgb-dm/scripts/generate_reference_outputs.py --dataset pku_posterlayout
PARITY_REQUIRE=1 uv run --package cgb-dm pytest models/cgb-dm/tests/vendor_parity -m vendor_parity
uv run --package cgb-dm python models/cgb-dm/scripts/convert_training_checkpoint.py --checkpoint .cache/cgb-dm/checkpoints/example.ckpt --output-dir .cache/cgb-dm/converted/pku
uv run --package cgb-dm python models/cgb-dm/scripts/smoke_from_pretrained.py --path .cache/cgb-dm/converted/pku
```

The current first PR does not claim full PKU/CGL training or generated-metric
equivalence.
