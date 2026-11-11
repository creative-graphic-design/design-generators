# LayoutDiffusion Vendor Parity References

Heavyweight vendor outputs are generated locally under
`.cache/layout-diffusion/references/<dataset>/` and are not committed.

Source assets are downloaded from the Hugging Face repository
`Junyi42/layoutdiffusion`. The reference generator runs the vendored
`improved-diffusion` implementation from an isolated copy under
`.cache/layout-diffusion/vendor_run/improved-diffusion` and applies only the
PyTorch 2 CUDA tensor-index compatibility patch described in the generated
`meta.json`.

Choose a free GPU and substitute its index for `<gpu-index>` in the CUDA command below.

Regenerate references with:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-diffusion --extra vendor \
  --with spacy --with pyyaml --with sacremoses \
  models/layout-diffusion/scripts/generate_reference_outputs.py \
  --dataset all \
  --seed 101
```

The resulting `vendor_reference.pt` files contain tokenizer text and ids,
selected scheduler buffers, fixed denoiser logits, and one full unconditional
sample token sequence.
