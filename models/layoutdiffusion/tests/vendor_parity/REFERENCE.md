# LayoutDiffusion Vendor Parity References

Heavyweight vendor outputs are generated locally under
`.cache/layoutdiffusion/references/<dataset>/` and are not committed.

Source assets are downloaded from the Hugging Face repository
`Junyi42/layoutdiffusion`. The reference generator runs the vendored
`improved-diffusion` implementation from an isolated copy under
`.cache/layoutdiffusion/vendor_run/improved-diffusion` and applies only the
PyTorch 2 CUDA tensor-index compatibility patch described in the generated
`meta.json`.

Regenerate references with:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layoutdiffusion --extra vendor \
  --with spacy --with pyyaml --with sacremoses \
  python models/layoutdiffusion/scripts/generate_reference_outputs.py \
  --dataset all \
  --seed 101
```

The resulting `vendor_reference.pt` files contain tokenizer text and ids,
selected scheduler buffers, fixed denoiser logits, and one full unconditional
sample token sequence.
