# Reproducing LayouSyn Parity

This guide reproduces the original-implementation agreement checks for the LayouSyn package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Reproduce original-implementation agreement by generating CUDA fixed-seed vendor
references with real T5 and SentenceTransformer embeddings, converting the same
checkpoint, and running the marked parity tests against those external assets.

```bash
SNAPSHOT="$(uv run --package layousyn python models/layousyn/scripts/download_original.py)"
CKPT="${SNAPSHOT}/grit/model.pt"
CONFIG="${SNAPSHOT}/grit/config.json"

CUDA_VISIBLE_DEVICES=5 uv run --package layousyn --extra vendor python models/layousyn/scripts/save_reference_outputs.py \
  --vendor-root vendor/lay-your-scene \
  --ckpt "${CKPT}" \
  --ckpt-config "${CONFIG}" \
  --output-dir .cache/layousyn/reference \
  --seed 0 \
  --caption "a person sitting on a bench" \
  --concept person \
  --concept bench \
  --cfg-scale 2.0 \
  --aspect-ratio 1.0 \
  --num-sampling-steps 40

uv run --package layousyn python models/layousyn/scripts/convert_checkpoint.py \
  --checkpoint-path "${CKPT}" \
  --config-path "${CONFIG}" \
  --output-dir .cache/layousyn/converted \
  --variant-name grit

CUDA_VISIBLE_DEVICES=5 \
LAYOUSYN_REFERENCE_DIR=.cache/layousyn/reference \
LAYOUSYN_CONVERTED_DIR=.cache/layousyn/converted \
uv run --package layousyn pytest models/layousyn/tests -m vendor_parity -q

CUDA_VISIBLE_DEVICES=5 uv run --package layousyn python - <<'PY'
import torch
from layousyn import LayouSynPipeline

inputs = torch.load(".cache/layousyn/reference/inputs.pt", map_location="cuda")
pipe = LayouSynPipeline.from_pretrained(".cache/layousyn/converted").to("cuda")
pipe.set_progress_bar_config(disable=True)
out = pipe(
    prompt="a person sitting on a bench",
    labels=[["person", "bench"]],
    caption_embeds=inputs["pipeline_caption_embeds"],
    caption_padding_mask=inputs["pipeline_caption_padding_mask"],
    concept_embeds=inputs["pipeline_concept_embeds"],
    aspect_ratio=inputs["pipeline_aspect_ratio"],
    num_inference_steps=1,
    guidance_scale=2.0,
    generator=torch.Generator(device="cuda").manual_seed(0),
)
print(out.bbox.shape, out.id2label)
PY
```

The CUDA 5 Grit checkpoint parity run passes 4/4 marked checks:
alpha-scale beta respacing, denoiser logits, first scheduler step, and full
sample public layout output. The `from_pretrained` smoke prints
`torch.Size([1, 60, 4]) {0: 'person', 1: 'bench'}`.
