# LayouSyn

LayouSyn converts the Lay-Your-Scene text-to-layout model into a Diffusers-style
pipeline that returns the shared layout schema: `bbox`, `labels`, `mask`, and
`id2label`.

## Usage

```bash
uv run --package layousyn python - <<'PY'
import torch
from layousyn import LayouSynDiTModel, LayouSynPipeline, LayouSynProcessor, LayouSynScheduler

model = LayouSynDiTModel(
    model_name="DiT-D1-H32-N1",
    max_in_len=2,
    max_y_len=3,
    concept_in_channels=4,
    y_in_channels=4,
)
pipe = LayouSynPipeline(
    model=model,
    scheduler=LayouSynScheduler(num_train_timesteps=2),
    processor=LayouSynProcessor(max_in_len=2, max_y_len=3, concept_in_channels=4, y_in_channels=4),
)
out = pipe(
    labels=[["person", "bench"]],
    caption_embeds=torch.zeros(1, 3, 4),
    caption_padding_mask=torch.ones(1, 3, dtype=torch.bool),
    concept_embeds=torch.zeros(1, 2, 4),
    num_inference_steps=1,
    seed=0,
)
print(out.bbox.shape)
PY
```

## Checkpoints

Planned converted Hub repos are:

```text
creative-graphic-design/layousyn-grit
creative-graphic-design/layousyn-coco-grounded
creative-graphic-design/layousyn-grit-ft-coco-grounded
```

Hub publishing is intentionally not performed in this implementation PR.

## Datasets

COCO-17, COCO-Caption-Grounded, GriT, and NSR-1K are not yet available in the
`creative-graphic-design` Hugging Face org. Until those imports exist, parity
and conversion scripts follow the original repository's dataset/download path.

## Parity Results

Local fixtures compare the converted Diffusers pipeline against the original
Lay-Your-Scene implementation on the Grit checkpoint using real
`google/t5-v1_1-base` caption embeddings and
`sentence-transformers/sentence-t5-base` concept embeddings.

| Check | Cases | Criterion | Result |
| --- | ---: | --- | --- |
| Alpha-scale beta respacing | 1 | `timestep_map` exact, helper betas exact, scheduler betas max abs <= 2e-8 | pass; helper exact, scheduler max abs <= 1.85e-8 |
| Denoiser logits | 1 | max abs <= 2e-3, max rel <= 2e-3 | pass; max abs 1.2935e-3 |
| First DDIM scheduler step | 1 | `pred_xstart` max abs <= 4e-5 and `prev_sample` max abs <= 2e-5 | pass; 3.0518e-5 / 4.7684e-7 |
| Full 40-step sample public bbox | 1 | normalized public `xywh` max abs <= 2e-3 | pass; max abs 5.2923e-4 |

## Reproducibility

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
  --output-dir /tmp/layousyn-reference \
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
  --output-dir /tmp/layousyn-converted \
  --variant-name grit

CUDA_VISIBLE_DEVICES=5 \
LAYOUSYN_REFERENCE_DIR=/tmp/layousyn-reference \
LAYOUSYN_CONVERTED_DIR=/tmp/layousyn-converted \
uv run --package layousyn pytest models/layousyn/tests -m vendor_parity -q

CUDA_VISIBLE_DEVICES=5 uv run --package layousyn python - <<'PY'
import torch
from layousyn import LayouSynPipeline

inputs = torch.load("/tmp/layousyn-reference/inputs.pt", map_location="cuda")
pipe = LayouSynPipeline.from_pretrained("/tmp/layousyn-converted").to("cuda")
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

The CUDA 5 Grit checkpoint parity run for this PR passes 4/4 marked checks:
alpha-scale beta respacing, denoiser logits, first scheduler step, and full
sample public layout output. The `from_pretrained` smoke prints
`torch.Size([1, 60, 4]) {0: 'person', 1: 'bench'}`.

## License

The original Lay-Your-Scene implementation is CC BY-NC 4.0. Converted
checkpoints and model cards must retain the non-commercial notice.

## Citation

```text
Use the original Lay-Your-Scene citation from the upstream repository.
```
