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

## Reproducibility

Reproduce original-implementation agreement by downloading the vendor assets,
generating fixed-seed references on one GPU, converting the checkpoint, and
running the marked parity tests.

```bash
uv run --package layousyn python models/layousyn/scripts/download_original.py
CUDA_VISIBLE_DEVICES=5 uv run --package layousyn --extra vendor python models/layousyn/scripts/save_reference_outputs.py \
  --ckpt /path/to/model.pt \
  --ckpt-config /path/to/config.json \
  --output-dir /tmp/layousyn-reference \
  --seed 0 \
  --caption "a person sitting on a bench" \
  --concept person \
  --concept bench
uv run --package layousyn pytest models/layousyn/tests -m vendor_parity
uv run --package layousyn python models/layousyn/scripts/convert_checkpoint.py \
  --checkpoint-path /path/to/model.pt \
  --config-path /path/to/config.json \
  --output-dir /tmp/layousyn-converted \
  --variant-name grit
uv run --package layousyn python - <<'PY'
from layousyn import LayouSynPipeline
pipe = LayouSynPipeline.from_pretrained("/tmp/layousyn-converted")
print(pipe)
PY
```

## License

The original Lay-Your-Scene implementation is CC BY-NC 4.0. Converted
checkpoints and model cards must retain the non-commercial notice.

## Citation

```text
Use the original Lay-Your-Scene citation from the upstream repository.
```
