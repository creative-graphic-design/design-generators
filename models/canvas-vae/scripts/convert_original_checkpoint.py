"""Convert a CanvasVAE checkpoint into a ``save_pretrained`` pipeline directory.

Accepts a TensorFlow checkpoint prefix written by the original trainer (for
example ``<job-dir>/checkpoints/final.ckpt``, which needs the ``convert`` extra)
or a Lightning ``.ckpt`` file written by ``traingen fit``, which needs no
TensorFlow extra. The lookup tables come from the ``vocabulary.json`` of the
data the checkpoint was trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from jaxtyping import Shaped

from canvas_vae import CanvasVAEConfig, CanvasVAEModel, CanvasVAEPipeline
from canvas_vae.conversion import (
    convert_tensorflow_variables,
    load_lightning_state_dict,
)
from canvas_vae.processing_canvas_vae import build_vocabularies


def read_tensorflow_variables(prefix: Path) -> dict[str, Shaped[np.ndarray, "..."]]:
    """Read every variable of a TensorFlow checkpoint."""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(str(prefix))
    return {key: reader.get_tensor(key) for key in reader.get_variable_to_shape_map()}


def main() -> None:
    """Convert and save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="TensorFlow checkpoint prefix or Lightning .ckpt file.",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        required=True,
        help="vocabulary.json of the training data, such as .cache/canvas-vae/original/data/rico/vocabulary.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Pipeline output directory."
    )
    args = parser.parse_args()
    vocabularies = build_vocabularies(
        json.loads(args.vocabulary.read_text(encoding="utf-8"))
    )
    config = CanvasVAEConfig(vocabularies=vocabularies)
    if args.checkpoint.is_file():
        state_dict = load_lightning_state_dict(args.checkpoint)
    else:
        state_dict = convert_tensorflow_variables(
            read_tensorflow_variables(args.checkpoint), config
        )

    model = CanvasVAEModel(config)
    model.load_state_dict(state_dict, strict=True)
    CanvasVAEPipeline(model=model).save_pretrained(args.output_dir)
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
