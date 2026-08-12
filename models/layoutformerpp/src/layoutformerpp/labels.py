"""Semantic label translation for LayoutFormer++ token sequences."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypedDict

from laygen.common.labels import DatasetName, id2label_for_dataset


LABEL_NORMALIZATION: Final[str] = "unicode-nfkc-strip-collapse-whitespace-casefold-v1"


class LabelTranslationMetadata(TypedDict):
    """JSON-compatible persisted semantic label-map evidence."""

    normalization: str
    public_id2label: dict[int | str, str]
    sequence_id2label: dict[int | str, str]
    public_to_sequence: dict[int | str, int]
    sequence_to_public: dict[int | str, int]
    sha256: str


RICO25_SEQUENCE_LABELS: Final[tuple[str, ...]] = (
    "Text",
    "Image",
    "Icon",
    "List Item",
    "Text Button",
    "Toolbar",
    "Web View",
    "Input",
    "Card",
    "Advertisement",
    "Background Image",
    "Drawer",
    "Radio Button",
    "Checkbox",
    "Multi-Tab",
    "Pager Indicator",
    "Modal",
    "On/Off Switch",
    "Slider",
    "Map View",
    "Button Bar",
    "Video",
    "Bottom Navigation",
    "Number Stepper",
    "Date Picker",
)


def normalize_label_name(name: str) -> str:
    """Return the exact semantic join key used by label translations."""
    normalized = unicodedata.normalize("NFKC", name)
    return " ".join(normalized.strip().split()).casefold()


@dataclass(frozen=True, slots=True)
class LabelTranslation:
    """Persist source label maps and their name-joined translation."""

    public_id2label: Mapping[int, str]
    sequence_id2label: Mapping[int, str]
    public_to_sequence: Mapping[int, int]
    sequence_to_public: Mapping[int, int]
    canonical_json: str
    sha256: str

    def metadata(self) -> LabelTranslationMetadata:
        """Return JSON-safe maps for configs, checkpoints, and evidence."""
        public_id2label: dict[int | str, str] = {
            key: value for key, value in self.public_id2label.items()
        }
        sequence_id2label: dict[int | str, str] = {
            key: value for key, value in self.sequence_id2label.items()
        }
        public_to_sequence: dict[int | str, int] = {
            key: value for key, value in self.public_to_sequence.items()
        }
        sequence_to_public: dict[int | str, int] = {
            key: value for key, value in self.sequence_to_public.items()
        }
        return {
            "normalization": LABEL_NORMALIZATION,
            "public_id2label": public_id2label,
            "sequence_id2label": sequence_id2label,
            "public_to_sequence": public_to_sequence,
            "sequence_to_public": sequence_to_public,
            "sha256": self.sha256,
        }


def build_label_translation(
    public_id2label: Mapping[int, str],
    sequence_id2label: Mapping[int, str],
) -> LabelTranslation:
    """Join two complete label maps by normalized semantic name.

    Raises:
        ValueError: If ids, names, or inverse coverage are incomplete or ambiguous.
    """
    public = {int(key): str(value) for key, value in public_id2label.items()}
    sequence = {int(key): str(value) for key, value in sequence_id2label.items()}
    if set(public) != set(range(len(public))):
        raise ValueError("public ids must be contiguous from zero")
    if set(sequence) != set(range(1, len(sequence) + 1)):
        raise ValueError("sequence ids must be contiguous from one")
    if len(public) != len(sequence):
        raise ValueError("public and sequence label maps must have equal size")

    public_names = _normalized_name_to_id(public, map_name="public")
    sequence_names = _normalized_name_to_id(sequence, map_name="sequence")
    if set(public_names) != set(sequence_names):
        raise ValueError("normalized label-name sets differ")

    public_to_sequence = {
        public_id: sequence_names[normalize_label_name(label)]
        for public_id, label in public.items()
    }
    sequence_to_public = {
        sequence_id: public_id for public_id, sequence_id in public_to_sequence.items()
    }
    if len(sequence_to_public) != len(public):
        raise ValueError("label translation is not bijective")

    payload = {
        "normalization": LABEL_NORMALIZATION,
        "public_id2label": public,
        "sequence_id2label": sequence,
        "public_to_sequence": public_to_sequence,
        "sequence_to_public": sequence_to_public,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return LabelTranslation(
        public_id2label=MappingProxyType(public),
        sequence_id2label=MappingProxyType(sequence),
        public_to_sequence=MappingProxyType(public_to_sequence),
        sequence_to_public=MappingProxyType(sequence_to_public),
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )


def _normalized_name_to_id(
    id2label: Mapping[int, str], *, map_name: str
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for label_id, label in id2label.items():
        key = normalize_label_name(label)
        if key in normalized:
            raise ValueError(f"{map_name} label normalization collision: {label!r}")
        normalized[key] = label_id
    return normalized


RICO25_LABEL_TRANSLATION: Final[LabelTranslation] = build_label_translation(
    id2label_for_dataset(DatasetName.rico25),
    dict(enumerate(RICO25_SEQUENCE_LABELS, start=1)),
)
PUBLAYNET_LABEL_TRANSLATION: Final[LabelTranslation] = build_label_translation(
    id2label_for_dataset(DatasetName.publaynet),
    {
        index: label
        for index, label in enumerate(
            id2label_for_dataset(DatasetName.publaynet).values(), start=1
        )
    },
)


def label_translation_for_dataset(dataset: DatasetName) -> LabelTranslation:
    """Return the canonical public-to-sequence label translation."""
    if dataset is DatasetName.rico25:
        return RICO25_LABEL_TRANSLATION
    if dataset is DatasetName.publaynet:
        return PUBLAYNET_LABEL_TRANSLATION
    raise ValueError(f"Unsupported LayoutFormer++ dataset: {dataset}")


def validate_label_translation_metadata(
    metadata: LabelTranslationMetadata,
    expected: LabelTranslation,
) -> LabelTranslationMetadata:
    """Normalize persisted label metadata and reject any changed field."""
    try:
        public_raw = metadata["public_id2label"]
        sequence_raw = metadata["sequence_id2label"]
        public_to_sequence_raw = metadata["public_to_sequence"]
        sequence_to_public_raw = metadata["sequence_to_public"]
        normalization = metadata["normalization"]
        sha256 = metadata["sha256"]
    except KeyError as exc:
        raise ValueError("label translation metadata is incomplete") from exc
    public = {int(key): str(value) for key, value in public_raw.items()}
    sequence = {int(key): str(value) for key, value in sequence_raw.items()}
    rebuilt = build_label_translation(public, sequence)
    normalized = rebuilt.metadata()
    normalized["public_to_sequence"] = {
        int(key): int(value) for key, value in public_to_sequence_raw.items()
    }
    normalized["sequence_to_public"] = {
        int(key): int(value) for key, value in sequence_to_public_raw.items()
    }
    normalized["normalization"] = str(normalization)
    normalized["sha256"] = str(sha256)
    if normalized != expected.metadata() or rebuilt.sha256 != expected.sha256:
        raise ValueError("label translation metadata does not match the canonical map")
    return expected.metadata()


__all__ = [
    "LABEL_NORMALIZATION",
    "LabelTranslationMetadata",
    "LabelTranslation",
    "PUBLAYNET_LABEL_TRANSLATION",
    "RICO25_LABEL_TRANSLATION",
    "build_label_translation",
    "label_translation_for_dataset",
    "normalize_label_name",
    "validate_label_translation_metadata",
]
