"""Prompt serialization ported from the LayoutPrompter notebooks."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from layoutprompter.data import CANVAS_SIZE, LAYOUT_DOMAIN, id2label

PREAMBLE = (
    "Please generate a layout based on the given information. "
    "You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.\n"
    "Task Description: {}\n"
    "Layout Domain: {} layout\n"
    "Canvas Size: canvas width is {}px, canvas height is {}px"
)

HTML_PREFIX = '<html>\n<body>\n<div class="canvas" style="left: 0px; top: 0px; width: {}px; height: {}px"></div>\n'
HTML_SUFFIX = "</body>\n</html>"
HTML_TEMPLATE = (
    '<div class="{}" style="left: {}px; top: {}px; width: {}px; height: {}px"></div>\n'
)
HTML_TEMPLATE_WITH_INDEX = '<div class="{}" style="index: {}; left: {}px; top: {}px; width: {}px; height: {}px"></div>\n'


class Serializer:
    """Base serializer for seq/html prompt examples."""

    task_type = ""
    constraint_type: list[str] = []

    def __init__(
        self,
        *,
        input_format: str,
        output_format: str,
        index2label: dict[int, str],
        canvas_width: int,
        canvas_height: int,
        add_index_token: bool = True,
        add_sep_token: bool = True,
        sep_token: str = "|",
        add_unk_token: bool = False,
        unk_token: str = "<unk>",
    ) -> None:
        """Create a serializer with vendor token options."""
        self.input_format = input_format
        self.output_format = output_format
        self.index2label = index2label
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.add_index_token = add_index_token
        self.add_sep_token = add_sep_token
        self.sep_token = sep_token
        self.add_unk_token = add_unk_token
        self.unk_token = unk_token

    def build_input(self, data: dict[str, Any]) -> str:
        """Serialize test constraints."""
        if self.input_format == "seq":
            return self._build_seq_input(data)
        if self.input_format == "html":
            return self._build_html_input(data)
        raise ValueError(f"Unsupported input format: {self.input_format}")

    def build_output(
        self,
        data: dict[str, Any],
        label_key: str = "labels",
        bbox_key: str = "discrete_gold_bboxes",
    ) -> str:
        """Serialize an exemplar output layout."""
        if self.output_format == "seq":
            return self._build_seq_output(data, label_key, bbox_key)
        if self.output_format == "html":
            return self._build_html_output(data, label_key, bbox_key)
        raise ValueError(f"Unsupported output format: {self.output_format}")

    def _build_seq_input(self, data: dict[str, Any]) -> str:
        raise NotImplementedError

    def _build_html_input(self, data: dict[str, Any]) -> str:
        raise NotImplementedError

    def _build_seq_output(
        self, data: dict[str, Any], label_key: str, bbox_key: str
    ) -> str:
        labels = data[label_key]
        bboxes = data[bbox_key]
        tokens: list[str] = []
        for index in range(len(labels)):
            tokens.append(self.index2label[int(labels[index])])
            if self.add_index_token:
                tokens.append(str(index))
            tokens.extend(str(int(value)) for value in bboxes[index].tolist())
            if self.add_sep_token and index < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_output(
        self, data: dict[str, Any], label_key: str, bbox_key: str
    ) -> str:
        labels = data[label_key]
        bboxes = data[bbox_key]
        template = HTML_TEMPLATE_WITH_INDEX if self.add_index_token else HTML_TEMPLATE
        html = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        for index in range(len(labels)):
            element: list[str] = [self.index2label[int(labels[index])]]
            if self.add_index_token:
                element.append(str(index))
            element.extend(str(int(value)) for value in bboxes[index].tolist())
            html.append(template.format(*element))
        html.append(HTML_SUFFIX)
        return "".join(html)


class GenTypeSerializer(Serializer):
    """Serializer for element-type conditioned generation."""

    task_type = "generation conditioned on given element types"
    constraint_type = ["Element Type Constraint: "]

    @override
    def _build_seq_input(self, data: dict[str, Any]) -> str:
        tokens: list[str] = []
        labels = data["labels"]
        for index in range(len(labels)):
            tokens.append(self.index2label[int(labels[index])])
            if self.add_index_token:
                tokens.append(str(index))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and index < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    @override
    def _build_html_input(self, data: dict[str, Any]) -> str:
        html = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        for index in range(len(data["labels"])):
            label = self.index2label[int(data["labels"][index])]
            if self.add_unk_token:
                bbox = [self.unk_token] * 4
                element = (
                    [label, str(index), *bbox]
                    if self.add_index_token
                    else [label, *bbox]
                )
                html.append(
                    (
                        HTML_TEMPLATE_WITH_INDEX
                        if self.add_index_token
                        else HTML_TEMPLATE
                    ).format(*element)
                )
            elif self.add_index_token:
                html.append(f'<div class="{label}" style="index: {index}"></div>\n')
            else:
                html.append(f'<div class="{label}"></div>\n')
        html.append(HTML_SUFFIX)
        return "".join(html)

    @override
    def build_input(self, data: dict[str, Any]) -> str:
        """Serialize type constraints with the vendor prefix."""
        return self.constraint_type[0] + super().build_input(data)


class GenTypeSizeSerializer(GenTypeSerializer):
    """Serializer for element-type and size conditioned generation."""

    task_type = "generation conditioned on given element types and sizes"
    constraint_type = ["Element Type and Size Constraint: "]

    @override
    def _build_seq_input(self, data: dict[str, Any]) -> str:
        tokens: list[str] = []
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        for index in range(len(labels)):
            tokens.append(self.index2label[int(labels[index])])
            if self.add_index_token:
                tokens.append(str(index))
            if self.add_unk_token:
                tokens += [self.unk_token] * 2
            tokens.extend(str(int(value)) for value in bboxes[index].tolist()[2:])
            if self.add_sep_token and index < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    @override
    def _build_html_input(self, data: dict[str, Any]) -> str:
        html = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        for index in range(len(labels)):
            label = self.index2label[int(labels[index])]
            width, height = [int(value) for value in bboxes[index].tolist()[2:]]
            if self.add_index_token:
                html.append(
                    f'<div class="{label}" style="index: {index}; width: {width}px; height: {height}px"></div>\n'
                )
            else:
                html.append(
                    f'<div class="{label}" style="width: {width}px; height: {height}px"></div>\n'
                )
        html.append(HTML_SUFFIX)
        return "".join(html)


class GenRelationSerializer(GenTypeSerializer):
    """Serializer for relation-conditioned generation."""

    task_type = (
        "generation conditioned on given element relationships\n"
        "'A left B' means that the center coordinate of A is to the left of the center coordinate of B. "
        "'A right B' means that the center coordinate of A is to the right of the center coordinate of B. "
        "'A top B' means that the center coordinate of A is above the center coordinate of B. "
        "'A bottom B' means that the center coordinate of A is below the center coordinate of B. "
        "'A center B' means that the center coordinate of A and the center coordinate of B are very close. "
        "'A smaller B' means that the area of A is smaller than the ares of B. "
        "'A larger B' means that the area of A is larger than the ares of B. "
        "'A equal B' means that the area of A and the ares of B are very close. "
        "Here, center coordinate = (left + width / 2, top + height / 2), area = width * height"
    )
    constraint_type = ["Element Type Constraint: ", "Element Relationship Constraint: "]
    relation_types = (
        "smaller",
        "equal",
        "larger",
        "top",
        "center",
        "bottom",
        "left",
        "right",
    )

    @override
    def build_input(self, data: dict[str, Any]) -> str:
        """Serialize type and relation constraints."""
        type_constraints = self.constraint_type[0] + super(
            GenTypeSerializer, self
        ).build_input(data)
        relations = data.get("relations", [])
        if len(relations) == 0:
            return type_constraints
        relation_tokens: list[str] = []
        for index, relation in enumerate(relations):
            label_j, index_j, label_i, index_i, relation_type = [
                int(value) for value in relation
            ]
            relation_tokens.append(
                "canvas" if label_i < 0 else f"{self.index2label[label_i]} {index_i}"
            )
            relation_tokens.append(self.relation_types[relation_type])
            relation_tokens.append(
                "canvas" if label_j < 0 else f"{self.index2label[label_j]} {index_j}"
            )
            if self.add_sep_token and index < len(relations) - 1:
                relation_tokens.append(self.sep_token)
        return (
            type_constraints
            + "\n"
            + self.constraint_type[1]
            + " ".join(relation_tokens)
        )


class CompletionSerializer(Serializer):
    """Serializer for layout completion."""

    task_type = "layout completion"
    constraint_type = ["Partial Layout: "]

    @override
    def _build_seq_input(self, data: dict[str, Any]) -> str:
        return self._build_seq_output(
            {"labels": data["labels"][:1], "bboxes": data["discrete_bboxes"][:1]},
            "labels",
            "bboxes",
        )

    @override
    def _build_html_input(self, data: dict[str, Any]) -> str:
        return self._build_html_output(
            {"labels": data["labels"][:1], "bboxes": data["discrete_bboxes"][:1]},
            "labels",
            "bboxes",
        )

    @override
    def build_input(self, data: dict[str, Any]) -> str:
        """Serialize partial layout constraints with the vendor prefix."""
        return self.constraint_type[0] + super().build_input(data)


class RefinementSerializer(Serializer):
    """Serializer for noisy-layout refinement."""

    task_type = "layout refinement"
    constraint_type = ["Noise Layout: "]

    @override
    def _build_seq_input(self, data: dict[str, Any]) -> str:
        return self._build_seq_output(data, "labels", "discrete_bboxes")

    @override
    def _build_html_input(self, data: dict[str, Any]) -> str:
        return self._build_html_output(data, "labels", "discrete_bboxes")

    @override
    def build_input(self, data: dict[str, Any]) -> str:
        """Serialize noisy layout constraints with the vendor prefix."""
        return self.constraint_type[0] + super().build_input(data)


class TextToLayoutSerializer(Serializer):
    """Serializer for text-to-layout prompts."""

    task_type = (
        "text-to-layout\n"
        "There are ten optional element types, including: image, icon, logo, background, title, description, text, link, input, button. "
        "Please do not exceed the boundaries of the canvas. "
        "Besides, do not generate elements at the edge of the canvas, that is, reduce top: 0px and left: 0px predictions as much as possible."
    )
    constraint_type = ["Text: "]

    @override
    def _build_seq_input(self, data: dict[str, Any]) -> str:
        return str(data["text"])

    @override
    def _build_html_input(self, data: dict[str, Any]) -> str:
        return self._build_seq_input(data)

    @override
    def build_input(self, data: dict[str, Any]) -> str:
        """Serialize text input with the vendor prefix."""
        return self.constraint_type[0] + super().build_input(data)


class ContentAwareSerializer(GenTypeSerializer):
    """Serializer for content-aware poster layout generation."""

    task_type = (
        "content-aware layout generation\n"
        "Please place the following elements to avoid salient content, and underlay must be the background of text or logo."
    )
    constraint_type = ["Content Constraint: ", "Element Type Constraint: "]

    @override
    def _build_seq_input(self, data: dict[str, Any]) -> str:
        content_tokens = []
        for index, bbox in enumerate(data["discrete_content_bboxes"]):
            left, top, width, height = [int(value) for value in bbox.tolist()]
            content_tokens.append(
                f"left {left}px, top {top}px, width {width}px, height {height}px"
            )
            if self.add_sep_token and index < len(data["discrete_content_bboxes"]) - 1:
                content_tokens.append(self.sep_token)
        return (
            self.constraint_type[0]
            + " ".join(content_tokens)
            + "\n"
            + self.constraint_type[1]
            + GenTypeSerializer._build_seq_input(self, data)
        )

    @override
    def build_input(self, data: dict[str, Any]) -> str:
        """Serialize content masks and element type constraints."""
        return Serializer.build_input(self, data)


SERIALIZER_MAP: dict[str, type[Serializer]] = {
    "gent": GenTypeSerializer,
    "gents": GenTypeSizeSerializer,
    "genr": GenRelationSerializer,
    "completion": CompletionSerializer,
    "refinement": RefinementSerializer,
    "content": ContentAwareSerializer,
    "text": TextToLayoutSerializer,
}


def create_serializer(
    dataset: str,
    task: str,
    input_format: str,
    output_format: str,
    *,
    add_index_token: bool = True,
    add_sep_token: bool = True,
    add_unk_token: bool = False,
) -> Serializer:
    """Create a task serializer."""
    width, height = CANVAS_SIZE[dataset]
    return SERIALIZER_MAP[task](
        input_format=input_format,
        output_format=output_format,
        index2label=id2label(dataset),
        canvas_width=width,
        canvas_height=height,
        add_index_token=add_index_token,
        add_sep_token=add_sep_token,
        add_unk_token=add_unk_token,
    )


def build_prompt(
    serializer: Serializer,
    exemplars: list[dict[str, Any]],
    test_data: dict[str, Any],
    dataset: str,
    *,
    max_length: int = 8000,
    separator_in_samples: str = "\n",
    separator_between_samples: str = "\n\n",
) -> str:
    """Build the final few-shot LayoutPrompter prompt."""
    prompt = [
        PREAMBLE.format(
            serializer.task_type, LAYOUT_DOMAIN[dataset], *CANVAS_SIZE[dataset]
        )
    ]
    for exemplar in exemplars:
        sample = (
            serializer.build_input(exemplar)
            + separator_in_samples
            + serializer.build_output(exemplar)
        )
        if len(separator_between_samples.join(prompt) + sample) <= max_length:
            prompt.append(sample)
        else:
            break
    prompt.append(serializer.build_input(test_data) + separator_in_samples)
    return separator_between_samples.join(prompt)
