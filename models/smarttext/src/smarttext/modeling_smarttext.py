"""SmartText scorer model ported from the original SMT architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_smarttext import SmartTextConfig


@dataclass
class SmartTextScorerOutput(ModelOutput):
    """Output of ``SmartTextScorer.forward``."""

    scores: Float[torch.Tensor, "candidates"]


def _conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class _InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, benchmodel: Literal[1, 2]
    ) -> None:
        super().__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        oup_inc = oup // 2
        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        elif self.benchmodel == 2:
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            self.banch2 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            assert_never(self.benchmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one vendor ShuffleNetV2 residual block."""
        if self.benchmodel == 1:
            x1 = x[:, : (x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2) :, :, :]
            out = torch.cat((x1, self.banch2(x2)), 1)
        elif self.benchmodel == 2:
            out = torch.cat((self.banch1(x), self.banch2(x)), 1)
        else:
            assert_never(self.benchmodel)
        return _channel_shuffle(out, 2)


class _ShuffleNetV2(nn.Module):
    def __init__(self, width_mult: float = 1.0) -> None:
        super().__init__()
        stage_repeats = [4, 8, 4]
        if width_mult != 1.0:
            raise ValueError("SmartText released SMT checkpoint uses width_mult=1.0")
        stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        input_channel = stage_out_channels[1]
        self.conv1 = _conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        features: list[nn.Module] = []
        for idxstage, numrepeat in enumerate(stage_repeats):
            output_channel = stage_out_channels[idxstage + 2]
            for index in range(numrepeat):
                if index == 0:
                    features.append(
                        _InvertedResidual(input_channel, output_channel, 2, 2)
                    )
                else:
                    features.append(
                        _InvertedResidual(input_channel, output_channel, 1, 1)
                    )
                input_channel = output_channel
        self.features = nn.Sequential(*features)


class _ShuffleNetV2Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = _ShuffleNetV2(width_mult=1.0)
        self.feature3 = nn.Sequential(model.conv1, model.maxpool, model.features[:4])
        self.feature4 = nn.Sequential(model.features[4:12])
        self.feature5 = nn.Sequential(model.features[12:])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return vendor feature stages f3, f4, and f5."""
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


def _fc_layers(reddim: int = 32, alignsize: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(reddim, 768, kernel_size=alignsize, padding=0),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ),
        nn.Dropout(p=0.5),
        nn.Conv2d(128, 1, kernel_size=1),
    )


class _AlignBase(nn.Module):
    def __init__(
        self, aligned_height: int, aligned_width: int, spatial_scale: float
    ) -> None:
        super().__init__()
        self.aligned_height = int(aligned_height)
        self.aligned_width = int(aligned_width)
        self.spatial_scale = float(spatial_scale)

    def _sample(
        self, features: torch.Tensor, rois: torch.Tensor, mode: Literal["roi", "rod"]
    ) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        del batch_size
        output = features.new_zeros(
            (rois.shape[0], channels, self.aligned_height, self.aligned_width)
        )
        for roi_index, roi in enumerate(rois):
            batch_index = int(roi[0].item())
            start_w = roi[1] * self.spatial_scale
            start_h = roi[2] * self.spatial_scale
            end_w = roi[3] * self.spatial_scale
            end_h = roi[4] * self.spatial_scale
            if mode == "roi":
                roi_width = torch.clamp(end_w - start_w + 1.0, min=0.0)
                roi_height = torch.clamp(end_h - start_h + 1.0, min=0.0)
                bin_h = roi_height / float(self.aligned_height - 1)
                bin_w = roi_width / float(self.aligned_width - 1)
            elif mode == "rod":
                bin_h = features.new_tensor(
                    (height - 1.001) / float(self.aligned_height - 1)
                )
                bin_w = features.new_tensor(
                    (width - 1.001) / float(self.aligned_width - 1)
                )
            else:
                assert_never(mode)
            for ph in range(self.aligned_height):
                h = float((ph * bin_h + (start_h if mode == "roi" else 0.0)).item())
                hstart = min(
                    int(torch.floor(features.new_tensor(h)).item()), height - 2
                )
                for pw in range(self.aligned_width):
                    w = float((pw * bin_w + (start_w if mode == "roi" else 0.0)).item())
                    wstart = min(
                        int(torch.floor(features.new_tensor(w)).item()), width - 2
                    )
                    if mode == "roi" and (h < 0 or h >= height or w < 0 or w >= width):
                        continue
                    if (
                        mode == "rod"
                        and start_h <= h <= end_h
                        and start_w <= w <= end_w
                    ):
                        continue
                    h_ratio = h - float(hstart)
                    w_ratio = w - float(wstart)
                    output[roi_index, :, ph, pw] = (
                        features[batch_index, :, hstart, wstart]
                        * (1.0 - h_ratio)
                        * (1.0 - w_ratio)
                        + features[batch_index, :, hstart, wstart + 1]
                        * (1.0 - h_ratio)
                        * w_ratio
                        + features[batch_index, :, hstart + 1, wstart]
                        * h_ratio
                        * (1.0 - w_ratio)
                        + features[batch_index, :, hstart + 1, wstart + 1]
                        * h_ratio
                        * w_ratio
                    )
        return output


class SmartTextRoIAlignAvg(_AlignBase):
    """PyTorch port of the vendor ``RoIAlignAvg`` forward kernel."""

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """Align RoI features and average adjacent samples."""
        sampled = self._sample(features, rois, "roi")
        return F.avg_pool2d(sampled, kernel_size=2, stride=1)


class SmartTextRoDAlignAvg(_AlignBase):
    """PyTorch port of the vendor ``RoDAlignAvg`` forward kernel."""

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """Align outside-region features and average adjacent samples."""
        sampled = self._sample(features, rois, "rod")
        return F.avg_pool2d(sampled, kernel_size=2, stride=1)


class SmartTextScorer(PreTrainedModel):
    """Vendor-compatible SMT candidate scorer.

    The module names match ``vendor/smarttext/smtModel.py`` for
    ``build_smt_model(scale="multi", alignsize=9, reddim=8,
    model="shufflenetv2", downsample=4)`` so ``SMT.pth`` loads directly.

    Args:
        config: SmartText configuration.

    Examples:
        >>> config = SmartTextConfig()
        >>> model = SmartTextScorer(config)
        >>> "Feat_ext.feature3.0.0.weight" in model.state_dict()
        True
    """

    config_class = SmartTextConfig
    main_input_name = "pixel_values"
    _tied_weights_keys: list[str] = []

    def __init__(self, config: SmartTextConfig) -> None:
        """Initialize vendor SMT scorer modules."""
        super().__init__(config)
        self.all_tied_weights_keys: dict[str, str] = {}
        if config.scorer_scale != "multi" or config.scorer_backbone != "shufflenetv2":
            raise ValueError(
                "Only the released multi-scale ShuffleNetV2 SMT scorer is supported"
            )
        self.Feat_ext = _ShuffleNetV2Base()
        self.DimRed = nn.Conv2d(812, config.reduction_dim, kernel_size=1, padding=0)
        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0 / 2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        spatial_scale = 1.0 / 2**config.downsample
        self.RoIAlign = SmartTextRoIAlignAvg(
            config.align_size + 1, config.align_size + 1, spatial_scale
        )
        self.RoDAlign = SmartTextRoDAlignAvg(
            config.align_size + 1, config.align_size + 1, spatial_scale
        )
        self.FC_layers = _fc_layers(config.reduction_dim * 2, config.align_size)

    def forward(
        self,
        pixel_values: Float[torch.Tensor, "batch channels height width"],
        boxes: Float[torch.Tensor, "candidates 5"],
        return_dict: bool | None = None,
    ) -> SmartTextScorerOutput | tuple[Float[torch.Tensor, "candidates"]]:
        """Score candidate text regions.

        Args:
            pixel_values: RGB scorer tensor shaped ``(B, 3, H, W)``.
            boxes: RoI rows shaped ``(N, 5)`` with batch index in column zero.
            return_dict: Whether to return a ``ModelOutput``.

        Returns:
            Candidate scores as a ``SmartTextScorerOutput`` or tuple.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        f3, f4, f5 = self.Feat_ext(pixel_values)
        cat_feat = torch.cat((self.downsample2(f3), f4, 0.5 * self.upsample2(f5)), 1)
        red_feat = self.DimRed(cat_feat)
        roi_feat = self.RoIAlign(red_feat, boxes)
        rod_feat = self.RoDAlign(red_feat, boxes)
        prediction = self.FC_layers(torch.cat((roi_feat, rod_feat), 1)).flatten()
        if not return_dict:
            return (prediction,)
        return SmartTextScorerOutput(scores=prediction)
