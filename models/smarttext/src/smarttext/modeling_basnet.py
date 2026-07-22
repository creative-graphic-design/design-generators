"""BASNet saliency model ported from the SmartText vendor architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_smarttext import SmartTextConfig


@dataclass
class SmartTextSaliencyOutput(ModelOutput):
    """Output of ``SmartTextBASNet.forward``."""

    saliency: torch.Tensor
    side_outputs: tuple[torch.Tensor, ...] | None = None


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one vendor BASNet residual block."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class _RefUnet(nn.Module):
    def __init__(self, in_ch: int, inc_ch: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)
        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)
        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)
        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)
        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)
        self.upscore2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the vendor RefUnet refinement module."""
        hx = self.conv0(x)
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)
        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)
        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)
        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)
        hx5 = self.relu5(self.bn5(self.conv5(hx)))
        hx = self.upscore2(hx5)
        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)
        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)
        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)
        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))
        return x + self.conv_d0(d1)


def normalize_saliency(pred: torch.Tensor) -> torch.Tensor:
    """Normalize saliency maps exactly like ``basnet_test.py::normPRED``."""
    min_value = pred.amin(dim=(-2, -1), keepdim=True)
    max_value = pred.amax(dim=(-2, -1), keepdim=True)
    return (pred - min_value) / (max_value - min_value)


class SmartTextBASNet(PreTrainedModel):
    """Vendor-compatible BASNet saliency predictor.

    Module names match ``vendor/smarttext/BASNet/model/BASNet.py::BASNet(3, 1)``
    so ``gdi-basnet.pth`` can be loaded directly after any documented key
    normalization needed by the released file.
    """

    config_class = SmartTextConfig
    main_input_name = "pixel_values"
    _tied_weights_keys: list[str] = []

    def __init__(self, config: SmartTextConfig) -> None:
        """Initialize the vendor BASNet architecture without downloading weights."""
        super().__init__(config)
        self.all_tied_weights_keys: dict[str, str] = {}
        resnet = models.resnet34(weights=None)
        self.inconv = nn.Conv2d(3, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.resb5_1 = _BasicBlock(512, 512)
        self.resb5_2 = _BasicBlock(512, 512)
        self.resb5_3 = _BasicBlock(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.resb6_1 = _BasicBlock(512, 512)
        self.resb6_2 = _BasicBlock(512, 512)
        self.resb6_3 = _BasicBlock(512, 512)
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)
        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)
        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)
        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)
        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)
        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)
        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)
        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)
        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)
        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)
        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)
        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)
        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)
        self.upscore6 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=False
        )
        self.upscore5 = nn.Upsample(
            scale_factor=16, mode="bilinear", align_corners=False
        )
        self.upscore4 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=False
        )
        self.upscore3 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.upscore2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)
        self.refunet = _RefUnet(1, 64)

    def _forward_vendor(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hx = self.inrelu(self.inbn(self.inconv(x)))
        h1 = self.encoder1(hx)
        h2 = self.encoder2(h1)
        h3 = self.encoder3(h2)
        h4 = self.encoder4(h3)
        hx = self.pool4(h4)
        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)
        hx = self.pool5(h5)
        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))
        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn5d_2(self.conv6d_2(hx)))
        hx = self.upscore2(hd6)
        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))
        hx = self.upscore2(hd5)
        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))
        hx = self.upscore2(hd4)
        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))
        hx = self.upscore2(hd3)
        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))
        hx = self.upscore2(hd2)
        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))
        db = self.upscore6(self.outconvb(hbg))
        d6 = self.upscore6(self.outconv6(hd6))
        d5 = self.upscore5(self.outconv5(hd5))
        d4 = self.upscore4(self.outconv4(hd4))
        d3 = self.upscore3(self.outconv3(hd3))
        d2 = self.upscore2(self.outconv2(hd2))
        d1 = self.outconv1(hd1)
        dout = self.refunet(d1)
        return tuple(torch.sigmoid(row) for row in (dout, d1, d2, d3, d4, d5, d6, db))

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool | None = None,
    ) -> SmartTextSaliencyOutput | tuple[torch.Tensor, ...]:
        """Predict saliency with the vendor BASNet forward path."""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self._forward_vendor(pixel_values)
        saliency = normalize_saliency(outputs[0][:, 0, :, :])
        if not return_dict:
            return (saliency, *outputs)
        return SmartTextSaliencyOutput(saliency=saliency, side_outputs=outputs)
