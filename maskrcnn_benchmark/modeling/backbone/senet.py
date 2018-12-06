# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Numer of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# SENet models
# -----------------------------------------------------------------------------
# SEResNext50_32x4d

SEResNext50_32x4d_FPNStagesTo5 = (
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)


class SENet(nn.Module):
    def __init__(self, cfg):
        super(SENet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        # num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        num_groups = 32
        #width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        width_per_group = 4
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor

            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

## se_resnext
class BottleneckWithFixedBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=32,
        stride_in_1x1=True,
        stride=1,
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                FrozenBatchNorm2d(out_channels),
            )

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        width = bottleneck_channels
        
        self.conv1 = Conv2d(
            in_channels,
            width,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = FrozenBatchNorm2d(width)
        # TODO: specify init for the above

        self.conv2 = Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        self.bn2 = FrozenBatchNorm2d(width)

        self.conv3 = Conv2d(
            width, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = FrozenBatchNorm2d(out_channels)
        self.se_module = SEModule(out_channels, reduction=16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        #out += residual
        out = F.relu_(out)

        return out


class StemWithFixedBatchNorm(nn.Module):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


_TRANSFORMATION_MODULES = {"BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm}

_STEM_MODULES = {"StemWithFixedBatchNorm": StemWithFixedBatchNorm}

_STAGE_SPECS = {
    "SE-R-50-32x4d-FPN": SEResNext50_32x4d_FPNStagesTo5,
}


def register_transformation_module(module_name, module):
    _register_generic(_TRANSFORMATION_MODULES, module_name, module)


def register_stem_module(module_name, module):
    _register_generic(_STEM_MODULES, module_name, module)


def register_stage_spec(stage_spec_name, stage_spec):
    _register_generic(_STAGE_SPECS, stage_spec_name, stage_spec)


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module
