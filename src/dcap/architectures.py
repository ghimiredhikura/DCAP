"""Architectures used in the manuscript."""

import torch
from torch import nn
from torch.nn import functional as F


def normalization_layer(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        group_count = min(8, channels)
        while channels % group_count:
            group_count -= 1
        return nn.GroupNorm(group_count, channels)
    raise ValueError(f"Unsupported normalization: {kind}")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        normalization: str = "batch",
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            normalization_layer(normalization, out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        normalization: str = "batch",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = normalization_layer(normalization, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = normalization_layer(normalization, out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                normalization_layer(normalization, out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.activation(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.activation(outputs + residual)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        normalization: str = "batch",
    ):
        super().__init__()
        self.reduce = ConvNormAct(
            in_channels, out_channels, kernel_size=1, normalization=normalization
        )
        self.fuse = ResidualBlock(
            out_channels + skip_channels, out_channels, normalization=normalization
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        outputs = F.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        outputs = self.reduce(outputs)
        return self.fuse(torch.cat((outputs, skip), dim=1))


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 15,
        base_channels: int = 32,
        normalization: str = "batch",
    ):
        super().__init__()
        channels = [base_channels * factor for factor in (1, 2, 4, 8, 16)]
        self.encoder1 = ResidualBlock(
            in_channels, channels[0], normalization=normalization
        )
        self.encoder2 = ResidualBlock(
            channels[0], channels[1], stride=2, normalization=normalization
        )
        self.encoder3 = ResidualBlock(
            channels[1], channels[2], stride=2, normalization=normalization
        )
        self.encoder4 = ResidualBlock(
            channels[2], channels[3], stride=2, normalization=normalization
        )
        self.bridge = ResidualBlock(
            channels[3], channels[4], stride=2, normalization=normalization
        )
        self.decoder4 = UpBlock(
            channels[4], channels[3], channels[3], normalization=normalization
        )
        self.decoder3 = UpBlock(
            channels[3], channels[2], channels[2], normalization=normalization
        )
        self.decoder2 = UpBlock(
            channels[2], channels[1], channels[1], normalization=normalization
        )
        self.decoder1 = UpBlock(
            channels[1], channels[0], channels[0], normalization=normalization
        )
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        bridge = self.bridge(encoder4)
        outputs = self.decoder4(bridge, encoder4)
        outputs = self.decoder3(outputs, encoder3)
        outputs = self.decoder2(outputs, encoder2)
        outputs = self.decoder1(outputs, encoder1)
        return self.head(outputs)


RGB_INDICES = {
    "post_only": (2, 1, 0),
    "post_only_no_terrain": (2, 1, 0),
    "raw_bitemporal": (8, 7, 6),
    "raw_bitemporal_no_terrain": (8, 7, 6),
    "reflectance_change": (8, 7, 6),
    "reflectance_change_no_terrain": (8, 7, 6),
}


def adapt_rgb_stem(
    original: nn.Conv2d,
    *,
    in_channels: int,
    input_mode: str,
    pretrained: bool,
    extra_channel_initialization: str,
) -> nn.Conv2d:
    if input_mode not in RGB_INDICES:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    if extra_channel_initialization not in ("zero", "rgb_mean"):
        raise ValueError("extra_channel_initialization must be zero or rgb_mean")
    adapted = nn.Conv2d(
        in_channels,
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=False,
    )
    with torch.no_grad():
        if pretrained:
            if extra_channel_initialization == "zero":
                adapted.weight.zero_()
            else:
                mean_kernel = original.weight.mean(dim=1, keepdim=True)
                adapted.weight.copy_(mean_kernel.repeat(1, in_channels, 1, 1))
            red, green, blue = RGB_INDICES[input_mode]
            adapted.weight[:, red] = original.weight[:, 0]
            adapted.weight[:, green] = original.weight[:, 1]
            adapted.weight[:, blue] = original.weight[:, 2]
        else:
            nn.init.kaiming_normal_(adapted.weight, mode="fan_out", nonlinearity="relu")
    return adapted


class ASPPBranch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=1 if dilation == 1 else dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.branches = nn.ModuleList(
            (
                ASPPBranch(in_channels, out_channels, dilation)
                for dilation in (1, 6, 12, 18)
            )
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [branch(inputs) for branch in self.branches]
        pooled = self.pool_projection(self.pool(inputs))
        pooled = F.interpolate(
            pooled, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.project(torch.cat((*outputs, pooled), dim=1))


class ResNet34DeepLabV3Plus(nn.Module):
    """DeepLabV3+ control with an explicitly adapted ResNet-34 encoder."""

    def __init__(
        self,
        in_channels: int,
        input_mode: str,
        pretrained: bool = True,
        extra_channel_initialization: str = "zero",
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet34(weights=weights)
        encoder.conv1 = adapt_rgb_stem(
            encoder.conv1,
            in_channels=in_channels,
            input_mode=input_mode,
            pretrained=pretrained,
            extra_channel_initialization=extra_channel_initialization,
        )
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.aspp = ASPP(512, 128)
        self.low_projection = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(176, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, 1, 1)
        self.input_mode = input_mode
        self.extra_channel_initialization = extra_channel_initialization

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_size = inputs.shape[-2:]
        stem = self.stem(inputs)
        low = self.layer1(self.maxpool(stem))
        features = self.layer2(low)
        features = self.layer3(features)
        features = self.layer4(features)
        context = self.aspp(features)
        low = self.low_projection(low)
        context = F.interpolate(
            context, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        outputs = self.head(self.decoder(torch.cat((context, low), dim=1)))
        return F.interpolate(
            outputs, size=input_size, mode="bilinear", align_corners=False
        )

    def optimizer_parameter_groups(
        self, encoder_learning_rate: float, decoder_learning_rate: float
    ) -> list[dict]:
        return [
            {
                "params": (
                    parameter
                    for module in (
                        self.stem,
                        self.layer1,
                        self.layer2,
                        self.layer3,
                        self.layer4,
                    )
                    for parameter in module.parameters()
                ),
                "lr": encoder_learning_rate,
            },
            {
                "params": (
                    parameter
                    for module in (
                        self.aspp,
                        self.low_projection,
                        self.decoder,
                        self.head,
                    )
                    for parameter in module.parameters()
                ),
                "lr": decoder_learning_rate,
            },
        ]


class VariableWidthResUNet(nn.Module):
    """Direct compact ResUNet with adjustable encoder and bridge widths."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        bridge_channels: int,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        if base_channels < 2 or bridge_channels < base_channels * 8:
            raise ValueError(
                "base_channels must be >=2 and bridge_channels >= 8*base_channels"
            )
        channels = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.encoder1 = ResidualBlock(
            in_channels, channels[0], normalization=normalization
        )
        self.encoder2 = ResidualBlock(
            channels[0], channels[1], stride=2, normalization=normalization
        )
        self.encoder3 = ResidualBlock(
            channels[1], channels[2], stride=2, normalization=normalization
        )
        self.encoder4 = ResidualBlock(
            channels[2], channels[3], stride=2, normalization=normalization
        )
        self.bridge = ResidualBlock(
            channels[3], bridge_channels, stride=2, normalization=normalization
        )
        self.decoder4 = UpBlock(
            bridge_channels, channels[3], channels[3], normalization=normalization
        )
        self.decoder3 = UpBlock(
            channels[3], channels[2], channels[2], normalization=normalization
        )
        self.decoder2 = UpBlock(
            channels[2], channels[1], channels[1], normalization=normalization
        )
        self.decoder1 = UpBlock(
            channels[1], channels[0], channels[0], normalization=normalization
        )
        self.head = nn.Conv2d(channels[0], 1, 1)
        self.base_channels = base_channels
        self.bridge_channels = bridge_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        bridge = self.bridge(encoder4)
        outputs = self.decoder4(bridge, encoder4)
        outputs = self.decoder3(outputs, encoder3)
        outputs = self.decoder2(outputs, encoder2)
        outputs = self.decoder1(outputs, encoder1)
        return self.head(outputs)


def optimizer_parameter_groups(
    model: nn.Module, encoder_learning_rate: float, decoder_learning_rate: float
) -> list[dict]:
    if isinstance(model, (ResNet34DeepLabV3Plus,)):
        return model.optimizer_parameter_groups(
            encoder_learning_rate, decoder_learning_rate
        )
    return [{"params": model.parameters(), "lr": decoder_learning_rate}]
