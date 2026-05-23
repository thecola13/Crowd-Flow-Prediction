import math

import torch.nn as nn

from .unet_comp import CustomOutConv, DoubleConv, Up


def _validate_power_of_two(value: int, name: str) -> int:
    if value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return int(math.log2(value))


class UNet(nn.Module):
    """
    Generic U-Net with explicit output reduction and skip-placement controls.

    output_reduction:
      1 -> full-resolution density map, 2 -> half-resolution, 4 -> quarter, etc.

    skip_placement:
      before_pool -> classical U-Net skips after convolution and before pooling.
      after_pool  -> skips after pooling, matching the original project variant.
    """

    def __init__(
        self,
        in_channels=3,
        num_filters=32,
        depth=4,
        output_reduction=2,
        skip_placement="after_pool",
        **kwargs,
    ):
        super().__init__()
        assert depth >= 1, "Depth must be >= 1"
        assert num_filters > 0, "Base channels must be > 0"
        if skip_placement not in {"before_pool", "after_pool"}:
            raise ValueError("skip_placement must be 'before_pool' or 'after_pool'")

        target_log2 = _validate_power_of_two(output_reduction, "output_reduction")
        min_log2 = 0 if skip_placement == "before_pool" else 1
        max_log2 = depth if skip_placement == "before_pool" else depth + 1
        if not min_log2 <= target_log2 <= max_log2:
            raise ValueError(
                f"output_reduction={output_reduction} is incompatible with "
                f"depth={depth} and skip_placement='{skip_placement}'"
            )

        self.depth = depth
        self.output_reduction = output_reduction
        self.skip_placement = skip_placement
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        custom_head = kwargs.get("custom_head", False)
        base = num_filters

        if skip_placement == "before_pool":
            self.reduction_levels = [2**i for i in range(depth + 1)]
            self.channels = [base * (2**i) for i in range(depth + 1)]
            conv_in = [in_channels] + self.channels[:-1]
            self.encoder_blocks = nn.ModuleList(
                DoubleConv(in_ch, out_ch, **kwargs)
                for in_ch, out_ch in zip(conv_in, self.channels)
            )
        else:
            self.reduction_levels = [2 ** (i + 1) for i in range(depth + 1)]
            self.channels = [base * (2**i) for i in range(depth + 1)]
            self.input_block = DoubleConv(in_channels, base, **kwargs)
            self.encoder_blocks = nn.ModuleList(
                DoubleConv(self.channels[i], self.channels[i + 1], **kwargs)
                for i in range(depth)
            )

        self.target_index = self.reduction_levels.index(output_reduction)
        self.ups = nn.ModuleList()
        current_ch = self.channels[-1]
        for skip_idx in range(len(self.channels) - 2, self.target_index - 1, -1):
            skip_ch = self.channels[skip_idx]
            self.ups.append(Up(current_ch + skip_ch, skip_ch, **kwargs))
            current_ch = skip_ch

        if custom_head:
            self.outc = CustomOutConv(current_ch, **kwargs)
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(current_ch, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def _encode_before_pool(self, x):
        features = []
        for level, block in enumerate(self.encoder_blocks):
            if level > 0:
                x = self.pool(x)
            x = block(x)
            features.append(x)
        return features

    def _encode_after_pool(self, x):
        features = []
        x = self.input_block(x)
        x = self.pool(x)
        features.append(x)
        for block in self.encoder_blocks:
            x = self.pool(x)
            x = block(x)
            features.append(x)
        return features

    def forward(self, x, return_intermediates=False):
        features = (
            self._encode_before_pool(x)
            if self.skip_placement == "before_pool"
            else self._encode_after_pool(x)
        )

        intermediates = []
        x_dec = features[-1]
        skip_features = features[self.target_index : -1]
        for up, skip in zip(self.ups, reversed(skip_features)):
            x_dec = up(x_dec, skip)
            if return_intermediates:
                intermediates.append(x_dec)

        out = self.outc(x_dec)
        if return_intermediates:
            intermediates.append(out)
            return intermediates

        return out
