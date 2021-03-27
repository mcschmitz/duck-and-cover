from typing import Any

import numpy as np
import torch
from torch import nn


class MinibatchStdDev(nn.Module):
    """
    Calculates the minibatch standard deviation and adds it to the output.

    Calculates the average standard deviation of each value in the input
    map across the channels and concatenates a blown up version of it
    (same size as the input map) to the input
    """

    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        squared_diff = torch.square(x - mean)
        avg_squared_diff = torch.mean(squared_diff, dim=0, keepdim=True)
        avg_squared_diff += 1e-8
        sd = torch.sqrt(avg_squared_diff)
        avg_sd = torch.mean(sd)
        shape = x.shape
        ones = torch.ones((shape[0], 1, shape[2], shape[3])).to(avg_sd.device)
        output = ones * avg_sd
        combined = torch.cat([x, output], dim=1)
        return combined


class ScaledDense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        gain: float = None,
        use_dynamic_wscale: bool = True,
    ):
        """
        Dense layer with weight scaling.

        Ordinary Dense layer, that scales the weights by He's scaling
        factor at each forward pass. He's scaling factor is defined as
        sqrt(2/fan_in) where fan_in is the number of input units of the
        layer
        """
        super().__init__(in_features, out_features, bias)
        self.use_dynamic_wscale = use_dynamic_wscale
        self.gain = gain if gain else np.sqrt(2)

        torch.nn.init.kaiming_normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        if self.use_dynamic_wscale:
            self.gain = gain if gain else np.sqrt(2)
            fan_in = self.in_features
            self.gain = gain / np.sqrt(max(1.0, fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dynamic_wscale:
            return nn.functional.linear(x, self.weight * self.gain, self.bias)
        return nn.functional.linear(x, self.weight, self.bias)


class ScaledConv2dTranspose(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        use_dynamic_wscale: bool = False,
        gain: float = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        self.use_dynamic_wscale = use_dynamic_wscale
        if self.use_dynamic_wscale:
            self.gain = gain if gain else np.sqrt(2)
            fan_in = np.prod(self.kernel_size) * self.in_channels
            self.gain = self.gain / np.sqrt(max(1.0, fan_in))

    def forward(
        self, x: torch.Tensor, output_size: Any = None
    ) -> torch.Tensor:
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size
        )
        weight = (
            self.weight * self.gain if self.use_dynamic_wscale else self.weight
        )
        return torch.conv_transpose2d(
            input=x,
            weight=weight,  # scale the weight on runtime
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class ScaledConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        use_dynamic_wscale: bool = False,
        gain: float = None,
    ):
        """
        Scaled Convolutional Layer.

        Scales the weights on each forward pass down to by He's initialization
        factor. For Progressive Growing GANS this results in an equalized
        learning rate for all layers, where bigger layers are scaled down.

        Args:
            gain: Constant to upscale the weight sclaing
            use_dynamic_wscale: Switch on or off dynamic weight scaling.
                Switching it off results in a ordinary 2D convolutional layer.
            **kwargs:
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        self.use_dynamic_wscale = use_dynamic_wscale
        if self.use_dynamic_wscale:
            self.gain = gain if gain else np.sqrt(2)
            fan_in = np.prod(self.kernel_size) * self.in_channels
            self.gain = self.gain / np.sqrt(max(1.0, fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (
            self.weight * self.gain if self.use_dynamic_wscale else self.weight
        )
        return torch.conv2d(
            input=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor, alpha: float = 1e-8) -> torch.Tensor:
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
