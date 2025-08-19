import torch
import torch.nn as nn
from torch import Tensor
from ..act_norm.activation import SiLU

class ConvModule(nn.Module):

    """Convolutional Layer (Optionally + Normalization Layer) + Activation

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride(int): Stride of the kernel
            Defaults to 1
        padding (int): Padding size around the input image
            Defaults to 0
        dilation (int):  Dilation of the input kernel
            Defaults to 0
        padding_mode (string): How to fill the padding around the image
            Defaults to "zeros"
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. 
            Defaults to dict(type='BN', momentum=0.03, eps=0.001, requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Weight Initialization config dict.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            padding_mode: str = "zeros",
            use_depthwise: bool = False,
            norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: dict = dict(type='SiLU'),
            init_cfg: dict = None,
        )  -> None:
        super().__init__()

        group_size = in_channels if use_depthwise else 1
        self.conv = nn.Conv2d(in_channels,
                         out_channels,
                         kernel_size,
                         stride = stride,
                         padding = padding,
                         dilation = dilation,
                         padding_mode = padding_mode,
                         groups=group_size,
                         bias=False)
        
        if act_cfg["type"] == 'ReLU':
            self.act_layer = nn.ReLU()
        elif act_cfg["type"] == 'SiLU':
            self.act_layer = SiLU(inplace=True)
        elif act_cfg["type"] == 'Tanh':
            self.act_layer = nn.Tanh()
        elif act_cfg["type"] == 'Sigmoid':
            self.act_layer = nn.Sigmoid()
        elif act_cfg["type"] == 'LeakyReLU':
            self.act_layer = nn.LeakyReLU()
        else:
            self.act_layer = nn.ReLU()


        self.bn = nn.BatchNorm2d(out_channels, momentum=norm_cfg["momentum"], eps = norm_cfg["eps"])

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


    