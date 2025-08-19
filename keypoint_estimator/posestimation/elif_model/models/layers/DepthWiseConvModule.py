import torch
import torch.nn as nn
from torch import Tensor
from elif_model.models.act_norm.activation import SiLU
from .ConvModule import ConvModule

class DepthWiseConvModule(nn.Module):

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
            out_channels: int,
            expansion: float = 0.5,
            norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: dict = dict(type='SiLU'),
            init_cfg: dict = None) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.depthwise_conv =  ConvModule(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                use_depthwise= True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                init_cfg=init_cfg
            )
        self.pointwise_conv = ConvModule( 
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                init_cfg=init_cfg
        )

    def forward(self, x: Tensor) -> Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x
