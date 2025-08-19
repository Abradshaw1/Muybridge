import torch
import torch.nn as nn

class DilatedConv(nn.Module): #temporal layer
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, batch_norm):
        super(DilatedConv, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.ConstantPad1d(padding, 0)) # Conv1D gives erorr with non symmetric padding
        layers.append(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation))
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_outputs))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalConv(nn.Module):
    def __init__(self, channel_list, in_channels, kernel_size, skip_conn, batch_norm, max_pooling=[2, 2, 2, 2]):
        super(TemporalConv, self).__init__()
        self.skip_conn = skip_conn
        self.layers = nn.ModuleList()
        num_levels = len(channel_list)

        assert len(max_pooling) == num_levels, "max_pooling must have the same length as channel_list"

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else channel_list[i-1]
            out_channels = channel_list[i]
            self.layers.append(DilatedConv(in_ch, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(int((kernel_size-1)*dilation_size), 0), batch_norm=batch_norm))
            if max_pooling[i] != 0:
                self.layers.append(nn.MaxPool1d(max_pooling[i]))
            
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.skip_conn:
            o = self.layers[0](x)
            s = o
            for i, layer in enumerate(self.layers[1:]):
                # We apply residuals on every other layer as in Imagenet
                # Skipping first layer
                o = layer(o)
                if (i % 2):
                    o = o + s
                    s = o
            return o
        else:
            return self.network(x)