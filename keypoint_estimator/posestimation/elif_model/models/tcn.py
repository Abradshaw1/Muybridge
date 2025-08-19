import torch.nn as nn
from models.submodules import conv_submodules, dense_submodules


class TCN(nn.Module):
    def __init__(self, device, seq_len, in_channels, channel_list, kernel_size, max_pooling, dense_list, output_size, dropout, skip_conn, batch_norm):
        super(TCN, self).__init__()

        temporal_conv_params = {
            'channel_list': channel_list,
            'in_channels': in_channels,
            'kernel_size': kernel_size,
            'skip_conn': skip_conn,
            'batch_norm': batch_norm,
            'max_pooling': max_pooling
        }
        self.TemporalConvNet = conv_submodules.TemporalConv(**temporal_conv_params)

        for i in range(len(max_pooling)):
            if max_pooling[i] != 0:
                # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
                seq_len = int((seq_len - max_pooling[i]) / max_pooling[i]) + 1

        dense_inputs = seq_len * channel_list[-1]
        self.DenseNet = dense_submodules.DenseNet(dense_list, output_size, dense_inputs, dropout, skip_conn, batch_norm)

    def forward(self, data):
        o = self.TemporalConvNet(data).flatten(1)

        o = self.DenseNet(o)
        return o