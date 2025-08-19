import torch.nn as nn

from models.submodules import conv_submodules, dense_submodules, transformer_submodules

class Transformer(nn.Module):
    def __init__(self, device, time_steps, n_features, n_heads, N, dense_layers, output_size, dropout, skip_conn, batch_norm):
        super(Transformer, self).__init__()
        
        self.pos_enc = transformer_submodules.PositionalEncoding(time_steps, n_features, device, n=100)
        self.trans = transformer_submodules.TransformerBlock(time_steps, n_features, n_heads, N, dropout)
        self.avg_pool = nn.AvgPool1d(128)
        self.DenseNet = dense_submodules.DenseNet(dense_layers, output_size, 96, dropout, skip_conn, batch_norm) # TODO


    def forward(self, x):
        x = x.permute(0, 2, 1)

        o = self.pos_enc(x)
        o = self.trans(o)
        o = self.avg_pool(o.permute(0, 2, 1))
        o = self.DenseNet(o.flatten(1))

        return o
