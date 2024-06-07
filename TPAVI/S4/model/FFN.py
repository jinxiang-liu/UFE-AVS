import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, num_layers=2, prob_aud_dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.prob_aud_dropout = prob_aud_dropout
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x, need_fp=False):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if need_fp:
            x_fp = nn.Dropout(self.prob_aud_dropout)(x)
            return x, x_fp
        return x