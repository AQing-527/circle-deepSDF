import torch
import torch.nn as nn


# The structure proposed in DeepSDF paper
class SDFNet(nn.Module):
    def __init__(self, dropout_prob=0):
        super(SDFNet, self).__init__()
        self.fc_stack_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(2, 32)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(32, 1))
        )
        self.th = nn.Tanh()

    def forward(self, x):
        first = self.fc_stack_1(x)
        out = self.th(first)
        return out