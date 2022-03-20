import torch
import torch.nn as nn


# The structure proposed in DeepSDF paper
class SDFNet(nn.Module):
    def __init__(self, layers, dropout_prob=0):
        super(SDFNet, self).__init__()
        self.layers = layers
        if self.layers == 1:
            self.fc_stack_1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(2, 256)),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.utils.weight_norm(nn.Linear(256, 1))
            )
        elif self.layers == 2:
            self.fc_stack_1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(2, 512)),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.utils.weight_norm(nn.Linear(512, 1))
            )
        elif self.layers == 3:
            self.fc_stack_1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(2, 512)),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.utils.weight_norm(nn.Linear(512, 512)),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.utils.weight_norm(nn.Linear(512, 1))
            )
        elif self.layers == 4:
            self.fc_stack_1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(2, 512)),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.utils.weight_norm(nn.Linear(512, 512)),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.utils.weight_norm(nn.Linear(512, 512)),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.utils.weight_norm(nn.Linear(512, 1))

            )
        self.th = nn.Tanh()

    def forward(self, x):
        first = self.fc_stack_1(x)
        out = self.th(first)
        return out

