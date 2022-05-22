import torch.nn as nn
import torch.nn.functional as F


class FCRelu(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(FCRelu, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

