from ..builder import HEADS

from torch import nn


@HEADS.register_module()
class DoubleLinearHead(nn.Module):
    """分类器，判断信号是否需要分选"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out