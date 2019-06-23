import torch.nn as nn


class LogistRegression(nn.Module):

    def __init__(self):
        super(LogistRegression,self).__init__()
        self.logistRegression = nn.Linear(2, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.logistRegression(x)
        output = self.sigmoid(x)
        return output