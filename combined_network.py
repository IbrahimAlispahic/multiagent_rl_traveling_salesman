import torch.nn as nn

class CombinedNetwork(nn.Module):
    def __init__(self, network1, network2):
        super(CombinedNetwork, self).__init__()
        self.network1 = network1
        self.network2 = network2

    def forward(self, x):
        x1 = x[:self.network1.fc[0].in_features]  # First half of input goes to network1
        x2 = x[self.network1.fc[0].in_features:]  # Second half of input goes to network2
        out1 = self.network1(x1)
        out2 = self.network2(x2)
        return (out1 + out2) / 2  # Average the outputs

    def double(self):
        self.network1.double()
        self.network2.double()
