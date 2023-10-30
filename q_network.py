import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        # Convert the weights to Double
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.double()
                layer.bias.data = layer.bias.data.double()

    def forward(self, x):
        return self.fc(x.double())
