import torch.nn as nn

class ModifiedQNetwork(nn.Module):
    """
    A modified neural network for Q-learning with three hidden layers and dropout.
    """

    def __init__(self, input_dim, num_actions):
        super(ModifiedQNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(64, num_actions)

        self._convert_to_double()

    def _convert_to_double(self):
        """
        Convert the weights and biases of all layers to double precision.
        """
        for layer in [self.layer1, self.layer2, self.layer3, self.output_layer]:
            layer.weight.data = layer.weight.data.double()
            layer.bias.data = layer.bias.data.double()

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = x.double()
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        return self.output_layer(x)
