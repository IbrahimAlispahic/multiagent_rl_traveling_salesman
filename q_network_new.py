import torch.nn as nn


class QNetwork(nn.Module):
    """
    A neural network for Q-learning with two hidden layers.
    """

    def __init__(self, input_dim, num_actions):
        """
        Initialize the QNetwork.

        :param input_dim: Dimension of the input features.
        :param num_actions: Number of possible actions.
        """
        super(QNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(128, num_actions)

        self._convert_to_double()

    def _convert_to_double(self):
        """
        Convert the weights and biases of all layers to double precision.
        """
        for layer in [self.layer1, self.layer2, self.output_layer]:
            layer.weight.data = layer.weight.data.double()
            layer.bias.data = layer.bias.data.double()

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = x.double()
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.output_layer(x)
