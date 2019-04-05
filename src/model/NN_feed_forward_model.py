import torch.nn as nn


class FeedForwardNeuralNetworkModel(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int):
        super(FeedForwardNeuralNetworkModel, self).__init__()

        # Linear function
        self.fully_connected_1 = nn.Linear(input_dimension, hidden_dimension)

        # Non linearity
        self.sigmoid = nn.Sigmoid()

        # Readout linear function
        self.fully_connected_2 = nn.Linear(hidden_dimension, output_dimension)

    # def forward(self, *input): forward(*input, **kwargs)
    def forward(self, x):
        out = self.fully_connected_1(x)
        out = self.sigmoid(out)
        return self.fully_connected_2(out)
