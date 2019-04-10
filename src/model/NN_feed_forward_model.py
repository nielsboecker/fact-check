import torch.nn as nn


class ShallowFeedForwardNeuralNetworkModel(nn.Module):
    def __init__(self, input_dimension: int,
                 hidden_dimension: int,
                 output_dimension: int,
                 non_linearity: str):
        super(ShallowFeedForwardNeuralNetworkModel, self).__init__()

        # Linear function
        self.fully_connected_1 = nn.Linear(input_dimension, hidden_dimension)

        # Non linearity
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'sigmoid':
            self.non_linearity= nn.Sigmoid()
        else:
            raise NotImplementedError

        # Readout linear function
        self.fully_connected_2 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, input):
        out = self.fully_connected_1(input)
        out = self.non_linearity(out)
        return self.fully_connected_2(out)


class DeepFeedForwardNeuralNetworkModel(nn.Module):
    def __init__(self, input_dimension: int,
                 hidden_dimension: int,
                 output_dimension: int,
                 non_linearity: str):
        super(DeepFeedForwardNeuralNetworkModel, self).__init__()

        # in
        self.fully_connected_1 = nn.Linear(input_dimension, hidden_dimension)

        # hidden
        self.fully_connected_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fully_connected_3 = nn.Linear(hidden_dimension, hidden_dimension)

        # readout
        self.fully_connected_4 = nn.Linear(input_dimension, output_dimension)

        # Non linearity
        if non_linearity == 'relu':
            self.non_linearity_1 = nn.ReLU()
            self.non_linearity_2 = nn.ReLU()
            self.non_linearity_3 = nn.ReLU()
        elif non_linearity == 'sigmoid':
            self.non_linearity_1 = nn.Sigmoid()
            self.non_linearity_2 = nn.Sigmoid()
            self.non_linearity_3 = nn.Sigmoid()
        else:
            raise NotImplementedError

        # Readout linear function
        self.fully_connected_2 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, input):
        out = self.fully_connected_1(input)
        out = self.non_linearity_1(out)
        out = self.fully_connected_2(out)
        out = self.non_linearity_2(out)
        out = self.fully_connected_3(out)
        out = self.non_linearity_3(out)
        out = self.fully_connected_4(out)
        return out
