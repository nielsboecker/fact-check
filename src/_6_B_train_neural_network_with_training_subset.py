import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataaccess.files_constants import GENERATED_NN_PREPROCESSED_TRAINING_DATA, GENERATED_NEURAL_NETWORK_MODEL, \
    GENERATED_NEURAL_NETWORK_LOSS_HISTORY
from dataaccess.files_io import write_pickle
from model.NN_feed_forward_model import FeedForwardNeuralNetworkModel
from model.fever_claims_dataset import FeverClaimsDataset

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='use less data', action='store_true')
parser.add_argument('--num_epochs', type=int, default=100000)
parser.add_argument('--learning_rate', type=float, default=0.01)
args = parser.parse_args()

train_dataset = FeverClaimsDataset(GENERATED_NN_PREPROCESSED_TRAINING_DATA, args.debug)

batch_size = 100
num_iterations = 3000
num_epochs = int(num_iterations / len(train_dataset) / batch_size)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


input_dimension = 607
hidden_dimension = 100
output_dimension = 2

model = FeedForwardNeuralNetworkModel(input_dimension, hidden_dimension, output_dimension)

# loss class
criterion = nn.CrossEntropyLoss()
loss_history = []

# optimiser
learning_rate = 0.1
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        input_var = Variable(inputs)
        labels_var = Variable(labels)

        # clear gradients
        optimiser.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calculate loss: softmax, cross entropy loss
        loss = criterion(outputs, labels)

        # get gradients
        loss.backward()

        # update params
        optimiser.step()

        if i % 100 == 0:
            print('Iteration: {}\tLoss: {:.5f}'.format(i, loss.data))

write_pickle(GENERATED_NEURAL_NETWORK_MODEL, model)
write_pickle(GENERATED_NEURAL_NETWORK_LOSS_HISTORY, loss_history)
