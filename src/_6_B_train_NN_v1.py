import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataaccess.files_constants import GENERATED_NN_PREPROCESSED_TRAINING_DATA, GENERATED_NEURAL_NETWORK_MODEL, \
    GENERATED_NEURAL_NETWORK_LOSS_HISTORY, GENERATED_NN_PREPROCESSED_DEV_DATA
from dataaccess.files_io import write_pickle
from model.NN_feed_forward_model import FeedForwardNeuralNetworkModel
from model.fever_claims_dataset import FeverClaimsDataset
from util.plots import plot_loss_values

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='use less data', action='store_true')
parser.add_argument('--num_iterations', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()


train_dataset = FeverClaimsDataset(GENERATED_NN_PREPROCESSED_TRAINING_DATA, args.debug)
dev_dataset = FeverClaimsDataset(GENERATED_NN_PREPROCESSED_DEV_DATA, args.debug)
batch_size = 5000

input_size = len(train_dataset)
num_of_batches = input_size / batch_size
num_epochs = int(args.num_iterations / num_of_batches)

print('Starting training\nInput size: {:,}\nIterations: {}\nEpochs: {}'.format(input_size,
                                                                               args.num_iterations, num_epochs))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

input_dimension = 607
hidden_dimension = 100
output_dimension = 2

model = FeedForwardNeuralNetworkModel(input_dimension, hidden_dimension, output_dimension)

# loss class
criterion = nn.CrossEntropyLoss()
loss_history_frequency = 100
loss_history = []

# optimiser
optimiser = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

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

        iter = epoch * input_size + i
        if iter % loss_history_frequency == 0:
            print('Iteration: {}\tEpoch: {}\t|\tLoss: {:.5f}'.format(iter, epoch, loss.item()))
            loss_history.append(loss.item())

            #predicted = model.forward(dev_dataset)
            #acc = get_accuracy(predicted)



plot_loss_values(args.num_iterations, args.learning_rate, loss_history, loss_history_frequency)

write_pickle(GENERATED_NEURAL_NETWORK_MODEL, model)
write_pickle(GENERATED_NEURAL_NETWORK_LOSS_HISTORY, loss_history)
