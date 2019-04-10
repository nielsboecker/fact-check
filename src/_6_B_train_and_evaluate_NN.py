# Disclosure: This file has been written after studying https://github.com/yunjey/pytorch-tutorial

import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from dataaccess.files_constants import GENERATED_NN_PREPROCESSED_DATA, GENERATED_NEURAL_NETWORK_MODEL, \
    GENERATED_NEURAL_NETWORK_LOSS_HISTORY, GENERATED_NN_PREPROCESSED_DEV_DATA
from dataaccess.files_io import write_pickle
from model.NN_feed_forward_model import ShallowFeedForwardNeuralNetworkModel, DeepFeedForwardNeuralNetworkModel
from model.fever_claims_dataset import FeverClaimsDataset
from util.plots import plot_loss_values

parser = argparse.ArgumentParser()
#parser.add_argument('--debug', help='use less data', action='store_true')
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')
#parser.add_argument('--num_iterations', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--input_dimension', type=int, default=600)
parser.add_argument('--hidden_dimension', type=int, default=100)
parser.add_argument('--non_linearity', choices=['relu', 'sigmoid'], default='relu')
parser.add_argument('--optimiser', choices=['sgd', 'adam'], default='adam')
parser.add_argument('--preprocessed_format', choices=['v1', 'v2', 'v3', 'v4'], required=True)
parser.add_argument('--depth', choices=['shallow', 'deep'], default='deep')
parser.add_argument('--num_epochs', type=int, default=20)
args = parser.parse_args()

train_path = GENERATED_NN_PREPROCESSED_DATA.format('train', args.preprocessed_format)
train_dataset = FeverClaimsDataset(train_path)
dev_path = GENERATED_NN_PREPROCESSED_DATA.format('dev', args.preprocessed_format)
dev_dataset = FeverClaimsDataset(dev_path)

#num_of_batches = input_size / args.batch_size
#num_epochs = int(args.num_iterations / num_of_batches)

print('Starting training...\nInput size: {:,}\nEpochs: {}'.format(len(train_dataset), args.num_epochs))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_dataset = FeverClaimsDataset(GENERATED_NN_PREPROCESSED_DEV_DATA)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)

# predicting one of two labels (supports / refutes)
output_dimension = 2

model = None
if args.depth == 'shallow':
    model = ShallowFeedForwardNeuralNetworkModel(args.input_dimension, args.hidden_dimension,
                                                 output_dimension, args.non_linearity).to(args.device)
elif args.depth == 'deep':
    model = DeepFeedForwardNeuralNetworkModel(args.input_dimension, args.hidden_dimension,
                                              output_dimension, args.non_linearity).to(args.device)

# loss class
criterion = nn.CrossEntropyLoss()
loss_history_frequency = 100
loss_history = []

# optimiser
optimiser = None
if args.optimiser == 'sgd':
    optimiser = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
elif args.optimiser == 'adam':
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# train
num_iterations = len(train_loader)
for epoch in range(args.num_epochs):
    for i, (inputs, labels) in tqdm(enumerate(train_loader)):

        input_var = Variable(inputs).to(args.device) #todo
        labels_var = Variable(labels).to(args.device)

        # clear gradients
        optimiser.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calculate cross entropy loss
        loss = criterion(outputs, labels)

        # get gradients in backward pass
        loss.backward()

        # update params
        optimiser.step()

        #iter = epoch * input_size + i
        #if iter % loss_history_frequency == 0:
        if (i + 1) % loss_history_frequency == 0:
            print('Iteration: {}/{}\tEpoch: {}/{}\t|\tLoss: {:.5f}'
                  .format(i + 1, num_iterations, epoch, args.num_epochs, loss.item()))
            loss_history.append(loss.item())

        #if i == num_iterations - 1:
            #'Final accuracy on dev set: '
            #acc = get_accuracy(predicted)

print('Done with training...')

# test result
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy on dev subset: {:.5f} %'.format(correct / total))

# plot loss history
plot_loss_values(args.num_iterations, args.learning_rate, loss_history, loss_history_frequency)

# save results
write_pickle(GENERATED_NEURAL_NETWORK_MODEL.format(args.preprocessed_format), model)
write_pickle(GENERATED_NEURAL_NETWORK_LOSS_HISTORY.format(args.preprocessed_format), loss_history)
