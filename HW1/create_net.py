from torch import nn
from generalized_logistic_layer import GeneralizedLogisticLayer
from fully_connected_layer import FullyConnectedLayer


def create_net(input_features, hidden_units, non_linearity, output_size):
    """
    Constructs a network based on the specifications passed as input arguments

    Arguments
    --------
    input_features: (integer) the number of input features
    hidden_units: (list) of length L where L is the number of hidden layers. hidden_units[i] denotes the
                        number of units at hidden layer i + 1 for i  = 0, ..., L - 1
    non_linearity: (list)  of length L. non_linearity[i] contains a string describing the type of non-linearity to use
                           hidden layer i + 1 for i = 0, ... L-1
    output_size: (integer), the number of units in the output layer

    Returns
    -------
    net: (Sequential) the constructed model
    """

    # instantiate a sequential network
    sequential_net = nn.Sequential()

    # add the hidden layers
    prev_units = input_features
    for i, (units, non_lin) in enumerate(zip(hidden_units, non_linearity), start=1):
        # add fully connected layer
        sequential_net.add_module(f'fc_{i}', FullyConnectedLayer(prev_units, units))
        

        # add non-linearity
        #sequential_net.add_module(f'gl_{i}', GeneralizedLogisticLayer(non_lin))

        if non_lin == 'tanh':
            sequential_net.add_module(f'tanh_{i}', nn.Tanh())
        elif non_lin == 'sigmoid':
            sequential_net.add_module(f'sigmoid_{i}', nn.Sigmoid())
        elif non_lin == 'relu':
            sequential_net.add_module(f'relu_{i}', nn.ReLU())
        else:
            raise ValueError(f"Invalid non-linearity: {non_lin}")

        prev_units = units

    # add output layer
    sequential_net.add_module('predictions', nn.Linear(prev_units, output_size))

    return sequential_net
