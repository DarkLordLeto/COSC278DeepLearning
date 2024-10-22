from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE

# Specify the lead_data arguments
# data_path
data_path = 'xor_dataset.pt'
# mean_subtraction
mean_subtraction = False
# normalization
normalization = False

xor_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
# in_features
in_features = 2
# out_size
out_size = 2
# hidden_units
hidden_units = [3]
# non_linearity
non_linearity = ['tanh']

# create a network base on the architecture
# net
net = create_net(in_features, hidden_units, non_linearity, out_size)

# specify the training opts
# train_opts
train_opts = {
    'num_epochs': 30,
    'lr': 0.5,
    'momentum': 0.9,
    'batch_size': 4,  
    'weight_decay': 0,
    'step_size': 15,
    'gamma': 1,
}

# train  and save the model
train(net, xor_dataset, train_opts)
save(net, 'xor_solution.pt')
