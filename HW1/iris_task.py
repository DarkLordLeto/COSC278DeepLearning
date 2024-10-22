from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE %%%

# Specify the lead_data arguments
# data_path
data_path = 'iris_dataset.pt'
# mean_subtraction
mean_subtraction = True
# normalization
normalization = True

iris_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
# in_features
in_features = 4
# out_size
out_size = 3
# hidden_units
hidden_units = [16, 12]
# non_linearity
non_linearity = ['tanh', 'tanh']

# create a network base on the architecture
net = create_net(in_features, hidden_units, non_linearity, out_size)

# net

# specify the training opts
# train_opts
train_opts = {
   'num_epochs': 80, 
   'lr': 0.1,
   'momentum': 0.9,
   'batch_size': 24,
   'weight_decay': 0.0001,
   'step_size': 40, 
   'gamma': 0.1,  
}

# Train and save the trained model
train(net, iris_dataset, train_opts)
save(net, "iris_solution.pt")
