"""def cnn_categorization_improved(netspec_opts):

 Constructs a network for the improved categorization model.

 Arguments
 --------
 netspec_opts: (dictionary), the improved network's architecture.

 Returns
 -------
 A categorization model which can be trained by PyTorch


return net"""

from torch import nn

def cnn_categorization_improved(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()
    filterNum = 3
    # add layers as specified in netspec_opts to the network
    for i, layer_type in enumerate(netspec_opts['layer_type']):
        if layer_type == 'conv':
            #print(i)
            
            conv_layer = nn.Conv2d(
                
                in_channels=filterNum,  
                out_channels=netspec_opts['num_filters'][i],  
                kernel_size=netspec_opts['kernel_size'][i],  
                stride=netspec_opts['stride'][i],  
                padding=0 if i==len(netspec_opts['layer_type'])-1 else (netspec_opts['kernel_size'][i]-1)//2
            )
            net.add_module(f'conv_{i+1}', conv_layer)
            filterNum = netspec_opts['num_filters'][i]

        elif layer_type == 'bn':
            bn_layer = nn.BatchNorm2d(num_features=netspec_opts['num_filters'][i])
            net.add_module(f'bn_{i+1}', bn_layer)

        elif layer_type == 'relu':
            net.add_module(f'relu_{i+1}', nn.ReLU())

        elif layer_type == 'pool':
            pool_layer = nn.AvgPool2d(kernel_size=netspec_opts['kernel_size'][i], stride=netspec_opts['stride'][i])
            net.add_module(f'pool_{i+1}', pool_layer)

        elif layer_type == 'dropout':  # Add dropout layer condition
            dropout_layer = nn.Dropout(p=0.25)
            net.add_module(f'dropout_{i+1}', dropout_layer)

    

    return net

