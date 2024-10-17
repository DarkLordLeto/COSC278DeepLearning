import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
import numpy as  np
import random

data_augmentation = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),  # Randomly crop and resize the image to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
])


def create_dataset(data_path, output_path=None, contrast_normalization=True, whiten=True):
    """
    Reads and optionally preprocesses the data.

    Arguments
    --------
    data_path: (String), the path to the file containing the data
    output_path: (String), the name of the file to save the preprocessed data to (optional)
    contrast_normalization: (boolean), flags whether or not to normalize the data (optional). Default (False)
    whiten: (boolean), flags whether or not to whiten the data (optional). Default (False)

    Returns
    ------
    train_ds: (TensorDataset, the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """
    # read the data and extract the various sets

    data = torch.load(data_path)
    data_tr, data_te = data['data_tr'], data['data_te']
    sets_tr, label_tr = data['sets_tr'], data['label_tr']

    #un comment the code below to access data augmentation
    '''trImg = []
    trLbl = []
    trSet = []
    
    
    for i in range(len(data_tr)):
        random_number = random.randint(0,1)
        image = data_tr[i]
        label = label_tr[i]
        
        if sets_tr[i]==1 & random_number ==1:
            image = data_augmentation(image)
            trImg.append(image)
            trLbl.append(label)
            trSet.append(1)

    trImg = torch.stack(trImg)
    trLbl = torch.tensor(trLbl)
    trSet = torch.tensor(trSet)
    data_tr = torch.cat((data_tr, trImg))
    label_tr = torch.cat((label_tr, trLbl))
    sets_tr = torch.cat((sets_tr, trSet))'''

    


            
    
    # apply the necessary preprocessing as described in the assignment handout.
    # You must zero-center both the training and test data
    if data_path == "image_categorization_dataset.pt":
        # do mean centering here
        mean = data_tr[sets_tr == 1].mean(dim=(0, 2, 3), keepdim=True)
        data_tr = data_tr - mean
        data_te = data_te - mean

        # %%% DO NOT EDIT BELOW %%%% #
        if contrast_normalization:
            image_std = torch.std(data_tr[sets_tr == 1], unbiased=True)
            image_std[image_std == 0] = 1
            data_tr = data_tr / image_std
            data_te = data_te / image_std
        if whiten:
            examples, rows, cols, channels = data_tr.size()
            data_tr = data_tr.reshape(examples, -1)
            W = torch.matmul(data_tr[sets_tr == 1].T, data_tr[sets_tr == 1]) / examples
            E, V = torch.linalg.eigh(W)
            E = E.real
            V = V.real

            en = torch.sqrt(torch.mean(E).squeeze())
            M = torch.diag(en / torch.max(torch.sqrt(E.squeeze()), torch.tensor([10.0])))

            data_tr = torch.matmul(data_tr.mm(V.T), M.mm(V))
            data_tr = data_tr.reshape(examples, rows, cols, channels)

            data_te = data_te.reshape(-1, rows * cols * channels)
            data_te = torch.matmul(data_te.mm(V.T), M.mm(V))
            data_te = data_te.reshape(-1, rows, cols, channels)

        

        preprocessed_data = {"data_tr": data_tr, "data_te": data_te, "sets_tr": sets_tr, "label_tr": label_tr}
        
        if output_path:
            torch.save(preprocessed_data, output_path)

    train_ds = TensorDataset(data_tr[sets_tr == 1], label_tr[sets_tr == 1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], label_tr[sets_tr == 2])

    return train_ds, val_ds

