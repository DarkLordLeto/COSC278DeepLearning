import torch
from torch.utils.data import TensorDataset


def load_dataset(dataset_path, mean_subtraction, normalization):
    """
    Reads the train and validation data

    Arguments
    ---------
    dataset_path: (string) representing the file path of the dataset
    mean_subtraction: (boolean) specifies whether to do mean centering or not. Default: False
    normalization: (boolean) specifies whether to normalizes the data or not. Default: False

    Returns
    -------
    train_ds (TensorDataset): The features and their corresponding labels bundled as a dataset
    """
    # Load the dataset and extract the features and the labels
    dataset = torch.load(dataset_path)
    features, labels = dataset['features'], dataset['labels']

    # Do mean_subtraction if it is enabled
    if mean_subtraction:
        train_mean = features.mean(dim=0, keepdim=True)
        features = features - train_mean

    # do normalization if it is enabled
    if normalization:
        train_std = features.std(dim=0, keepdim=True)
        train_std[train_std == 0] = 1  # Avoid division by zero
        features = features / train_std

    # create tensor dataset train_ds
    train_ds = TensorDataset(features, labels)


    return train_ds
