import torch
from torch.utils.data import TensorDataset


def create_sequence(input_file, seq_len, val_ratio=0.15):
    """
    creates sequences of character encodings given an input text file

    Arguments
    ---------
    input_file: (string), the file path of the input text
    seq_len: (int), the expected length of each sequence. If the length of the last sequence
            is less than seq_len, it is discarded.
    val_ratio: (float), strictly between 0 and 1 - the fraction of the text to reserve for validation

    Return
    ------
    train_ds: (TensorDataset), the inputs and the targets in the training set. Both the inputs tensor and the
              targets tensor are of size (N x T) where N is the total number of sequences in the set and T is the
              sequence length
    val_ds: (TensorDataset), the inputs and targets in the validation set. Both the inputs tensor and the
              targets tensor are of size (N' x T) where N' is the total number of sequences in the set
    char_to_index: (dict), a mapping of the characters in the corpus to indices
    index_to_char: (dict), a mapping of indices to the characters

    """
    assert 0 <= val_ratio < 1
    with open(input_file, 'r') as f:
        text = f.read()

    vocab = list(set(text))

    index_to_char = {index: character for index, character in enumerate(vocab)}
    char_to_index = {character: index for index, character in enumerate(vocab)}

    encodings = [char_to_index[ch] for ch in text]
    i = 0
    x = []
    y = []
    n = len(encodings)
    while i + seq_len < n - 1:
        x.append(encodings[i: i + seq_len])
        y.append(encodings[i + 1: i + seq_len + 1])
        i += seq_len

    num_train = int((1 - val_ratio) * len(x))
    train_ds = TensorDataset(torch.tensor(x[:num_train]), torch.tensor(y[:num_train]))
    val_ds = TensorDataset(torch.tensor(x[num_train:]), torch.tensor(y[num_train:]))

    return train_ds, val_ds, char_to_index, index_to_char
