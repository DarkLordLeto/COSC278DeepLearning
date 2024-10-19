from torch import optim, nn, save
from torch.utils.data import DataLoader
from torch import eye as E
from os import path, mkdir
from matplotlib import pyplot as plt


def train(model, train_ds, val_ds, train_opts, exp_dir=None):
    """
    fits a PyTorch module to a given dataset

    Arguments
    ---------
    model: (nn.Module), the network to train
    train_ds: (TensorDataset), the training set composed of the inputs and the targets. Both the inputs
            and targets are of size (N x T) where N is the total number of sequences in the training set
             and T is the sequence length
    val_ds: (TensorDataset), the validation set composed of the inputs and the targets. Both the inputs tensor
            the targets tensor are of size (N' x T) where N' is the total number of sequence in the validation set.
    train_opts: (dict), the training schedule. train_opts has keys: batch_size (int), num_epochs (int) and lr (float)
    exp_dir: (string), optional, if specified, the model's state is saved to exp_dir at the end of every epoch.
            Saving the model's checkpoints is skipped if exp_dir is not a valid directory.
    :return:
    """
    print(f"Training on {len(train_ds)} and validating on {len(val_ds)} sequences.")

    optimizer = optim.Adam(model.parameters(), lr=train_opts['lr'], weight_decay=train_opts['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    train_dl = DataLoader(train_ds, batch_size=train_opts['batch_size'], drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=train_opts['batch_size'], drop_last=True)

    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []

    for i in range(train_opts['num_epochs']):
        model.train()
        tr_l, tr_a = fit(model, train_dl, criterion, optimizer)
        tr_loss.append(tr_l)
        tr_acc.append(tr_a)

        # validation
        model.eval()
        val_l, val_a = fit(model, val_dl, criterion)
        val_loss.append(val_l)
        val_acc.append(val_a)

        if exp_dir:
            if not path.exists(exp_dir):
                try:
                    mkdir(exp_dir)
                    save(model.state_dict(), path.join(exp_dir, f"checkpoint _{i + 1}.pt"))
                except FileNotFoundError:
                    pass
            else:
                save(model, path.join(exp_dir, f"checkpoint _{i + 1}.pt"))

        print(f"[{i + 1}/{train_opts['num_epochs']}: tr_loss {tr_l:.4} tr_acc {tr_a:.2%} "
              f"val_loss {val_l:.4} val_acc {val_a:.2%}]")
    plot(tr_loss, tr_acc, val_loss, val_acc)


def fit(model, data_dl, criterion, optimizer=None):
    """
        Executes one epoch of training or validation

        Arguments
        --------
        model: (nn.Module), the RNN model
        data_dl: (DataLoader), the training or validation dataloader
        criterion: (CrossEntropy) for this task, the loss function
        optimizer: (Adam) for this task, the optimization function

        Returns
        -------
        loss_e: (float), The average loss over the number of epochs
        acc_e: (int) the number of instances which have been predicted correctly
    """
    loss_e = 0
    acc_e = 0
    num = 0
    one_hot_gen = E(model.rnn.vocab_size)
    h, c = model.initial_hidden_states(data_dl.batch_size)
    for x, y in data_dl:
        # expand the input mini-batch into a one-hot-encoded vectors
        # and remove the hidden and cell states from the computational graph
        x = one_hot_gen[x]
        h, c = h.detach(), c.detach()
        y = y.view(-1)
        output, h, c = model(x, h, c)

        # the loss
        loss = criterion(output, y)
        loss_e += loss.item()

        # accuracy
        acc_e += y.eq(output.argmax(dim=1)).sum().item()
        num += y.numel()

        # if in training mode, take a training step
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_e /= len(data_dl)
    acc_e /= num
    return loss_e, acc_e


def plot(tr_loss, tr_acc, val_loss, val_acc):
    """
    plots the training metrics

    Arguments
    ---------
    tr_loss: (list), the average epoch loss on the training set for each epoch
    tr_acc: (list), the epoch categorization accuracy on the training set for each epoch
    val_loss: (list), the average epoch loss on the validation set for each epoch
    val_acc: (list), the epoch categorization accuracy on the validation set for each epoch

    """

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    n = [i + 1 for i in range(len(tr_loss))]

    ax1.plot(n, tr_loss, 'bs-', markersize=3, label="train")
    ax1.plot(n, val_loss, 'rs-', markersize=3, label="val")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    tr_acc = [x * 100 for x in tr_acc]
    val_acc = [x * 100 for x in val_acc]
    ax2.plot(n, tr_acc, 'bo-', markersize=3, label="train")
    ax2.plot(n, val_acc, 'ro-', markersize=3, label="val")
    ax2.legend(loc="lower right")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Epoch")
