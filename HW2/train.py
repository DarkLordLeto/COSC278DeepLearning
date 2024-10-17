from torch import optim, save
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from os import path, mkdir


def train(model, train_ds, val_ds, train_opts, exp_dir=None):
    """
    Fits a categorization model on the provided data

    Arguments
    ---------
    model: (A pytorch module), the categorization model to train
    train_ds: (TensorDataset), the examples (images and labels) in the training set
    val_ds: (TensorDataset), the examples (images and labels) in the validation set
    train_opts: (dict), the training schedule. Read the assignment handout
                for the keys and values expected in train_opts
    exp_dir: (string), a directory where the model checkpoints should be saved (optional)

    """
    train_dl = DataLoader(train_ds, train_opts["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, train_opts["batch_size"] * 2, shuffle=False)

    num_tr = train_ds.tensors[0].size(0)
    num_val = val_ds.tensors[0].size(0)
    print(f"Training on {num_tr} and validating on {num_val} examples")

    # we will use stochastic gradient descent
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_opts["lr"],
        momentum=train_opts["momentum"],
        weight_decay=train_opts["weight_decay"]
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=train_opts["step_size"],
        gamma=train_opts["gamma"]
    )

    # the loss function of choice for most image categorization tasks
    # is  categorical cross-entropy
    criterion = CrossEntropyLoss()

    # track the training metrics
    epoch_loss_tr = []
    epoch_acc_tr = []
    epoch_loss_val = []
    epoch_acc_val = []

    num_epochs = train_opts["num_epochs"]

    for epoch in range(num_epochs):

        # training phase
        model.train()
        tr_loss, train_acc = fit(epoch, model, train_dl, criterion, optimizer, lr_scheduler)
        lr_scheduler.step()
        train_acc = train_acc / num_tr
        epoch_loss_tr.append(tr_loss)
        epoch_acc_tr.append(train_acc)

        # validation phase
        model.eval()
        val_loss, val_acc = fit(epoch, model, val_dl, criterion)
        val_acc = val_acc / num_val
        epoch_loss_val.append(val_loss)
        epoch_acc_val.append(val_acc)

        # it is always good to report the training metrics at the end of every epoch
        print(f"[{epoch + 1}/{num_epochs}: tr_loss {tr_loss:.4} val_loss {val_loss:.4} "
              f"t_acc {train_acc:.2%} val_acc {val_acc:.2%}]")

        # save model checkpoint if exp_dir is specified
        if exp_dir:
            if path.exists(exp_dir):
                save(model.state_dict(), path.join(exp_dir, f"checkpoint_{epoch + 1}.pt"))
            else:
                try:
                    mkdir(exp_dir)
                    save(model.state_dict(), path.join(exp_dir, f"checkpoint_{epoch + 1}.pt"))
                except FileNotFoundError:
                    pass

    # plot the training metrics at the end of training
    plot(epoch_loss_tr, epoch_acc_tr, epoch_loss_val, epoch_acc_val)


def fit(epoch, model, data_loader, criterion, optimizer=None, scheduler=None):
    """
    Executes a training (or validation) epoch
    epoch: (int), the training epoch. This parameter is used by the learning rate scheduler
    model: (a pytorch module), the categorization model begin trained
    data_loader: (DataLoader), the training or validation set
    criterion: (CrossEntropy) for this task. The objective function
    optimizer: (SGD) for this task. The optimization function (optional)
    scheduler: (StepLR) for this schedule. The learning rate scheduler (optional)

    Return
    ------
    epoch_loss: (float), the average loss on the given set for the epoch
    epoch_acc: (float), the categorization accuracy on the given set for the epoch

    """
    epoch_loss = epoch_acc = 0
    for mini_x, mini_y in data_loader:
        pred = model(mini_x).squeeze()
        loss = criterion(pred, mini_y)

        epoch_loss += loss.item()
        epoch_acc += mini_y.eq(pred.argmax(dim=1)).sum().item()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss, epoch_acc


def plot(loss_tr, acc_tr, loss_val, acc_val):
    """
    plots the training metrics

    Arguments
    ---------
    loss_tr: (list), the average epoch loss on the training set for each epoch
    acc_tr: (list), the epoch categorization accuracy on the training set for each epoch
    loss_val: (list), the average epoch loss on the validation set for each epoch
    acc_val: (list), the epoch categorization accuracy on the validation set for each epoch

    """
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    n = [i + 1 for i in range(len(loss_tr))]
    acc_tr = [x * 100 for x in acc_tr]
    acc_val = [x * 100 for x in acc_val]

    ax1.plot(n, loss_tr, 'bs-', markersize=3, label="train")
    ax1.plot(n, loss_val, 'rs-', markersize=3, label="val")
    ax1.legend(loc="upper right")
    ax1.set_title("Losses")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    ax2.plot(n, acc_tr, 'bo-',  markersize=3, label="train")
    ax2.plot(n, acc_val, 'ro-', markersize=3, label="val")
    ax2.legend(loc="upper right")
    ax2.set_title("Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Epoch")
