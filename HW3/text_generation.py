from torch import save, random
from train import train
from create_sequence import create_sequence
from sample import sample
from rnn import RNN
from argparse import ArgumentParser
import matplotlib.pyplot as plt

random.manual_seed(0)


# ***** DO NOT EDIT ABOVE ****


def text_generation(rnn_type='basic_rnn', input_file="shakespeare.txt", sample_len=2000, seed="KING"):
    if rnn_type == "basic_rnn":
        # specify the sequence length, hidden size and number of layers
        seq_len = 100
        hidden_size = 128
        num_layers = 2
        # specify train opts
        train_opts = {
            # "num_epochs":
            # "lr": 
            # "batch_size":
            # "weight_decay":
            "num_epochs": 120,
            "lr": 0.001,
            "batch_size": 128,
            "weight_decay": 0.0001
        }

    elif rnn_type == "lstm_rnn":
        # specify the sequence length, hidden size and number of layers
        seq_len = 100
        hidden_size = 256
        num_layers = 2
        # specify train_opts
        train_opts = {
            # "num_epochs":
            # "lr":
            # "batch_size":
            # "weight_decay":
            "num_epochs": 120,
            "lr": 0.001,
            "batch_size": 128,
            "weight_decay": 0.0001
        }
        
    else:
        raise ValueError(f"Unknown RNN type {rnn_type}")

    train_ds, val_ds, char_to_index, index_to_char = create_sequence(input_file, seq_len=seq_len)
    #print(len(char_to_index))
    vocab_size = len(char_to_index)
    
    model = RNN(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, rnn_type=rnn_type)

    # specify an optional directory to save the model checkpoints
    exp_dir = None

    train(model, train_ds, val_ds, train_opts=train_opts, exp_dir=exp_dir)
    state = {
        'state': model.state_dict(), "hidden_size": hidden_size,
        "num_layers": num_layers, "vocab_size": vocab_size
    }

    save(state, f"{rnn_type}.pt")

    # time to compose essays
    sampled_text = sample(
        model=model,
        seed=seed,
        sample_len=sample_len,
        char_to_index=char_to_index,
        index_to_char=index_to_char
    )

    with open(f"{rnn_type}_output.txt", 'w') as f:
        f.write(sampled_text)

    print("\n", sampled_text)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    # change to "lstm_rnn for an lstm cell
    parser.add_argument("--rnn_type", default="lstm_rnn", type=str, help="Specify rnn type")
    parser.add_argument("--input_file", default="shakespeare.txt", type=str, help="The source corpus")
    parser.add_argument("--sample_len", default=2000, type=int, help="The length of text to generate")
    parser.add_argument("--seed", default="KING LEAR", type=str, help="The seed text")
    args, _ = parser.parse_known_args()

    text_generation(**args.__dict__)
