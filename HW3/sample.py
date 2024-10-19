import numpy as np
from torch.nn import functional as F
from torch import eye as E


def sample(model, seed, sample_len, char_to_index, index_to_char, keep_seed=True, top_k=5,):
    """
    generate a sample text from a trained RNN model given a prompt

    Arguments
    ----------
    model: (nn.Module), a trained RNN model
    seed: (string), a prompt to induce the model
    sample_len: (int), the number of characters the model is expected to generate
    char_to_index: (dict), a mapping of the characters in the corpus to indices
    index_to_char: (dict), a mapping of indices to characters
    keep_seed: (boolean), flag whether or not to keep the seed as part of the generated text. If true, the seed
                forms the start of the generated text
    top_k: (int), the top k predicted characters the model should sample from.

    Returns
    ------
    output_text: (string), the generated text
    """
    model.eval()

    if keep_seed:
        output_text = seed
    else:
        output_text = ""

    h, c = model.initial_hidden_states(batch_size=1)

    try:
        for char in seed:
            char, h, c = predict(model, char, h, c, char_to_index, index_to_char, top_k=top_k)
            output_text += char
    except KeyError as e:
        print(f"Unrecognized character code {e}")

    for _ in range(sample_len):
        char, h, c = predict(model, output_text[-1], h, c, char_to_index, index_to_char, top_k=top_k)
        output_text += char

    return output_text


def predict(model, x, h, c,  char_to_index, index_to_char, top_k=5):
    """
    predicts the next character given an input character

    model: (nn.Module), the text generator
    x: (str), the input character represented as a string
    h: (Tensor) of size (1 x m) where m is the hidden size. The hidden state
    c: (Tensor) of size (1 x m), the cell state if nn.Module is an LSTM RNN
    char_to_index: (dict)
    index_to_char: (dict)
    top_k: (int)

    Return
    -------
    next_char: (string), the predicted character
    h: (Tensor) of size (1 x m), the new hidden state
    c: (Tensor) of size (1 x m), the new cell state
    """
    x = char_to_index[x]
    x = E(len(char_to_index))[x]

    out, h, c = model(x.view(1, 1, -1), h, c)
    p = F.softmax(out, dim=1)

    if top_k is None:
        top_ch = np.arange(len(char_to_index))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.cpu().numpy().squeeze()

    p = p.squeeze()
    p = np.array(p.tolist())
    char = np.random.choice(top_ch, p=p / p.sum())
    next_char = index_to_char[char.item()]
    return next_char, h, c
