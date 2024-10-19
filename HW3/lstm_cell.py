from torch import nn, sigmoid, tanh, Tensor
from math import sqrt


class LSTMCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Creates an RNN layer with an LSTM activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        super(LSTMCell, self).__init__()
        self.vocab_size = vocab_size
        #print(self.vocab_size)
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables

        
        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables

        # W, the input weights matrix has size (n x (4 * m)) where n is
        # the number of input features and m is the hidden size
        # V, the hidden state weights matrix has size (m, (4 * m))
        # b, the vector of biases has size (4 * m)
        
        self.W = nn.Parameter(Tensor(vocab_size, 4 * hidden_size))
        self.V = nn.Parameter(Tensor(hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(Tensor(4 * hidden_size))

        k = 1 / sqrt(hidden_size)
        self.W.data.uniform_(-k, k)
        self.V.data.uniform_(-k, k)
        self.b.data.uniform_(-k, k)

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an LSTM layer

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input-features.
            If the RNN has only one layer at each time step, x is the input data of the current time-step.
            In a multi-layer RNN, x is the previous layer's hidden state (usually after applying a dropout)
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m), the cell state of the previous time step

        Return
        ------
        h_out: (Tensor) of size (B x m), the new hidden
        c_out: (Tensor) of size (B x m), he new cell state

        """
        batch_size = x.size(0)
        
        a = x @ self.W + h @ self.V + self.b
        a_i, a_f, a_o, a_g = a.chunk(4, dim=1)
        
        i = sigmoid(a_i)
        f = sigmoid(a_f)
        o = sigmoid(a_o)
        g = tanh(a_g)
        
        c_out = f * c + i * g
        h_out = o * tanh(c_out)
        return h_out, c_out


