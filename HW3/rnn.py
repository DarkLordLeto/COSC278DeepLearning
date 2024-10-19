from torch import nn, zeros
from create_rnn import CustomRNN


class RNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1, rnn_type='basic_rnn'):
        super().__init__()
        """
        Creates an RNN given the rnn type {basic_rnn, lstm_rnn}
         
        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn layer.
        num_layers: (int), the number of RNN layers at each time step 
        rnn_type: (string), the desired rnn type. 
        
        """
        self.rnn = CustomRNN(vocab_size, hidden_size, num_layers, rnn_type)
        self.fc = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an RNN for a given sequence

        Arguments
        ----------
        x: (Tensor) of size (B x T x n) where B is the mini-batch size, T is the sequence length and n is the
            number of input features. x the mini-batch of input sequence
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m). c is the cell state of the previous time step if the rnn is an LSTM RNN

        Return
        ------
        outs: (Tensor) of size (B x T x m), the hidden state at the end of each time step in order
        h: (Tensor) of size (B x m), the hidden state of the last time step
        c: (Tensor) of size (B x m), the cell state of the last time step, if the rnn is a basic_rnn, c should be
            the cell state passed in as input.
        """
        x, h, c = self.rnn(x, h, c)
        x = x.reshape(-1, self.rnn.hidden_size)
        x = self.fc(x)

        return x, h, c

    def initial_hidden_states(self, batch_size):
        """
        a utility method for creating initial hidden and cell states

        Arguments
        ---------
        batch_size: (int), the mini-batch size

        Return
        -----
        h: (Tensor) of size (l x B x m) where l is the number of layers, B is the mini-batch size and m is the hidden
            size. h is the initial hidden state
        c: (Tensor) of size (l x B x m), the initial cell state.
        """
        h = zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)
        c = zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)

        return h, c


