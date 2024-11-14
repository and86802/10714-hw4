"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math

from .nn_basic import Parameter, Module

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        one = init.ones(*x.shape, device=x.device)
        return one / (one + ops.exp(-x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION

        bound = math.sqrt(1 / hidden_size)    

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        def init_param(*shape, requires_grad=True):
            """Helper function for parameter initialization."""
            return Parameter(
                init.rand(*shape, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=requires_grad)
            )

        # Initialize weights
        self.W_ih = init_param(input_size, hidden_size)
        self.W_hh = init_param(hidden_size, hidden_size)

        # Initialize biases if applicable
        self.bias_ih = init_param(hidden_size) if bias else None
        self.bias_hh = init_param(hidden_size) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size, _ = X.shape

        # Initialize hidden state if not provided
        h = h if h is not None else init.zeros(batch_size, self.hidden_size, device=X.device, dtype=X.dtype)

        # Compute output
        out = (X @ self.W_ih) + (h @ self.W_hh)

        # Add biases if they are defined
        if self.bias_ih is not None:
            bias_broadcast_shape = (batch_size, self.hidden_size)
            out += self.bias_ih.broadcast_to(bias_broadcast_shape) + self.bias_hh.broadcast_to(bias_broadcast_shape)

        # Apply non-linearity
        activation_fn = ops.tanh if self.nonlinearity == 'tanh' else ops.relu
        return activation_fn(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = []

        for layer in range(self.num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            
            rnn_cell = RNNCell(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype
            )
            
            self.rnn_cells.append(rnn_cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)

        X_t = ops.split(X, axis=0)
        H_n = ops.split(h0, axis=0)
        H_t = []

        for t in range(seq_len):
            x_t = X_t[t]
            h_n = []

            for l in range(self.num_layers):
                rnn_cell = self.rnn_cells[l]
                h_tl = rnn_cell(x_t, H_n[l])
                x_t = h_tl
                h_n.append(h_tl)

                if l == self.num_layers - 1:
                    H_t.append(h_tl)

            H_n = h_n

        H_t = ops.stack(H_t, axis=0)
        H_n = ops.stack(H_n, axis=0)

        return H_t, H_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = math.sqrt(1 / hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

        def create_param(shape):
            return Parameter(init.rand(
                *shape,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            ))

        self.W_ih = create_param((input_size, hidden_size * 4))
        self.W_hh = create_param((hidden_size, hidden_size * 4))
        
        self.bias_ih = create_param((hidden_size * 4,)) if bias else None
        self.bias_hh = create_param((hidden_size * 4,)) if bias else None

        self.sigmoid = Sigmoid()
        self.tanh = ops.tanh
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h:
            h0, c0 = h
        else:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)

        x = X @ self.W_ih + h0 @ self.W_hh # (bs, hidden_size * 4)

        if self.bias_ih:
            x += self.bias_ih.broadcast_to((bs, self.hidden_size * 4))
            x += self.bias_hh.broadcast_to((bs, self.hidden_size * 4))

        x_split = tuple(ops.split(x, axis=1)) # [(bs, 1), ...] (hidden_size * 4)
        gates = []
        for i in range(4):
            gates.append(ops.stack(x_split[i * self.hidden_size: (i + 1) * self.hidden_size], axis=1))
        
        i, f, g, o = gates # (bs, hidden_size)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)

        c = f * c0 + i * g
        h = o * self.tanh(c)
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_cells = [
            LSTMCell(
                input_size=input_size if k == 0 else hidden_size, 
                hidden_size=hidden_size, 
                bias=bias, 
                device=device, 
                dtype=dtype,
            )
            for k in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h:
            h0, c0 = h
        else:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)

        X_t = ops.split(X, axis=0) # [(bs, input_size), ...] (seq_len)
        H_n = ops.split(h0, axis=0) # [(bs, hidden_size), ...] (num_layers)
        C_n = ops.split(c0, axis=0) # [(bs, hidden_size), ...] (num_layers)
        H_t = []

        for t in range(seq_len):
            x_t = X_t[t] # (bs, input_size or hidden_size)
            h_n, c_n = [], []

            for l in range(self.num_layers):
                lstm_cell = self.lstm_cells[l]
                h_tl, c_tl = lstm_cell(x_t, (H_n[l], C_n[l])) # (bs, hidden_size)
                x_t = h_tl

                h_n.append(h_tl)
                c_n.append(c_tl)

                if l == self.num_layers - 1:
                    H_t.append(h_tl)
            
            H_n, C_n = h_n, c_n
        
        H_t = ops.stack(H_t, axis=0) # (seq_len, bs, hidden_size)
        H_n = ops.stack(H_n, axis=0) # (num_layer, bs, hidden_size)
        C_n = ops.stack(C_n, axis=0) # # (num_layer, bs, hidden_size)
        return H_t, (H_n, C_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(
            num_embeddings,
            embedding_dim,
            mean=0,
            std=1,
            device=device, 
            dtype=dtype,
            requires_grad=True,
        ))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)  # (seq_len, bs, num_embeddings)
        reshaped_one_hot = one_hot.reshape((seq_len * bs, self.num_embeddings))  # (seq_len * bs, num_embeddings)
        weighted_input = reshaped_one_hot @ self.weight  # (seq_len * bs, embedding_dim)
        output = weighted_input.reshape((seq_len, bs, self.embedding_dim))  # (seq_len, bs, embedding_dim)
        return output
        ### END YOUR SOLUTION