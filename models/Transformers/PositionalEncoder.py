import numpy as np
import torch.nn as nn
import torch.zeros as zeros


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_size, max_sequence_length=128, dropout=0.0):
        super(PositionalEncoder, self).__init__()

        # dropout
        self.dropout = nn.Dropout(dropout)

        # positional encoding
        pe = zeros(max_sequence_length, embedding_size)
        for position in range(max_sequence_length):
            for i in range(0, embedding_size, 2):
                theta = position / (10000 ** ((2 * i) / embedding_size))
                pe[position, i] = np.sin(theta)
                pe[position, i + 1] = np.cos(theta)

        # pack it together (add dimension)
        pe = pe.unsqueeze(0)

        # the positional encoding should not be trained by
        # the optimizer, but should be saved to the dict for
        # loading/reloading, so register buffer is used here
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add positional encoding to input embeddings (x)
        x = x + self.pe[:, :x.size(1)]  # shape (1, seq_len, embedding_size)
        return self.dropout(x)
