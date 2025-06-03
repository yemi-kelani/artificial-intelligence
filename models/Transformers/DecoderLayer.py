import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # multi-head attention, dropout
        self.multihead_attention = nn.MultiheadAttention(
            embedding_size, num_heads)
        self.masked_multihead_attention = nn.MultiheadAttention(embedding_size,
                                                                num_heads)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        # layer normalization
        self.norm_layer_1 = nn.LayerNorm(embedding_size)
        self.norm_layer_2 = nn.LayerNorm(embedding_size)
        self.norm_layer_3 = nn.LayerNorm(embedding_size)

        # feed forward specified by paper
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )

    def forward(self, x, encoder_output, encoder_mask=None, decoder_mask=None):
        attn_output, _ = self.multihead_attention(
            x, x, x, attn_mask=decoder_mask)
        out = x + self.dropout_1(attn_output)  # residual connection
        out = self.norm_layer_1(out)

        attn_output, _ = self.masked_multihead_attention(
            out, encoder_output, encoder_output, attn_mask=encoder_mask)
        out = out + self.dropout_2(attn_output)  # residual connection
        out = self.norm_layer_2(out)

        out = self.feedforward(out)
        out = out + self.dropout_3(out)
        out = self.norm_layer_3(out)

        return out
