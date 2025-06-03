import torch.nn as nn
import PositionalEncoder
import EncoderLayer
import DecoderLayer


class Transformer(nn.Module):
    def __init__(self, ec_vocab_size, dc_vocab_size, output_size, embedding_size,
                 hidden_size, num_heads=1, max_sequence_length=128, dropout=0.1,
                 num_layers=1):
        super(Transformer, self).__init__()
        self.embedding_encoder = nn.Embedding(ec_vocab_size, embedding_size)
        self.embedding_decoder = nn.Embedding(dc_vocab_size, embedding_size)

        # positionally encode with custom classes
        self.pe_encoder = PositionalEncoder(
            embedding_size, max_sequence_length)
        self.pe_decoder = PositionalEncoder(
            embedding_size, max_sequence_length)

        # init encoder, decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embedding_size, hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_size, hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # linear layer, no softmax
        self.linear = nn.Linear(embedding_size, output_size)

    def forward(self, encoder_input, decoder_input, encoder_mask=None,
                decoder_mask=None):
        encoder_input = self.embedding_encoder(encoder_input)
        encoder_input = self.pe_encoder(encoder_input)

        decoder_input = self.embedding_decoder(decoder_input)
        decoder_input = self.pe_decoder(decoder_input)

        # process inputs through layers
        encoder_output = encoder_input
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, encoder_mask)

        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output,
                                   encoder_mask, decoder_mask)

        logits = self.linear(decoder_output)
        return logits
