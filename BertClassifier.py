import torch
import torch.nn as nn
import torch.nn.functional as F
from PositionalEncoding import PositionalEncoding
import math

class Bertclassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embed_drop, nhead, nlayers, num_class):
        super(Bertclassifier, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed_drop = embed_drop
        self.nhead = nhead
        self.nlayers = nlayers
        self.Y = num_class
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position = PositionalEncoding(self.embedding_dim, self.embed_drop)
        self.encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.nlayers)
        self.fc = nn.Linear(self.embedding_dim, self.Y)
        
    def forward(self, x, src_padding = None):

        # B: Batch Size
        # N: Text Length
        # D: Embedding Dimension
        # text_input: [B, N+1, D]

        x = self.embedding(x)
        x = self.position(x)
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        # src_key_padding_mask=src_padding.cuda()
        # [B, N+1, D]
        x = x[:, 0, :]
        # [B, D]
        x = self.fc(x)
        # [B, Y]
        return x


