import torch
import torch.nn as nn

class SharedCodebook(nn.Module):
    def __init__(self, codebook=None, num = 128, dim = 128):
        super(SharedCodebook, self).__init__()
        if codebook is None:
            n_embeddings = num 
            embedding_dim = dim  
            self.codebook = nn.Embedding(n_embeddings, embedding_dim)
            self.codebook.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)
        else:
            self.codebook = codebook

    def forward(self):
        return self.codebook