import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init

if __name__ == "__main__":
    from models import RelationsNetwork, AttentiveRelationsNetwork, MultiHeadAttention

    batch_size, max_sentence_len, embed_dim = 2, 25, 300
    model = RelationsNetwork(batch_size=batch_size, max_sentence_len=max_sentence_len, embed_dim=embed_dim,
                                      cuda=False)

    sentences = Variable(torch.randn(batch_size, max_sentence_len, embed_dim))
    condition = Variable(torch.randn(batch_size, embed_dim))

    model(sentences, condition)

    test_attention = MultiHeadAttention(n_head=8, d_model=embed_dim, d_k=64, d_v=64)
