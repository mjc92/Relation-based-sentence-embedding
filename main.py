import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init

if __name__ == "__main__":
    from models.Models import AttentiveRelationsNetwork

    batch_size, max_sentence_len, embed_dim, vocab = 2, 25, 300, 100
    model = AttentiveRelationsNetwork(vocab,vocab,max_sentence_len,batch_size)
    
#     # inputs for Attentional Relations network
    batch_size, seq_length, vocab = 2, 25, 100
    inputs=np.random.randint(3,vocab, batch_size*seq_length,
                  dtype=int).reshape([-1,seq_length])
    
    len_list = []
    for i,line in enumerate(inputs):
        inputs[i,0]=1
        length = np.random.randint(10,max_sentence_len)
        len_list.append(length)
        inputs[i,length]=2
        inputs[i,length+1:]=0
    positions = np.zeros(inputs.shape,dtype=int)
    for i in range(positions.shape[0]):
        length = len_list[i]
        positions[i,:length] = np.arange(length)+1
    
    inputs = Variable(torch.LongTensor(inputs))
    positions = Variable(torch.LongTensor(positions))
    outputs = model((inputs,positions))
    print(outputs)
    