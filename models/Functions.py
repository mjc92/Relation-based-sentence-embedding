import torch
from torch.autograd import Variable

def BatchToData(batch):
    # changes the batch output into Variables which can be applied directly to the model
    simple = batch.simple[0].data.transpose(1,0)
    normal = batch.normal[0].data.transpose(1,0)
    
    # get labels
    labels = torch.cat([torch.zeros(simple.size(0)),torch.ones(normal.size(0))]).long()
    # [batch]
    
    # match the lengths of normal and simple datasets by padding
    diff = simple.size(1)-normal.size(1)
    if diff>0:
        normal = torch.cat([normal,torch.ones(normal.size(0),diff).long()],1)
    elif diff<0:
        simple = torch.cat([simple,torch.ones(simple.size(0),diff*(-1)).long()],1)
    inputs = torch.cat([simple,normal],0) # [batch x max_len_in_batch]
    
    # get lengths of each sequence
    lengths = torch.cat([batch.simple[1],batch.normal[1]])
    # lengths = (1-inputs.eq(1)).long().sum(1) # [batch]
    positions = torch.zeros(inputs.size()).long()
    for i in range(lengths.size(0)):
        positions[i,:lengths[i]] = torch.arange(0,lengths[i]).long()+1
        
    # do random permutation
    perm = torch.randperm(simple.size(0))
    inputs = inputs[perm]
    positions = positions[perm]
    labels = labels[perm]
    
    return Variable(inputs), Variable(positions), Variable(labels)