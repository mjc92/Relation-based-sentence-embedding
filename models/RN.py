import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class RelationsNetwork(nn.Module):
    def __init__(self, batch_size, n_max_seq, d_model, out_classes, d_hidden=256):
        super(RelationsNetwork, self).__init__()
        
        self.hidden = d_hidden

        # (max sentence length + x, y coordinates) * 2 + question vector
        self.g = nn.Sequential(
            nn.Linear((d_model + 2) * 3, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_hidden, out_classes)
        )

        # Prepare coordinates.
        n_max_seq += 1
        self.coordinate_map = torch.zeros(batch_size, n_max_seq, 2)

        for i in range(n_max_seq):
            self.coordinate_map[:, i, 0] = (i / n_max_seq ** 0.5 - 2) / 2.
            self.coordinate_map[:, i, 1] = (i % n_max_seq ** 0.5 - 2) / 2.

        # if cuda:
        self.coordinate_map = self.coordinate_map.cuda()

        self.coordinate_map = Variable(self.coordinate_map)

    def forward(self, sentences, condition=None):
        batch_size, seq_length, d_model = sentences.size()

        # Append coordinates.
        sentences = torch.cat([sentences,self.coordinate_map[:batch_size,:seq_length,:]], 
                                          dim=2)

        # Build relations map by permutating words `i` and `j`.

        object_i = torch.unsqueeze(sentences.clone(), dim=1) # [batch x 1 x seq x d_model+2]
        object_i = object_i.repeat(1, seq_length, 1, 1) # [batch x seq x seq x d_model+2]

        object_j = torch.unsqueeze(sentences.clone(), dim=2) # [batch x seq x 1 x d_model+2]
        object_j = object_j.repeat(1, 1, seq_length, 1) # [batch x seq x seq x d_model+2]

        # Prepare condition (question vector).
        condition = object_i * object_j # [batch x seq x seq x d_model+2]
        relations_map = torch.cat([object_i, object_j, condition], 3)
        # [batch x seq x seq x d_model*3+6]
        # Reshape for passing it through g(x).

        relations_map = relations_map.view(batch_size * seq_length * seq_length, -1)
    
        relations_map = self.g(relations_map)
        # [batch*seq*seq x 256]

        # Reshape yet again and sum.

        relations_map = relations_map.view(batch_size, seq_length * seq_length, self.hidden)
        relations_map = relations_map.sum(1).squeeze() # [batch x 256]

        # Reshape for passing it through f(x).
        relations_map = self.f(relations_map) # [batch x class_out(2)]

        return relations_map