import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class RelationsNetwork(nn.Module):
    def __init__(self, batch_size, max_sentence_len, embed_dim, cuda):
        super(RelationsNetwork, self).__init__()

        # (max sentence length + x, y coordinates) * 2 + question vector

        self.g = nn.Sequential(
            nn.Linear((embed_dim + 2) * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(256, 10)
        )

        # Prepare coordinates.
        self.coordinate_map = torch.zeros(batch_size, max_sentence_len, 2)

        for i in range(max_sentence_len):
            self.coordinate_map[:, i, 0] = (i / max_sentence_len ** 0.5 - 2) / 2.
            self.coordinate_map[:, i, 1] = (i % max_sentence_len ** 0.5 - 2) / 2.

        if cuda:
            self.coordinate_map = self.coordinate_map.cuda()

        self.coordinate_map = Variable(self.coordinate_map)

    def forward(self, sentences, condition=None):
        batch_size, seq_length, embed_dim = sentences.size()

        # Append coordinates.
        sentences = torch.cat([sentences, self.coordinate_map], dim=2)

        # Build relations map by permutating words `i` and `j`.

        object_i = torch.unsqueeze(sentences, dim=1)
        object_i = object_i.repeat(1, seq_length, 1, 1)

        object_j = torch.unsqueeze(sentences, dim=2)
        object_j = object_j.repeat(1, 1, seq_length, 1)

        # Prepare condition (question vector).
        if condition!=None:
            condition = torch.unsqueeze(condition, dim=1)
            condition = condition.repeat(1, seq_length, 1)
            condition = torch.unsqueeze(condition, dim=2)
            condition = condition.repeat(1, 1, seq_length, 1)
            relations_map = torch.cat([object_i, object_j, condition], 3)
        else:
            relations_map = torch.cat([object_i, object_j], 3)
        # Concatenate and build relations map.


        # Reshape for passing it through g(x).

        relations_map = relations_map.view(batch_size * seq_length * seq_length, -1)
        relations_map = self.g(relations_map)

        # Reshape yet again and sum.

        relations_map = relations_map.view(batch_size, seq_length * seq_length, 256)
        relations_map = relations_map.sum(1).squeeze()

        # Reshape for passing it through f(x).
        relations_map = self.f(relations_map)

        return F.log_softmax(relations_map)