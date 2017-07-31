import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init


def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, dim=-1)
        sigma = torch.std(z, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = d_model ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = F.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s,
                                        attn_mask=attn_mask.repeat(n_head, 1, 1) if attn_mask is not None else None)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        print(outputs.size())

        # project back to residual size
        outputs = self.proj(outputs.view(mb_size, -1))
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns


class RelationsNetwork(nn.Module):
    def __init__(self, batch_size, max_sentence_len, embed_dim, cuda):
        super(RelationsNetwork, self).__init__()

        # (max sentence length + x, y coordinates) * 2 + question vector

        self.g = nn.Sequential(
            nn.Linear((embed_dim + 2) * 2 + embed_dim, 256),
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

    def forward(self, sentences, condition):
        batch_size, seq_length, embed_dim = sentences.size()

        # Append coordinates.

        sentences = torch.cat([sentences, self.coordinate_map], dim=2)

        # Prepare condition (question vector).

        condition = torch.unsqueeze(condition, dim=1)
        condition = condition.repeat(1, seq_length, 1)
        condition = torch.unsqueeze(condition, dim=2)
        condition = condition.repeat(1, 1, seq_length, 1)

        # Build relations map by permutating words `i` and `j`.

        object_i = torch.unsqueeze(sentences, dim=1)
        object_i = object_i.repeat(1, seq_length, 1, 1)

        object_j = torch.unsqueeze(sentences, dim=2)
        object_j = object_j.repeat(1, 1, seq_length, 1)

        # Concatenate and build relations map.

        relations_map = torch.cat([object_i, object_j, condition], 3)

        # Reshape for passing it through g(x).

        relations_map = relations_map.view(batch_size * seq_length * seq_length,
                                           (embed_dim + 2) * 2 + embed_dim)
        relations_map = self.g(relations_map)

        # Reshape yet again and sum.

        relations_map = relations_map.view(batch_size, seq_length * seq_length, 256)
        relations_map = relations_map.sum(1).squeeze()

        # Reshape for passing it through f(x).
        relations_map = self.f(relations_map)

        return F.log_softmax(relations_map)


class AttentiveRelationsNetwork(nn.Module):
    def __init__(self, batch_size, max_sentence_len, embed_dim, cuda):
        super(AttentiveRelationsNetwork, self).__init__()

        # (max sentence length + x, y coordinates) * 2 + question vector

        self.g = nn.Sequential(
            nn.Linear((embed_dim + 2) * 2 + embed_dim, 256),
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

        self.attention = MultiHeadAttention(n_head=8, d_model=embed_dim + 2, d_k=64, d_v=64)

    def forward(self, sentences, condition):
        batch_size, seq_length, embed_dim = sentences.size()

        # Append coordinates.

        sentences = torch.cat([sentences, self.coordinate_map], dim=2)

        # Prepare condition (question vector).

        condition = torch.unsqueeze(condition, dim=1)
        condition = condition.repeat(1, seq_length, 1)
        condition = torch.unsqueeze(condition, dim=2)
        condition = condition.repeat(1, 1, seq_length, 1)

        # Build relations map by permutating words `i` and `j`.

        object_i = torch.unsqueeze(sentences, dim=1)
        object_i = object_i.repeat(1, seq_length, 1, 1)

        object_j = torch.unsqueeze(sentences, dim=2)
        object_j = object_j.repeat(1, 1, seq_length, 1)

        # Apply multi-head attention here.

        test = self.attention(sentences, sentences, sentences)

        # Concatenate and build relations map.

        relations_map = torch.cat([object_i, object_j, condition], 3)

        # Reshape for passing it through g(x).

        relations_map = relations_map.view(batch_size * seq_length * seq_length,
                                           (embed_dim + 2) * 2 + embed_dim)

        relations_map = self.g(relations_map)

        # Reshape yet again and sum.

        relations_map = relations_map.view(batch_size, seq_length * seq_length, 256)
        relations_map = relations_map.sum(1).squeeze()

        # Reshape for passing it through f(x).
        relations_map = self.f(relations_map)

        return F.log_softmax(relations_map)
