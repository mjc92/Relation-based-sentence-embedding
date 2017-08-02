''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import models.Constants as Constants
from models.Modules import BottleLinear as Linear
from models.Layers import EncoderLayer
from models.RN import RelationsNetwork
import time

__author__ = "Yu-Hsiang Huang"

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''
    # n_pos: max_seq + 1, so the position encoding is based on the max sequence length
    # sequences shorter than this will be given less positional changes
    # d_pos_vec: d_word_vec = 512
    
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    
    # resulting position encoder: [n_pos x d_pos_vec]
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        # this Embedding seemingly has vocab size of n_position
        # and embedding dimension of d_word_vec
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):
        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)
        # enc_input: [b x max_len_seq x d_word_vec]
        
        # Position Encoding addition
        enc_input += self.position_enc(src_pos) # [b x max_len_seq x d_word_vec]
        enc_outputs, enc_slf_attns = [], []

        enc_output = enc_input
        
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq) # [b x max_len x max_len]
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]
        
 
        # returns a list of encoder outputs and encoder self attentions
        return enc_outputs, enc_slf_attns
    
class AttentiveRelationsNetwork(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, out_classes, batch_size=64,
            n_layers=6, n_head=8, d_word_vec=512, d_model=512, 
            d_inner_hid=1024, d_k=64, d_v=64, dropout=0.1, 
            proj_share_weight=True, embs_share_weight=True):

        super(AttentiveRelationsNetwork, self).__init__()
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        
        self.relations = RelationsNetwork(
            batch_size, n_max_seq, d_model, out_classes, d_hidden=128)
  
        self.dropout = nn.Dropout(dropout)
    
    
    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src):
        src_seq, src_pos = src
        enc_outputs, enc_slf_attns = self.encoder(src_seq, src_pos)
        # enc_outputs[-1]: [mb x seq_len x d_model]
        # we could either try averaging all attentions or using the last one
        # outputs = self.relations(enc_outputs[-1])
        outputs = enc_outputs[-1].view(enc_outputs[-1].size(0),-1)[:,:2]
        return outputs