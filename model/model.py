import numpy as np
from numpy import random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

def sort_batch(data, lens):
    # print("in sort_batch:")
    # print("data.size=",data.size())
    sorted_lens, sorted_idx = torch.sort(lens, dim=0, descending=True)
    sorted_data = data[sorted_idx.data]
    _, recover_idx = torch.sort(sorted_idx, dim=0, descending=False)
    return (sorted_data, sorted_lens), recover_idx

# Encoder
class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers):
        '''

        :param input_dim: 输入源词库的大小
        :param emb_dim:  输入单词Embedding的维度
        :param hid_dim: 隐层的维度
        :param n_layers: 几个隐层
        :param dropout:  dropout参数 0.5
        '''

        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers = n_layers, batch_first=True)
        

    def forward(self, embedded):
        # embedded: [batch_size*(negNum+2)*max_len, embed_len]
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.transpose(hidden, 0,1)
        cell = torch.transpose(cell, 0,1)

        # src sen len, batch size, hid dim, n directions, n layers
        # outputs: [batch size, src sent len, hid dim * n directions]
        # hidden, cell: [batch size, n layers* n directions, hid dim]
        # outputs are always from the top hidden layer

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,  batch_first=True)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, hidden, cell):

        # print("before decoding, hidden.size()=",hidden.size(), "cell.size()=",cell.size())
        output, (hidden, cell) = self.rnn(hidden, (torch.transpose(hidden, 0,1), torch.transpose(cell, 0,1)))

        # output = [batch size, 1, hid dim]
        # hidden = [batch size, n layers, hid dim]
        # cell = [batch size, n layers, hid dim]

        prediction = self.out(self.dropout(output.squeeze(0)))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell        

class Seq2Seq(nn.Module):
    def __init__(self, para, device):
        super().__init__()

        self.encoder = Encoder(para.ENC_EMB_DIM, para.HID_DIM, para.N_LAYERS)
        self.decoder = Decoder(para.OUTPUT_DIM, para.DEC_EMB_DIM, para.HID_DIM, para.N_LAYERS, para.DEC_DROPOUT)
        self.dropout = nn.Dropout(para.ENC_DROPOUT)
        self.device = device

    # src: [batch_size*(negNum+2), max_len, embed_len]
    def forward(self, src, src_lens):
        src=src.float()
        old_size = list(src.size())

        batch_size = src.size()[0]
        max_len = src.size()[1]
        output_dim = src.size()[2]
        
        # sorted_src: [batch_size*(negNum+2), max_len, embed_len]
        (sorted_src, sorted_lens), recover_idx = sort_batch(self.dropout(src), src_lens)
        # sorted_src: [efficient_word_num, embed_len]
        packed_src = pack(sorted_src, list(sorted_lens.data), batch_first=True)
        print(packed_src.data.size())
        
        if self.device == 'gpu':
        # tensor to store decoder outputs
            outputs = torch.zeros(max_len, batch_size,
                              output_dim).cuda()
        else:
            outputs = torch.zeros(max_len, batch_size,
                              output_dim)

        hidden, cell = self.encoder.forward(packed_src)
        hidden = hidden[recover_idx.data]
        cell = cell[recover_idx.data]

        for t in range(1, max_len):
            # hidden, cell: [batch size, 1, hid dim]
            output, hidden, cell = self.decoder.forward(hidden, cell)

            # print("after decoding, output.size()=",output.size(),"outputs.size()=",outputs.size())
            outputs[t] = output.squeeze(1)
            hidden = torch.transpose(hidden,0,1)
            cell = torch.transpose(cell, 0,1)
        
        outputs = outputs.transpose(0,1)

        return outputs


class parameters():
    batch_size = 64
    embedding_dim = 100
    ENC_EMB_DIM = 300
    HID_DIM = 256
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    OUTPUT_DIM = 300
    DEC_EMB_DIM = 256
    DEC_DROPOUT = 0.5
para = parameters()

s2s = Seq2Seq(para, 'cpu')

print(s2s)

data = torch.zeros(64*5,500,300)
lengths = torch.linspace(1, 500, steps=64*5)
for i in range(len(lengths)):
    lengths[i] = int(lengths[i])
output = s2s.forward(data, lengths)
print(output.size())