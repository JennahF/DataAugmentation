import torch
from torch import nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F

def sort_batch(data, lens):
    # print("in sort_batch:")
    # print("data.size=",data.size())
    sorted_lens, sorted_idx = torch.sort(lens, dim=0, descending=True)
    sorted_data = data[sorted_idx.data]
    _, recover_idx = torch.sort(sorted_idx, dim=0, descending=False)
    return (sorted_data, sorted_lens), recover_idx

class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.config = config
        
        self.lstm = nn.LSTM(input_size = self.config.word_embedding_dimension,
                            hidden_size = self.config.hidden_size,
                            num_layers = self.config.hidden_layers,
                            bidirectional = self.config.bidirectional,
                            batch_first=True)
        
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * self.config.hidden_layers * (1+self.config.bidirectional),
            self.config.label_num
        )
        
        # Softmax non-linearity
        
    def forward(self, x, len):
        x = x.squeeze(1)
        # x.shape = (batch_size, max_sen_len, embed_size)

        # (sorted_src, sorted_lens), recover_idx = sort_batch(x.float(), len)
        # packed_src = pack(sorted_src, list(sorted_lens.data), batch_first=True)

        _, (h_n,c_n) = self.lstm(x)
        # print(lstm_out, h_n.size(), c_n.size())
        final_feature_map = self.dropout(h_n) # shape=(num_layers * num_directions, 64, hidden_size)
        
        # Convert input to (64, hidden_size * hidden_layers * num_directions) for linear layer
        final_feature_map = torch.cat([final_feature_map[i,:,:] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        return F.log_softmax(final_out, dim=1)

# class parameters():
#     batch_size = 64
#     word_embedding_dimension = 300
#     hidden_size = 256
#     hidden_layers = 1
#     dropout = 0.1
#     bidirectional = True
#     label_num = 6

# para = parameters()

# s2s = TextRNN(para)

# print(s2s)

# data = torch.zeros(64*5,500,300)
# lengths = torch.linspace(1, 500, steps=64*5)
# for i in range(len(lengths)):
#     lengths[i] = int(lengths[i])
# output = s2s.forward(data, lengths)
# print(output.size())