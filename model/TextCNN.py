# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class TextCnn(nn.Module):
    def __init__(self, config, device):
        super(TextCnn, self).__init__()

        Ci = 1
        Co = len(config.kernel_sizes)
        self.device = device
        self.kernel_sizes = config.kernel_sizes

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, config.word_embedding_dimension), padding = (2, 0)) for f in config.kernel_sizes])

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(Co * len(config.kernel_sizes), config.label_num)

    def forward(self, x, lens=[]):
        x=x.float()
        # (N, Ci, token_num, embed_dim)
        # (N)
        # print("1:", x.size())
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        # print("2:", x[0].size(), x[1].size(), x[2].size())
        # print(x)

        # if self.device == 'cpu':
        #     w = [(torch.zeros(temp.size())) for temp in x]
        # else:
        #     w = [(torch.zeros(temp.size())).cuda() for temp in x]
        # for i in range(len(w)):
        #     ks = self.kernel_sizes[i]
        #     wi = w[i].transpose(1,2)
        #     for j in range(len(wi)):
        #         wi[j][:int(lens[j].item()+5-ks)]=1
        #     x[i] = x[i]*w[i]

        # print(x)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        # print(x)
        # print("3:", x[0].si ze(), x[1].size(), x[2].size())
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        # print("4:", x.size())
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        out = logit
        
        return out

# class Config(object):
#     def __init__(self, word_embedding_dimension=300,
#                  epoch=2, sentence_max_size=500, cuda=False,
#                  label_num=2, learning_rate=0.01, batch_size=64,
#                  out_channel=100):
#         self.word_embedding_dimension = word_embedding_dimension     # 词向量的维度
#         self.epoch = epoch                                           # 遍历样本次数
#         self.sentence_max_size = sentence_max_size                   # 句子长度
#         self.label_num = label_num                                   # 分类标签个数
#         self.lr = learning_rate
#         self.batch_size = batch_size
#         self.out_channel=out_channel
#         self.cuda = cuda
#         self.kernel_sizes = [3,4,5]
#         self.dropout = 0.5


# if __name__ == '__main__':
#     print('running the TextCNN...')
#     config = Config(sentence_max_size=500,
#                 batch_size=64,
#                 label_num=2)
#     tCNN = TextCnn(config, 'cpu')
#     input_data = torch.zeros([64,1,500,300])
#     lens = torch.linspace(1,499,64)
#     for i in range(len(lens)):
#         lens[i] = int(lens[i])
#     print(lens)
#     output_data = tCNN(input_data, lens)
#     print("input data size:", input_data.size())
#     print("output data size:", output_data.size())
