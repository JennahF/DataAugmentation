import random
import torch
import math
import os
from torch.utils import data
import pickle
from collections import Counter

def create_masks(lengths):
    mask = torch.zeros(len(lengths), max(lengths))
    for i in range(len(lengths)):
        mask[i][:lengths[i]] = 1
    return mask

class myDataloader():
    def __init__(self, trainset, labels, batch_size, negSampleNum, posSampleNum):
        self.trainset = trainset
        self.batch_size = batch_size
        self.term_num = len(trainset) #n*2*review_lentgh*embedding_dim
        self.negSampleNum = negSampleNum
        self.posSampleNum = posSampleNum
        self.batch_number = math.ceil(self.term_num/self.batch_size)
        self.labels = labels
        print("dataset len:{}, labels len:{}".format(len(self.trainset), len(self.labels)))
    
    def _batch(self, data, output_lengths=True):
        lengths = torch.tensor([len(i) for i in data])
        out = torch.zeros(len(data), max(lengths), len(data[0][0]))
        for i in range(len(data)):
            for j in range(lengths[i]):
                out[i][j] = torch.FloatTensor(data[i][j])
        if output_lengths:
            return out.long(), lengths
        else:
            return out.long()

    def __getitem__(self, index):
        start_index = index*self.batch_size
        end_index = (index+1)*self.batch_size if (index+1)*self.batch_size < self.term_num else self.term_num
        subtrainset = self.trainset[start_index:end_index]

        batched_set = []
        for i, pair_sample in enumerate(subtrainset):
            temp = pair_sample
            while len(temp) < self.negSampleNum+self.posSampleNum+1:
                # index = random.randint(0, len(subtrainset)-1)
                try:
                    index1 = random.randint(0,len(self.labels)-1)
                    if index1 != start_index+i and self.labels[index1]!=self.labels[start_index+i]:
                    # if index1 != start_index+i:
                        temp.append(self.trainset[index1][0])
                except:
                    print("total:{},index1:{},start_index+i:{}".format(len(self.labels), index1, start_index+i))
            batched_set += (temp)
        
        # [batch_size*(negNum+2), max_len, embed_len]
        batch, batch_length = self._batch(batched_set)
        # batch_masks = create_masks(batch_length)

        return torch.LongTensor(batch).unsqueeze(1), torch.LongTensor(batch_length)

class TextDataset(data.Dataset):

    def __init__(self, train_embeddings, batch_size, original_dataset, subset):
        
        self.original_dataset = original_dataset
        if original_dataset:
            with open(train_embeddings, 'rb') as f:
                trainset = pickle.load(f)
            self.labels = trainset[1]
            print("train:",Counter(self.labels))
            trainset = trainset[0]
            self.trainset, self.labels = self.shuffle(trainset, self.labels) # n*1*review_lentgh*embedding_dim
            self.trainset = self.trainset[:int(subset*len(self.trainset))]
            self.labels = self.labels[:int((subset*len(self.labels)))]
            self.lens = []
            print("dataset len:{}, labels len:{}".format(len(self.trainset), len(self.labels)))
        else:
            with open(train_embeddings, 'rb') as f:
                trainset = pickle.load(f) #n*1*sentence_length*embedding_dim
            self.labels = trainset[2]
            self.trainset, self.labels, self.lens = self.shuffle(trainset[0], self.labels, trainset[1])
            self.trainset = self.trainset[:int(len(self.trainset)*subset)]
            self.lens = self.lens[:int(len(self.lens)*subset)]
            self.labels = self.labels[:int(len(self.labels)*subset)]

        self.batch_size = batch_size
        self.term_num = len(self.trainset)
        self.batch_num = math.ceil(self.term_num/self.batch_size)
        print("term num=", self.term_num)
    
    def shuffle(self, data, labels, lens=[]):
        if self.original_dataset:
            new_index = torch.randperm(len(data))
            data1 = [data[i] for i in new_index]
            labels1 = [labels[i] for i in new_index]
            return data1, labels1
        else:
            new_index = torch.randperm(data.shape[0])
            data1 = data[new_index].view(data.size())
            lens1 = lens[new_index].view(lens.size())
            labels1 = [labels[i] for i in new_index]
            return data1, labels1, lens1

    def _batch(self, data, output_lengths=True):
        lengths = torch.tensor([len(i[0]) for i in data])
        out = torch.zeros(len(data), max(lengths), len(data[0][0][0]))
        for i in range(len(data)):
            for j in range(lengths[i]):
                out[i][j] = torch.FloatTensor(data[i][0][j])
        if output_lengths:
            return out.unsqueeze(1).float(), lengths
        else:
            return out.unsqueeze(1).float()

    def __getitem__(self, index):
        start_index = index*self.batch_size
        end_index = (index+1)*self.batch_size if (index+1)*self.batch_size < self.term_num else self.term_num
        subtrainset = self.trainset[start_index:end_index] 
        
        if self.original_dataset:
            batched_set, batch_lengths = self._batch(subtrainset)
        else:
            batched_set = self.trainset[start_index:end_index]
            batch_lengths = self.lens[start_index:end_index]
        
        labels = torch.LongTensor(self.labels[start_index:end_index])
        # print("dataloader: (start, end)=({},{})".format(start_index, end_index))
        # print(labels)
        return batched_set, batch_lengths, labels

    def __len__(self):
        return self.term_num
