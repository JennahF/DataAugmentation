from utils.read import read_IMDB, read_Subj, read_TREC, read_PC, read_CoLA
from collections import Counter
import torch
from model.TextCNN import TextCnn
from model.BiLSTM import TextRNN
from config import Config

def count(labels):
    print("train:",Counter(labels[0]))

    print("test:",Counter(labels[1]))

# all_docs, all_labels = read_IMDB(1)
# print("IMDB")
# count(all_labels)

# all_docs, all_labels = read_Subj(1)
# print("Subj")
# count(all_labels)

# all_docs, all_labels = read_TREC(1)
# print("TREC")
# count(all_labels)

# all_docs, all_labels = read_PC(1)
# print("PC")
# count(all_labels)

all_docs, all_labels = read_CoLA(1,0)
print("PC")
count(all_labels)

# checkpoint = torch.load("./model/trained_model/1IMDB_22/model_0.pt", map_location='cuda:0')
# print(checkpoint['para'].__dict__)

# checkpoint = torch.load("./model/trained_model/2Subj_25/model_1.pt", map_location='cuda:0')
# print(checkpoint['para'])

# checkpoint = torch.load("./model/trained_model/3TREC/model_0.pt", map_location='cuda:0')
# print(checkpoint['para'])