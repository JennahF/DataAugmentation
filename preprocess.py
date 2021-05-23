import pickle
from tqdm import tqdm
import pandas as pd
import config
import numpy as np
import os
import gensim
from utils.read import read_IMDB, read_Subj, read_TREC, read_PC, read_CoLA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--subsetportion', type=float, default=1)
parser.add_argument('--posNum', type=int, default=1)
args = parser.parse_args()

save_files1 = ['./temp/word2vec/'+args.dataset+"_model1_"+str(args.posNum)+".pickle",
                './temp/word2vec/'+args.dataset+"_model2_original.pickle",
                './temp/word2vec/'+args.dataset+"_test.pickle"]

def to_files(model, all_reviews, labels, posNum):
    '''
        all_reviews:[[train_pos + train_pos_aug + train_neg + train_neg_aug],
                     [test_pos + test_pos_aug + test_neg + test_neg_aug]]
        labels:     [[1(*len(train_pos)) + 0(*len(train_neg))],
                    [1(*len(test_pos)) + 0(*len(test_neg))]]
    '''
    switch = [0,0,1]

    for num in range(len(save_files1)):
        
        if os.path.exists(save_files1[num]):
            continue
        
        print("start generating file:", save_files1[num])

        # pos
        print("set len=", len(all_reviews[switch[num]]))
        print("label len=", len(all_reviews[switch[num]]))
        embeddings = [] # n*(1+posNum)*review_lentgh*embedding_dim
        for i in tqdm(range(0, len(all_reviews[switch[num]]), posNum+1)):
            if num != 0:
                pairSentences = [all_reviews[switch[num]][i]]
            else:
                pairSentences = all_reviews[switch[num]][i:i+posNum+1]

            embed = []
            for j, review in enumerate(pairSentences):
                e = []
                for word in review:
                    try:
                        e.append(model.wv[word])
                    except:
                        print(review, word)
                embed.append(e)
            embeddings.append(embed)
        label = labels[switch[num]]

        print("two lens:",len(embeddings), len(label))
        out = [embeddings, label]
        with open(save_files1[num], 'wb') as f:
            pickle.dump(out, f)
        print("dim=(",len(embeddings),len(embeddings[0]), len(embeddings[0][0]), len(embeddings[0][0][0]),")")

all_docs = []
words_save_file = './temp/word2vec/train_test_document.pickle'
if not os.path.exists(words_save_file):
    if args.dataset == 'IMDB':
        all_docs, labels = read_IMDB(args.subsetportion, args.posNum)
    elif args.dataset == 'Subj':
        all_docs, labels = read_Subj(args.subsetportion, args.posNum)
    elif args.dataset == 'TREC':
        all_docs, labels = read_TREC(args.subsetportion, args.posNum)
    elif args.dataset == 'PC':
        all_docs, labels = read_PC(args.subsetportion, args.posNum)
    elif args.dataset == 'CoLA':
        all_docs, labels = read_CoLA(args.subsetportion, args.posNum)


# traing word2vec model
print("size = ",config.parameters.embedding_dim)
model_path = './temp/word2vec/word2vec.pickle'
if not os.path.exists(model_path):
    if all_docs == []:
        with open(words_save_file, 'rb') as f:
            docs_labels = pickle.load(f)
        all_docs = docs_labels[0]
        labels = docs_labels[1]
    print(len(all_docs[0])+len(all_docs[1]))
    model = gensim.models.Word2Vec(
        all_docs[0]+all_docs[1],
        size = config.parameters.embedding_dim,
        window = 10,
        min_count = 0,
        workers = 10,
        iter = 20
    )
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

# generate test dataset for testing TCNN network or generate train dataset for training Seq2Seq network
print("...converting words to word embeddings...")

if all_docs != []:
    all_reviews = all_docs
else:
    with open(words_save_file, 'rb') as f:
        docs = pickle.load(f)
    all_reviews = docs[0]
    labels = docs[1]

# print(all_reviews[0])
# print(all_reviews[1])
# print(labels[0])
# print(labels[1])

to_files(model, all_reviews, labels, args.posNum)
