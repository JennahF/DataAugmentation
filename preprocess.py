import pickle
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import random
from collections import Counter
from utils.dict import Dictionary
import pandas as pd
import config
import numpy as np
import os

en_stops = set(stopwords.words('english'))
nltk.download('wordnet')

dirs = ['./data/aclImdb_v1/aclImdb/train/pos/',
    './data/aclImdb_v1/aclImdb/train/neg/',
    './data/aclImdb_v1/aclImdb/test/pos/',
    './data/aclImdb_v1/aclImdb/test/neg/']

files = [[dir + file for file in os.listdir(dir)] for dir in dirs]
words_save_files = ['./temp/words_train_pos.pickle', './temp/words_train_neg.pickle',
                './temp/words_test_pos.pickle', './temp/words_test_neg.pickle']
dic_path = './temp/dictionary'

dic = Counter()
print("...finding similar samples for all samples...")
for i in range(len(files)):

    if os.path.exists(words_save_files[i]):
        with open(words_save_files[i], 'wb') as f:
            file = pickle.load(f)
        print(file[0:3])
        continue

    all_docs = []
    for file in tqdm(files[i]):
        with open(file, 'r', encoding='utf8') as f:
            review = f.readlines()
        words = review[0].lower().split()

        for j in range(len(words)-1, -1, -1):
            words[j] = PorterStemmer().stem(words[j])
            if words[j] in en_stops or words[j] == '':
                del words[j]
                j += 1
        sub_index = random.randint(0,len(words)-1)
        sub_word = words[sub_index]
        target_word = sub_word

        synsets = wn.synsets(sub_word)
        temp = []
        for syn in synsets:
            temp += syn.lemma_names()
        for syn in temp:
            if syn != sub_word:
                target_word = syn
                break

        words1 = words
        words1[sub_index] = target_word
        all_docs.append([words, words1])
        dic += Counter(words) + Counter(words1)
    with open(words_save_files[i], 'wb') as f:
        pickle.dump(all_docs, f)

if not os.path.exists(dic_path):
    sorted_dic, _ = zip(*dic.most_common())
    word2id = {token: i + 1 for i, token in enumerate(sorted_dic)}
    dictionary = Dictionary(word2id)
    with open(dic_path, 'wb') as f:
        pickle.dump(dictionary, f)


print("...converting words to word embeddings...")
embeddings_save_files = ['./temp/embeddings_train_pos.pickle', './temp/embeddings_train_neg.pickle',
                './temp/embeddings_test_pos.pickle', './temp/embeddings_test_neg.pickle']

glove_path = './data/glove.txt'
glove = pd.read_csv(glove_path, sep='\s+', header=None, index_col=0)
glove = pd.DataFrame(glove)

for i in range(len(embeddings_save_files)):
    
    if os.path.exists(embeddings_save_files[i]):
        continue
    
    with open(words_save_files[i], 'rb') as f:
        all_reviews = pickle.load(f)

    embeddings = [] # n*2*review_lentgh*embedding_dim
    for pairSentences in tqdm(all_reviews):
        embed = []
        for review in pairSentences:
            e = []
            review = review[0:config.parameters.review_length]
            for word in review:
                if word in glove.index:
                    e.append(list(glove.loc[word])[:config.parameters.embedding_dim])
                else:
                    e.append(np.random.rand(config.parameters.embedding_dim).tolist())
            if len(e) < config.parameters.review_length:
                for i in range(len(e), config.parameters.review_length):
                    e.append(np.random.rand(config.parameters.embedding_dim).tolist())
            embed.append(e)
        embeddings.append(embed)

    with open(embeddings_save_files[i], 'wb') as f:
        pickle.dump(embeddings, f)
