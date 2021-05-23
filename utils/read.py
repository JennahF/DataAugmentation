import enum
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import os
import re
import random
from collections import Counter
import copy


def remove_sig(str: str):
    '''remove_sig, remove signals from the input string
    Args:
        str: the input string

    Returns:
        A string without signals like .'", etc
    '''
    return re.sub("[+\.\!\/<>“”''"
                  ",$?\-%^*():+\"\']+|[+——！，。？、~#￥%……&*（）]+", "", str.strip())

def substitude_word(word, posNum, test=0, IMDB=0):
    en_stops = set(stopwords.words('english'))
    words = re.split(' |/', word.strip('\n').lower())

     # 去停用词、词根化
    if IMDB:
        for j in range(len(words)-1, -1, -1):
            words[j] = PorterStemmer().stem(remove_sig(words[j]))
            words[j] = remove_sig(words[j])
            if words[j] in en_stops or words[j] == '':
                del words[j]
                j += 1
    else:
        for j in range(len(words)-1, -1, -1):
            words[j] = remove_sig(words[j])
            if words[j] == '':
                del words[j]
                j += 1

    # 替换一个词

    data = []
    data.append(words)

    if test:
        for _ in range(posNum):
            data.append([])
        return data

    # print(words)
    sub_dict = {w:[] for w in words}
    count = 0
    total_iter_times = len(words)+20
    iter_times=0
    while count<posNum:
        iter_times+=1
        if iter_times >= total_iter_times:
            # print("?")
            return [[], []*posNum]
        # print("***{}***".format(count))
        try:
            sub_index = random.randint(0,len(words)-1)
        except:
            return [[],[]*posNum]
        sub_word = words[sub_index]
        target_word = sub_word
        # print(sub_index, sub_word)

        synsets = wn.synsets(sub_word)
        temp = []
        for syn in synsets:
            temp += syn.lemma_names()
        if temp == []:
            continue
        for syn in temp:
            if syn != sub_word and syn != PorterStemmer().stem(sub_word) and syn not in sub_dict[sub_word]:
                target_word = syn
                break
        # print(temp)
        sub_dict[sub_word].append(target_word)
        target_word = target_word.split("_")
        # print(target_word)
        words1 = copy.deepcopy(words)
        del words1[sub_index]
        for j in range(len(target_word)):
            words1.insert(sub_index+j, target_word[j])
        data.append(words1)
        # print(words1)
        count+=1

    return data

def read_IMDB(subsetportion, posNum):
    # print("...transferring train and test set into documents...")
    all_docs = []
    train = []
    test = []

    dirs = ['./data/aclImdb_v1/aclImdb/train/pos/',
    './data/aclImdb_v1/aclImdb/train/neg/',
    './data/aclImdb_v1/aclImdb/test/pos/',
    './data/aclImdb_v1/aclImdb/test/neg/']
    files = [[dir + file for file in os.listdir(dir)] for dir in dirs]
    subsetlen = int(subsetportion*len(files[0]))
    labels = [[1]*subsetlen+[0]*subsetlen,[1]*subsetlen+[0]*subsetlen]

    words_save_file = './temp/word2vec/train_test_document.pickle'

    for i in range(len(files)):
        files[i] = files[i][:subsetlen]
        for file in tqdm(files[i]):
            with open(file, 'r', encoding='utf8') as f:
                review = f.readlines()
                
            if i == 0 or i == 1:
                words_words1 = substitude_word(review[0], posNum, IMDB=1)
                train+=words_words1
            else:
                words_words1 = substitude_word(review[0], posNum, test=1, IMDB=1)
                test+=words_words1

    all_docs = [train, test]
    with open(words_save_file, 'wb') as f:
        pickle.dump([all_docs, labels], f)
    return all_docs, labels

# 80% for training, 20% for testing
def read_Subj(subsetportion, posNum):
    # print("...transferring train and test set into documents...")

    words_save_file = './temp/word2vec/train_test_document.pickle'
    files = ['./data/rotten_imdb/plot.tok.gt9.5000.txt', './data/rotten_imdb/quote.tok.gt9.5000.txt']
    train_docs = []
    test_docs = []
    labeltrain = []
    labeltest = []
    for i, file in enumerate(files):
        with open(file, 'r', encoding='utf8') as f:
            doc = f.readlines()
            train = doc[:int(0.8*len(doc))]
            test = doc[int(0.8*len(doc))+1:]
            labeltrain += [1-i] * len(train)
            labeltest += [1-i] * len(test)
            train_docs += train
            test_docs += test
            # # print(labeltrain, len(labeltrain))
    labels = [labeltrain, labeltest]
    
    words_save_file = './temp/word2vec/train_test_document.pickle'
    train = []
    for sentence in tqdm(train_docs):
        sentence_sentence1 = substitude_word(str(sentence), posNum)
        train+=sentence_sentence1
    
    test = []
    for sentence in tqdm(test_docs):
        sentence_sentence1 = substitude_word(str(sentence), posNum, test=1)
        test+=(sentence_sentence1)

    all_docs = [train, test]

    print("train, test docs len={},{}, train, test labels len={},{}".format(len(train), len(test), len(labeltrain), len(labeltest)))
    # # print(labeltrain)
    with open(words_save_file, 'wb') as f:
        pickle.dump([all_docs, labels], f)
    return all_docs, labels

def read_TREC(subsetportion, posNum):
    # print("...transferring TREC train and test set into documents...")

    words_save_file = './temp/word2vec/train_test_document.pickle'
    files = ['./data/TREC/train.txt', './data/TREC/test.txt']

    with open(files[0], 'r', encoding='utf8') as f:
        train = f.readlines()
    with open(files[1], 'r', encoding='utf8') as f:
        test = f.readlines()
    
    labeltrain = []
    train_docs = []
    for i, data in enumerate(train):
        data = data.split(':')
        labeltrain.append(data[0])
        words = data[1].split()
        sentence_sentence1 = substitude_word(" ".join(words[1:]), posNum)
        if sentence_sentence1[0] == []:
            del labeltrain[i]
            continue
        train_docs+=sentence_sentence1

    labeltest = []
    test_docs = []
    for i, data in enumerate(test):
        data = data.split(':')
        labeltest.append(data[0])
        words = data[1].split()
        sentence_sentence1 = substitude_word(" ".join(words[1:]), posNum)
        if sentence_sentence1[0] == []:
            del labeltest[i]
            continue
        test_docs+=sentence_sentence1
    
    labels = Counter(labeltrain)
    labels, _ = zip(*labels.most_common())
    label2id = {l:i for i,l in enumerate(labels)}

    for i in range(len(labeltrain)):
        labeltrain[i] = label2id[labeltrain[i]]
    for i in range(len(labeltest)):
        labeltest[i] = label2id[labeltest[i]]

    all_labels = [labeltrain, labeltest]
    all_docs = [train_docs, test_docs]

    with open(words_save_file, 'wb') as f:
        pickle.dump([all_docs, all_labels], f)
    
    # print("label num:{}, trainset len:{}, testset len:{}".format(len(labels), len(train_docs), len(test_docs)))

    return all_docs, all_labels

def read_PC(subsetportion, posNum):
    # print("...transferring TREC train and test set into documents...")

    words_save_file = './temp/word2vec/train_test_document.pickle'
    files = ['./data/PC/IntegratedPros.txt', './data/PC/IntegratedCons.txt']

    with open(files[0], 'r', encoding='utf8') as f:
        pros = f.readlines()
    with open(files[1], 'r', encoding='utf8') as f:
        cons = f.readlines()

    train = []
    test = []
    labeltrain = []
    labeltest = []
    pro_train_count = 0
    pro_test_count = 0
    for i in range(len(pros)):
        if i < 0.8*len(pros):
            train.append(pros[i][14:-8])
            pro_train_count += 1
        else:
            test.append(pros[i][14:-8])
            pro_test_count += 1
    
    for i in range(len(cons)):
        if i < 0.8*len(cons):
            train.append(cons[i][14:-8])
        else:
            test.append(cons[i][14:-8])
    
    train_doc = []
    test_doc = []
    for i in tqdm(range(len(train))):
        words = train[i]
        sen_sen1 = substitude_word(words, posNum)
        if sen_sen1[0] == []:
            continue

        train_doc+=sen_sen1
        if i < pro_train_count:
            labeltrain.append(1)
        else:
            labeltrain.append(0)

    for i in tqdm(range(len(test))):
        words = test[i]
        sen_sen1 = substitude_word(words, posNum, test=1)
        if sen_sen1[0] == []:
            continue
        test_doc+=sen_sen1
        if i < pro_test_count:
            labeltest.append(1)
        else:
            labeltest.append(0)

    all_docs = [train_doc, test_doc]
    all_labels = [labeltrain, labeltest]

    print("train, test docs len={},{}, train, test labels len={},{}".format(len(train_doc), len(test_doc), len(labeltrain), len(labeltest)))
    with open(words_save_file, 'wb') as f:
        pickle.dump([all_docs, all_labels], f)
    
    return all_docs, all_labels

def read_CoLA(subset, posNum):
    words_save_file = './temp/word2vec/train_test_document.pickle'
    files = ['./data/cola_public_1.1/tokenized/in_domain_train.tsv', './data/cola_public_1.1/tokenized/in_domain_dev.tsv']
    
    with open(files[0], 'r', encoding='utf8') as f:
        train = f.readlines()
    with open(files[1], 'r', encoding='utf8') as f:
        test = f.readlines()
    
    labeltrain = []
    train_docs = []
    for i, data in enumerate(train):
        data = data.split('\t')
        words = data[3]
        sentence_sentence1 = substitude_word(words, posNum)
        if sentence_sentence1[0] == []:
            continue
        train_docs += sentence_sentence1
        labeltrain.append(int(data[1]))
    
    labeltest = []
    test_docs = []
    for i, data in enumerate(test):
        data = data.split('\t')
        words = data[3]
        sentence_sentence1 = substitude_word(words, posNum)
        if sentence_sentence1[0] == []:
            continue
        test_docs+=sentence_sentence1
        labeltest.append(int(data[1]))
    
    all_labels = [labeltrain, labeltest]
    all_docs = [train_docs, test_docs]

    with open(words_save_file, 'wb') as f:
        pickle.dump([all_docs, all_labels], f)
    
    return all_docs, all_labels


# sentence = "the movie begins in the past where a young boy named sam attempts to save celebi from a hunter . "
# sens = substitude_word(sentence, 10)
# print(sens)