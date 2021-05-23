# import torch
import pickle
import os
from re import sub

from numpy.core.fromnumeric import size
from model.TextCNN import TextCnn
from model.BiLSTM import TextRNN
from config import Config
from matplotlib import pyplot as plt 
from tqdm import tqdm
import argparse
import time
import logging
import torch.nn.functional as F
import matplotlib as mpl
import numpy as np
import math
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
# mpl.rcParams['axes.unicode_minus'] = False 
# mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--resultnum', type=str, default='0')
parser.add_argument('--modelname', type=str, default='TextCNN')
args = parser.parse_args()

def test():
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建handler
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = 'logs/'
    logfile = log_path+rq+args.modelname+"_"+args.dataset+"_4.log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG) 
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    #定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(console)


    arr = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    mode = 'test'
    embeddings_save_files = './temp/word2vec/'+args.dataset+'_test.pickle'
    model_list = os.listdir('./model/trained_model/')

    config = Config()
    if args.modelname == 'TextCNN':
        model = TextCnn(config, 'cpu')
    elif args.modelname == 'TextRNN':
        model = TextRNN(config)


    with open(embeddings_save_files, 'rb') as f:
        testset0 = pickle.load(f) #n*1*sentence_length*embedding_dim

    testset = testset0[0]
    label = testset0[1]
    lens = torch.tensor([len(i[0]) for i in testset])
    testset_tensor = torch.zeros(len(testset), max(lens), len(testset[0][0][0]))
    for i, mat in enumerate(testset):
        for j in range(lens[i]):
            testset_tensor[i][j] = torch.Tensor(testset[i][0][j])
    testset_tensor = testset_tensor.unsqueeze(1)

    pretrain = []
    ori = []
    aug = []
    with torch.no_grad():
        for (i, model_name) in tqdm(enumerate(model_list)):
            name = model_name.split("_")
            if name[0] == "model":
                checkpoint = torch.load('model/trained_model/'+model_name, map_location=torch.device('cuda:0'))
                if args.modelname == 'TextCNN':
                    model = TextCnn(checkpoint['para'], 'cpu')
                elif args.modelname == 'TextRNN':
                    model = TextRNN(checkpoint['para'])
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(torch.load('model/trained_model/'+model_name, map_location=torch.device('cuda:0')))
            
            out = model(testset_tensor, lens)
            preds = F.softmax(out, dim=1).argmax(1)
            acc = (torch.eq(preds, torch.Tensor(label))).float().mean()

            if name[0] == "model":
                pretrain=round(acc.item(), 4)
            else:
                if len(name)==2:
                    aug=round(acc.item(), 4)
                else:
                    ori=round(acc.item(), 4)
        print(pretrain, aug, ori)
    # resultnum = len(os.listdir('./results/'))
    # logger.info("file:r{}.txt,(pretrain, ori, aug)=({}, {}, {})".format(resultnum+1, pretrain, ori, aug))

    # with open("results/r"+str(resultnum+1)+".txt", 'w') as f:
    #     # f.write(pretrain)
    #     # f.write("\n")
    #     f.write(str(ori))
    #     f.write("\n")
    #     f.write(str(aug))


def draw_figure():
    # subset_list = [5,10,20,30,40,50,60,70,80,90,100]
    subset_list = [100]
    # resultnum = len(os.listdir('./results/'))
    resultnum = '130'
    with open("results/r"+str(resultnum)+".txt", 'r') as f:
        data = f.readlines()
    model1_loss = data[0].rstrip('\n').split(",")
    model1_acc = data[1].rstrip('\n').split(",")

    origin_loss = data[2].rstrip('\n').split(",")
    origin_acc = data[3].rstrip('\n').split(",")
    origin_train_acc = data[4].rstrip('\n').split(",")
    
    aug_loss = data[5].rstrip('\n').split(",")
    aug_acc = data[6].rstrip('\n').split(",")
    aug_train_acc = data[7].rstrip('\n').split(",")
    
    modelname = data[-1].split(',')[0]
    dataset = data[-1].rstrip('\n').split(',')[1]

    print(model1_loss)
    if model1_loss != ['']:
        model1_loss = [float(data) for data in model1_loss]
        model1_acc = [float(data) for data in model1_acc]
    origin_loss = [float(data) for data in origin_loss]
    origin_acc = [float(data) for data in origin_acc]
    origin_train_acc = [float(data) for data in origin_train_acc]
    aug_loss = [float(data) for data in aug_loss]
    aug_acc = [float(data) for data in aug_acc]
    aug_train_acc = [float(data) for data in aug_train_acc]

    origin_loss1 = []
    origin_acc1 = []
    origin_train_acc1 = []
    aug_loss1 = []
    aug_acc1 = []
    aug_train_acc1 = []
    subsetnum = len(subset_list)

    index = 10

    for i in range(len(subset_list)):
        origin_loss1.append((origin_loss[i*int(len(origin_loss)/subsetnum):(i+1)*int(len(origin_loss)/subsetnum)]))
        origin_acc1.append(np.mean(origin_acc[i*int(len(origin_acc)/subsetnum):(i+1)*int(len(origin_acc)/subsetnum)]))
        origin_train_acc1.append(np.mean(origin_train_acc[i*int(len(origin_train_acc)/subsetnum):(i+1)*int(len(origin_train_acc)/subsetnum)]))
        aug_loss1.append((aug_loss[i*int(len(aug_loss)/subsetnum):(i+1)*int(len(aug_loss)/subsetnum)]))
        aug_acc1.append(np.mean(aug_acc[i*int(len(aug_acc)/subsetnum):(i+1)*int(len(aug_acc)/subsetnum)]))
        aug_train_acc1.append(np.mean(aug_train_acc[i*int(len(aug_train_acc)/subsetnum):(i+1)*int(len(aug_train_acc)/subsetnum)]))

        if i == index:
            test_acc = origin_acc[i*int(len(origin_acc)/subsetnum):(i+1)*int(len(origin_acc)/subsetnum)]
            aug_test_acc = aug_acc[i*int(len(aug_acc)/subsetnum):(i+1)*int(len(aug_acc)/subsetnum)]

    # print(origin_loss1)
    origin_loss1 = np.array(origin_loss1)
    aug_loss1 = np.array(aug_loss1)


    # plt.subplots()
    plt.figure(figsize=(16,8))

    # plt.subplot(121)
    # plt.title("测试集上平均正确率随使用数据集比例变化曲线("+modelname+","+dataset+")", fontsize=15)
    # plt.xlabel("数据集使用比例(%)", fontsize=15)
    # plt.ylabel("test accuracy", fontsize=15)
    # plt.plot(subset_list, origin_acc1, 'x-',label="without pretraining", color='C0')
    # plt.plot(subset_list, aug_acc1, '+-',label="with pretraining", color='C1')
    # plt.ylim((0.5,1))
    # plt.legend()

    # print("origin mean:{}, aug mean:{}".format(np.mean(origin_acc1),np.mean(aug_acc1)))

    # plt.subplot(122)
    # plt.title("训练集上正确率随使用数据集比例变化曲线("+modelname+","+dataset+")", fontsize=15)
    # plt.xlabel("数据集使用比例(%)", fontsize=15)
    # plt.ylabel("train accuracy", fontsize=15)
    # plt.plot(subset_list, origin_train_acc1, 'x-', label="without pretraining")
    # plt.plot(subset_list, aug_train_acc1, '+-', label="with pretraining")
    # plt.ylim((0,1))

    plt.subplot(121)
    # plt.fill_between([i+1 for i in range(origin_loss1.shape[1])], origin_loss1.min(axis=0), origin_loss1.max(axis=0), color='C0')
    # plt.fill_between([i+1 for i in range(aug_loss1.shape[1])], aug_loss1.min(axis=0), aug_loss1.max(axis=0), color='C1')
    plt.plot([i+1 for i in range(origin_loss1.shape[1])], origin_loss1[0], color='C0')
    plt.plot([i+1 for i in range(aug_loss1.shape[1])], aug_loss1[0], color='C1')
    # plt.ylim(0,500)

    plt.legend()
    plt.savefig("results/picture/"+ str(resultnum) +".png")
    plt.show()

def remake_data():
    path = 'results/202105191451_103_TextCNN_PC_3.log'
    with open(path, 'r') as f:
        data = f.readlines()
    
    sign = 'classify.py[line:184]'
    total_loss_list = []
    total_acc_list = []
    total_train_acc_list = []

    for i in range(len(data)):
        data[i] = data[i].split()
        if data[i][3] == sign:
            total_loss_list.append(data[i][9].rstrip(','))
            total_acc_list.append(data[i][21].rstrip(','))
            total_train_acc_list.append(data[i][15].rstrip(','))
    
    total_loss_list = ",".join(total_loss_list)
    total_acc_list = ",".join(total_acc_list)
    total_train_acc_list = ",".join(total_train_acc_list)
    
    # print(total_loss_list)
    # print(total_acc_list)
    # print(total_train_acc_list)

    resultnum = '103'
    dataset = "PC"
    model = "TextCNN"
    with open("results/r"+str(resultnum)+".txt", 'a') as f:
        f.write(total_loss_list+"\n"+total_acc_list+"\n"+total_train_acc_list+"\n"+model+","+dataset)

def pos_neg():
    resultnum = '121'
    with open("results/r"+str(resultnum)+".txt", 'r') as f:
        data = f.readlines()
    # pos_neg_list = [2,4,6,8,10,12]
    pos_neg_list = [[1,7],[2,6],[3,5],[4,4],[5,3],[6,2],[7,1]]
    aver_acc = []
    index = [2,3,4,8,9,10]
    print(len(data))
    plt.figure(figsize=(14,5))
    for i,n in enumerate(pos_neg_list):
        print(i,2*i+1)
        acc = data[2*i+1].rstrip('\n').split(',')
        for j in range(len(acc)):
            acc[j] = float(acc[j])
        aver_acc.append(np.mean(acc))
        x = [k+1 for k in range(len(acc))]
        plt.subplot(2,6,index[i])
        plt.plot(x, acc)
        plt.ylim(0,1)
        plt.xlabel("epoch")
        plt.ylabel("test accuracy")
        plt.title("posNum={}".format(n))
        plt.tight_layout(1)
        # plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(pos_neg_list, aver_acc)
    plt.ylim(0,1)
    plt.xlabel("posNum")
    plt.ylabel("average test accuracy")
    plt.title("posNum对平均test集正确率的影响")
    plt.show()

def pos_neg_portion():
    resultnum = '128'
    with open("results/r"+str(resultnum)+".txt", 'r') as f:
        data = f.readlines()
    # pos_neg_list = [2,4,6,8,10,12]
    pos_neg_list = ["[1,7]","[2,6]","[3,5]","[4,4]","[5,3]","[6,2]","[7,1]"]
    data = [data[1].rstrip().split(","),data[3].rstrip().split(","),data[5].rstrip().split(","),data[7].rstrip().split(","),data[9].rstrip().split(","),data[11].rstrip().split(","),data[13].rstrip().split(",")]
    data = [[float(data[i][j]) for j in range(len(data[i]))] for i in range(len(data))]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.boxplot(data, showfliers=False)
    plt.ylim(0.3,0.7)
    
    # plt.subplot(1,3,3)
    # plt.plot(pos_neg_list, aver_acc)
    # plt.ylim(0,1)
    plt.xlabel("posNum与negNum比例", fontsize=15)
    plt.ylabel("test accuracy", fontsize=15)
    plt.title("posNum与negNum比例对平均test集正确率的影响", fontsize=15)
    plt.setp(ax, xticks=[1,2,3,4,5,6,7], xticklabels=pos_neg_list)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.savefig("results/picture/"+ str(resultnum) +".png")
    plt.show()

# remake_data()
draw_figure()
# test()
# pos_neg()
# pos_neg_portion()